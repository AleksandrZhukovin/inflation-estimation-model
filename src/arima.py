from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox


class ARIMAResult:
    def __init__(
        self,
        country,
        model,
        order,
        aic,
        bic,
        lb_pvalue,
        rmse_train,
        fitted_values,
        residuals,
        train_dates,
    ):
        self.country = country
        self.model = model
        self.order = order
        self.aic = aic
        self.bic = bic
        self.lb_pvalue = lb_pvalue
        self.rmse_train = rmse_train
        self.fitted_values = fitted_values
        self.residuals = residuals
        self.train_dates = train_dates


def _get_country_series(train_df, country, cfg):
    subset = train_df[train_df[cfg.data.country_column] == country].copy()
    train_start = (
        cfg.data.ua_train_start if country == "UA" else cfg.data.lt_lv_train_start
    )
    subset = (
        subset[subset[cfg.data.date_column] >= pd.Timestamp(train_start)]
        .sort_values(cfg.data.date_column)
        .reset_index(drop=True)
    )
    valid = subset[cfg.data.target_column].notna()
    subset = subset[valid].reset_index(drop=True)
    series = subset[cfg.data.target_column].values.astype(float)
    dates = pd.DatetimeIndex(subset[cfg.data.date_column])
    return series, dates


def fit_arima_for_country(country, train_df, cfg):
    series, dates = _get_country_series(train_df, country, cfg)

    model = pm.auto_arima(
        series,
        start_p=0,
        start_q=0,
        max_p=cfg.arima.max_p,
        max_q=cfg.arima.max_q,
        d=None,
        seasonal=cfg.arima.seasonal,
        information_criterion=cfg.arima.information_criterion,
        stepwise=cfg.arima.stepwise,
        suppress_warnings=True,
        error_action="ignore",
        random_state=cfg.project.random_seed,
    )

    order = model.order
    aic = float(model.aic())
    bic = float(model.bic())

    fitted_values = np.asarray(model.predict_in_sample(), dtype=float)
    n_obs = len(series)
    n = min(n_obs, len(fitted_values))
    residuals = series[-n:] - fitted_values[-n:]

    valid_mask = ~np.isnan(residuals)
    rmse_train = float(np.sqrt(np.mean(residuals[valid_mask] ** 2)))

    innov = np.asarray(model.arima_res_.resid, dtype=float)
    innov = innov[~np.isnan(innov)]
    max_lag = min(cfg.arima.ljung_box_lags, len(innov) // 5)
    lb_df = acorr_ljungbox(innov, lags=[max_lag], return_df=True)
    lb_pvalue = float(lb_df["lb_pvalue"].iloc[-1])

    return ARIMAResult(
        country=country,
        model=model,
        order=order,
        aic=round(aic, 2),
        bic=round(bic, 2),
        lb_pvalue=round(lb_pvalue, 4),
        rmse_train=round(rmse_train, 4),
        fitted_values=fitted_values,
        residuals=residuals,
        train_dates=dates[-n:],
    )


def predict_arima(result, n_periods):
    forecast = result.model.predict(n_periods=n_periods)
    return np.asarray(forecast, dtype=float)


def save_arima_model(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)


def load_arima_model(path):
    path = Path(path)
    return joblib.load(path)


# run


def run(cfg):
    from src.data import load_dataset, split_train_test, validate_schema

    models_dir = Path(cfg.outputs.models_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "arima"

    for d in (models_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, test_df = split_train_test(df, cfg.data.train_end, cfg.data.test_start)
    assert train_df["Date"].max() < test_df["Date"].min(), "Train/test leakage detected"

    results = {}
    for country in cfg.data.countries:
        result = fit_arima_for_country(country, train_df, cfg)
        results[country] = result
        save_arima_model(result, models_dir / f"arima_{country.lower()}.pkl")

    spec_rows = []
    for country, result in results.items():
        p, d, q = result.order
        spec_rows.append(
            {
                "Country": country,
                "p": p,
                "d": d,
                "q": q,
                "AIC": result.aic,
                "BIC": result.bic,
                "LjungBox_pvalue": result.lb_pvalue,
                "RMSE_train": result.rmse_train,
            }
        )
    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(tables_dir / "arima_specifications.csv", index=False)

    print("\n[ARIMA] Done.")
    print(
        f"  {'Country':<6}  {'Order':<12}  {'AIC':>8}  {'BIC':>8}  {'LB p':>8}  {'RMSE':>8}"
    )
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for _, row in spec_df.iterrows():
        order_str = f"({int(row.p)},{int(row.d)},{int(row.q)})"
        print(
            f"  {row.Country:<6}  {order_str:<12}  {row.AIC:>8.2f}  {row.BIC:>8.2f}"
            f"  {row.LjungBox_pvalue:>8.4f}  {row.RMSE_train:>8.4f}"
        )
    print(f"\n  Models  → {models_dir}")
    print(f"  Tables  → {tables_dir}\n")
