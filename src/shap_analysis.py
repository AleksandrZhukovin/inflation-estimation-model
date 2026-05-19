from pathlib import Path

import numpy as np
import pandas as pd


def compute_shap_values(model, X):
    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    shap_values = explanation.values

    return explanation, shap_values


def global_feature_ranking(shap_values, feature_names):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs})
    return df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)


def export_shap_dataframe(shap_values, feature_names, X):
    cols = {f"{f}_shap": shap_values[:, i] for i, f in enumerate(feature_names)}
    return pd.DataFrame(cols, index=X.index).reset_index(drop=True)


# run


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema
    from src.features import build_feature_matrix, compute_arima_residuals
    from src.hybrid import _build_test_features
    from src.xgboost_model import load_xgboost_model

    models_dir = Path(cfg.outputs.models_dir)
    predictions_dir = Path(cfg.outputs.predictions_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "shap"

    for d in (predictions_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    xgb_model = load_xgboost_model(models_dir / "xgboost_final.pkl")
    arima_results = {
        country: load_arima_model(models_dir / f"arima_{country.lower()}.pkl")
        for country in cfg.data.countries
    }

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, test_df = split_train_test(
        df,
        cfg.data.train_end,
        cfg.data.test_start,
        test_countries=cfg.data.test_countries,
        test_end=cfg.data.test_end,
    )
    assert train_df["Date"].max() < test_df["Date"].min(), "Train/test leakage detected"

    residuals_df = compute_arima_residuals(arima_results)
    _, _, feature_names = build_feature_matrix(train_df, residuals_df, cfg)

    X_test = _build_test_features(test_df, cfg, feature_names)

    explanation, shap_values = compute_shap_values(xgb_model, X_test)

    ranking_df = global_feature_ranking(shap_values, feature_names)
    ranking_df.to_csv(tables_dir / "global_feature_ranking.csv", index=False)

    shap_df = export_shap_dataframe(shap_values, feature_names, X_test)
    shap_path = predictions_dir / "shap_values_test.csv"
    shap_df.to_csv(shap_path, index=False)

    print("\n[SHAP] Done.\n")
    print("  Top-5 features by |SHAP|:")
    for _, row in ranking_df.head(5).iterrows():
        print(f"    {row['Feature']:30s} {row['MeanAbsSHAP']:.4f}")
    print(f"\n  Tables  → {tables_dir}")
    print(f"  SHAP    → {shap_path}\n")
