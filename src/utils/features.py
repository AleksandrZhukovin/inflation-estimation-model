"""
Features related utils

Feature matrix building and test features preparation
for use in residual correction XGBoost model.
"""

import numpy as np
import pandas as pd


def build_feature_matrix(train_df, residuals_df, cfg):
    entity_col = getattr(
        cfg.data, "entity_column", getattr(cfg.data, "country_column", "Country")
    )
    drop_always = {
        cfg.data.date_column,
        cfg.data.target_column,
        *cfg.data.flag_columns,
    }

    merged = train_df.merge(
        residuals_df[["Date", entity_col, "arima_residual"]],
        on=["Date", entity_col],
        how="inner",
    )

    assert len(merged) > 0, (
        "Feature matrix merge returned 0 rows — check that train_df dates match residuals dates"
    )

    merged = merged.sort_values([cfg.data.date_column, entity_col])

    macro_cols = [
        c
        for c in merged.columns
        if c not in drop_always and c not in (entity_col, "arima_residual")
    ]
    merged = merged.copy()
    merged[macro_cols] = merged.groupby(entity_col)[macro_cols].shift(1)
    merged = merged.dropna(subset=macro_cols).reset_index(drop=True)

    _dates = pd.to_datetime(merged[cfg.data.date_column])
    merged["Month_sin"] = np.sin(2 * np.pi * _dates.dt.month / 12)
    merged["Month_cos"] = np.cos(2 * np.pi * _dates.dt.month / 12)
    merged["Year_trend"] = (_dates.dt.year - 2004) / 20.0

    y = merged["arima_residual"].reset_index(drop=True)

    feature_df = merged.drop(
        columns=[c for c in drop_always if c in merged.columns] + ["arima_residual"],
    )

    entity_dummies = pd.get_dummies(
        feature_df[entity_col], prefix=entity_col, dtype=float
    )
    feature_df = feature_df.drop(columns=[entity_col])
    X = pd.concat([feature_df, entity_dummies], axis=1).reset_index(drop=True)

    feature_names = list(X.columns)

    return X, y, feature_names


def build_test_features(test_df, cfg, feature_names, last_train_df=None):
    entity_col = getattr(
        cfg.data, "entity_column", getattr(cfg.data, "country_column", "Country")
    )
    drop_always = {
        cfg.data.date_column,
        cfg.data.target_column,
        *cfg.data.flag_columns,
    }
    macro_cols = [
        c for c in test_df.columns if c not in drop_always and c != entity_col
    ]

    working = test_df.copy()

    if last_train_df is not None:
        parts = []
        for entity in working[entity_col].unique():
            last_row = (
                last_train_df[last_train_df[entity_col] == entity]
                .sort_values(cfg.data.date_column)
                .iloc[[-1]]
            )
            entity_test = working[working[entity_col] == entity].sort_values(
                cfg.data.date_column
            )
            combined = pd.concat([last_row, entity_test], ignore_index=True)
            for col in macro_cols:
                combined[col] = combined[col].shift(1)
            parts.append(combined.iloc[1:])
        working = pd.concat(parts, ignore_index=True)

    _dates = pd.to_datetime(working[cfg.data.date_column])
    working["Month_sin"] = np.sin(2 * np.pi * _dates.dt.month / 12)
    working["Month_cos"] = np.cos(2 * np.pi * _dates.dt.month / 12)
    working["Year_trend"] = (_dates.dt.year - 2004) / 20.0

    entity_dummies = pd.get_dummies(working[entity_col], prefix=entity_col, dtype=float)
    working = working.drop(columns=[entity_col])
    working = pd.concat([working, entity_dummies], axis=1)
    working = working.drop(columns=[c for c in drop_always if c in working.columns])

    for col in feature_names:
        if col not in working.columns:
            working[col] = 0.0

    # Ensure no extra columns that are not in feature_names are present
    working = working[feature_names]

    return working.reset_index(drop=True)
