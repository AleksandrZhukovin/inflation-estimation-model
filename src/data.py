import pandas as pd

REQUIRED_COLUMNS = [
    "Date",
    "Country",
    "CPI_Index",
    "Official_Currency_to_USD",
    "Official_Currency_to_EUR",
    "Key_Rate",
    "Brent_Oil",
    "GDP_yoy",
    "Global_GPR_Index",
    "Gold_Price",
    "VIX_Index",
    "Unemployment",
    "REER",
    "Money_Supply_yoy",
    "Metals_Index",
    "FAO_FFPI",
    "Industrial_Production_yoy",
]


def _fix_global_gpr(df):
    # UA rows contain a country-specific GPR variant, not the global
    # Caldara-Iacoviello index. LT and LV carry the correct global series.
    gpr_global = df[df["Country"] == "LT"].set_index("Date")["Global_GPR_Index"]
    ua_mask = df["Country"] == "UA"
    df.loc[ua_mask, "Global_GPR_Index"] = df.loc[ua_mask, "Date"].map(gpr_global)
    return df


def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)
    df = _fix_global_gpr(df)
    return df


def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    for col in ("Date", "Country"):
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(
                f"Column '{col}' has {n_null} null values — check dataset integrity"
            )


def split_train_test(df, train_end, test_start, test_countries=None, test_end=None):
    train_df = df[df["Date"] <= pd.Timestamp(train_end)].reset_index(drop=True)

    test_df = df[df["Date"] >= pd.Timestamp(test_start)]
    if test_end:
        test_df = test_df[test_df["Date"] <= pd.Timestamp(test_end)]
    if test_countries:
        test_df = test_df[test_df["Country"].isin(test_countries)]
    return train_df, test_df.reset_index(drop=True)


def get_country_subset(df, country):
    return df[df["Country"] == country].reset_index(drop=True)


def get_country_train_data(df, country, cfg):
    subset = get_country_subset(df, country)
    train, _ = split_train_test(subset, cfg.data.train_end, cfg.data.test_start)
    train_start = (
        cfg.data.ua_train_start if country == "UA" else cfg.data.lt_lv_train_start
    )
    train = train[train["Date"] >= pd.Timestamp(train_start)].reset_index(drop=True)
    return train
