import pandas as pd
import eurostat

from src.inflation_estimation.data.constants import EU_MANUAL_COLUMNS


def get_eurostat_monthly(code, filter_pars, value_name):
    df = eurostat.get_data_df(code, filter_pars=filter_pars)

    # Find date columns and metadata columns
    date_cols = [c for c in df.columns if "M" in str(c) and c[0].isdigit()]
    meta_cols = [c for c in df.columns if c not in date_cols]

    df_melt = df.melt(
        id_vars=meta_cols, value_vars=date_cols, var_name="Date", value_name="Value"
    )

    # Force conversion to numbers. Non-numeric text becomes NaN (empty)
    df_melt["Value"] = pd.to_numeric(df_melt["Value"], errors="coerce")

    df_melt["Date"] = pd.to_datetime(
        df_melt["Date"].str.replace("M", "-"), format="%Y-%m"
    )

    geo_col = next((c for c in meta_cols if "geo" in c.lower()), None)

    df_pivot = df_melt.pivot(index="Date", columns=geo_col, values="Value")
    df_pivot.columns = [f"{c}_{value_name}" for c in df_pivot.columns]

    # Convert index to DatetimeIndex explicitly (avoids future warnings)
    df_pivot.index = pd.to_datetime(df_pivot.index)

    return df_pivot


def get_eurostat_quarterly(code, filter_pars, value_name):
    """
    Separate function to get data that is presented quarterly and interpolate it to monthly.
    :param code: Country code like LT, LV etc.
    :param filter_pars: Eurostat specific dict.
    :param value_name: The name to put as column title.
    :return: pandas.DataFrame
    """
    df = eurostat.get_data_df(code, filter_pars=filter_pars)

    # Find quarterly columns
    date_cols = [c for c in df.columns if "Q" in str(c) and c[0].isdigit()]
    meta_cols = [c for c in df.columns if c not in date_cols]

    df_melt = df.melt(
        id_vars=meta_cols, value_vars=date_cols, var_name="Date", value_name="Value"
    )

    # Force conversion to numbers. Non-numeric text becomes NaN
    df_melt["Value"] = pd.to_numeric(df_melt["Value"], errors="coerce")

    # Convert "2023Q1" to "2023-01-01"
    def q_to_date(x):
        if not isinstance(x, str):
            return pd.NaT
        try:
            y, q = x.split("Q")
            return pd.to_datetime(f"{y}-{int(q) * 3 - 2:02d}-01")
        except:  # noqa: E722
            return pd.NaT

    df_melt["Date"] = df_melt["Date"].apply(q_to_date)

    geo_col = next((c for c in meta_cols if "geo" in c.lower()), None)
    df_pivot = df_melt.pivot(index="Date", columns=geo_col, values="Value")
    df_pivot.columns = [f"{c}_{value_name}" for c in df_pivot.columns]

    # Convert index to DatetimeIndex explicitly
    df_pivot.index = pd.to_datetime(df_pivot.index)

    # Now Interpolation will work because data is numeric float
    df_monthly = df_pivot.resample("MS").interpolate(method="linear")

    return df_monthly


def create_europe_partition(country_code):
    print(f"Creating {country_code} dataset...")
    # Inflation (HICP) - Monthly
    cpi_filter = {"geo": [country_code], "coicop": "CP00", "unit": "I15", "freq": "M"}
    df_cpi = get_eurostat_monthly("prc_hicp_midx", cpi_filter, "CPI_Index")

    # Unemployment - Monthly
    unemp_filter = {
        "geo": [country_code],
        "age": "TOTAL",
        "unit": "PC_ACT",
        "s_adj": "SA",
    }
    df_unemp = get_eurostat_monthly("une_rt_m", unemp_filter, "Unemployment")

    # GDP - Quarterly (interpolated)
    gdp_filter = {
        "geo": [country_code],
        "unit": "CLV_I10",
        "na_item": "B1GQ",
        "s_adj": "SCA",
    }
    df_gdp = get_eurostat_quarterly("namq_10_gdp", gdp_filter, "GDP_Index")

    # Government Debt - Quarterly (interpolated)
    debt_filter = {
        "geo": [country_code],
        "unit": "PC_GDP",
        "sector": "S13",
        "na_item": "GD",
    }
    df_debt = get_eurostat_quarterly("gov_10q_ggdebt", debt_filter, "Gov_Debt")

    # Merge all country-specific data
    dfs_to_merge = [df_cpi, df_unemp, df_gdp, df_debt]
    df = pd.concat(dfs_to_merge, axis=1)

    # Rename columns to be generic (remove country prefix)
    df.columns = [col.replace(f"{country_code}_", "") for col in df.columns]

    df["Country"] = country_code
    df["Official_Currency_to_EUR"] = 1.0

    for col in EU_MANUAL_COLUMNS:
        df[col] = ""

    return df
