import pandas as pd
import yfinance as yf

from config import DATASET_PATH
from constants import END_DATE, START_DATE
from europe import create_europe_partition
from ua import create_ua_partition


def create_global_partition():
    """
    Function to create base dataset of global metrics to merge with countries specific ones later.
    :return: pandas.DataFrame
    """
    tickers = {
        "BZ=F": "Brent_Oil",
        "GC=F": "Gold_Price",
        "^VIX": "VIX_Index",
        "EUR=X": "EUR_USD_Rate",
    }
    df = yf.download(list(tickers.keys()), start=START_DATE, interval="1mo")

    if "Close" in df.columns.levels[0]:
        df = df["Close"]
    else:
        df = df["Adj Close"] if "Adj Close" in df else df["Close"]

    df = df.rename(columns=tickers)
    df.index = df.index.to_period("M").to_timestamp()
    return df


# Generate dataset partitions
df_global = create_global_partition()

df_ua = create_ua_partition()
df_ua = df_ua.join(df_global, how="outer")

df_lt = create_europe_partition("LT")
df_lt = df_lt.join(df_global, how="outer")
df_lt["Official_Currency_to_USD"] = df_lt["EUR_USD_Rate"]

df_lv = create_europe_partition("LV")
df_lv = df_lv.join(df_global, how="outer")
df_lv["Official_Currency_to_USD"] = df_lv["EUR_USD_Rate"]

# Merge all country-specific and global data
master_df = pd.concat([df_ua, df_lt, df_lv], ignore_index=False)
master_df = master_df.drop(columns=["EUR_USD_Rate"])
master_df = master_df[(master_df.index >= START_DATE) & (master_df.index <= END_DATE)]
master_df.index.name = "Date"

# Add columns for manual filling
manual_columns = ["Global_GPR_Index"]
for col in manual_columns:
    master_df[col] = ""

# Reorder columns for better readability
all_cols = list(master_df.columns)
# Remove key columns to re-insert them at the beginning
key_cols = [
    "Country",
    "Official_Currency_to_USD",
    "Official_Currency_to_EUR",
    "CPI_Index",
    "Key_Rate",
]
for col in key_cols:
    if col in all_cols:
        all_cols.remove(col)

new_order = key_cols + sorted(all_cols)
master_df = master_df[new_order]

# Save data file
output_file = DATASET_PATH
master_df.to_csv(output_file)
