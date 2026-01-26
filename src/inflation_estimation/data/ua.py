from typing import assert_never

import yfinance as yf
import pandas as pd

from constants import START_DATE, UA_MANUAL_COLUMNS


def get_uah_rate(currency_code):
    if currency_code == "USD":
        sym = "UAH=X"
    elif currency_code == "EUR":
        sym = "EURUAH=X"
    else:
        assert_never(currency_code)

    df = yf.download(sym, start=START_DATE, interval="1mo")["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df


def create_ua_partition():
    uah_usd = get_uah_rate("USD").rename("Official_Currency_to_USD")
    uah_eur = get_uah_rate("EUR").rename("Official_Currency_to_EUR")

    df = pd.concat([uah_usd, uah_eur], axis=1)
    df["Country"] = "UA"

    for col in UA_MANUAL_COLUMNS:
        df[col] = ""

    return df
