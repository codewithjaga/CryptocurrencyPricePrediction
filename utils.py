import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(symbol, period="2y"):
    df = yf.download(symbol, period=period)

    # 🔥 FLATTEN MULTIINDEX COLUMNS
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    return df
def add_indicators(df):
    df = df.copy()

    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(20).std()

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    return df


def annual_volatility(df):
    return df['Return'].std() * np.sqrt(252)
