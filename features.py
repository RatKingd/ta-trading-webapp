import pandas as pd

def add_indicators(df: pd.DataFrame, price_col="Adj Close", rsi_window=14, sma_fast=20, sma_slow=50) -> pd.DataFrame:
    """
    מקבל DF עם Date ו-Adj Close (או price_col) ומחזיר DF עם אינדיקטורים בסיסיים:
    RSI(14), SMA20, SMA50 ותשואות יומיות.
    """
    d = df.copy()
    d = d.sort_values("Date").reset_index(drop=True)
    p = d[price_col].astype(float)

    # תשואות
    d["ret1"] = p.pct_change()
    # RSI
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_window, adjust=False).mean()
    rs = gain / (loss.replace(0, 1e-9))
    d["rsi"] = 100 - (100 / (1 + rs))

    # ממוצעים
    d["sma20"] = p.rolling(sma_fast).mean()
    d["sma50"] = p.rolling(sma_slow).mean()

    return d
