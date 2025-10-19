import pandas as pd, numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['ret1'] = d['Adj Close'].pct_change(1)
    d['ret5'] = d['Adj Close'].pct_change(5)
    d['sma20'] = d['Adj Close'].rolling(20).mean()
    d['sma50'] = d['Adj Close'].rolling(50).mean()
    d['sma_ratio'] = d['sma20'] / d['sma50']
    delta = d['Adj Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    d['rsi'] = 100 - 100/(1 + (up / down))
    return d.dropna()
