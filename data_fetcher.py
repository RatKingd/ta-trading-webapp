import requests, pandas as pd, yfinance as yf
from tqdm import tqdm

def get_tickers_from_tase(index_url: str) -> list[str]:
    """קורא את טבלת רכיבי המדד מ-TASE ומחזיר סמלים בפורמט Yahoo (.TA)."""
    r = requests.get(index_url, timeout=15)
    r.raise_for_status()
    dfs = pd.read_html(r.text)
    for df in dfs:
        cols = [str(c).lower() for c in df.columns]
        if any('symbol' in c or 'instrument' in c or 'ticker' in c for c in cols):
            for candidate in ['Instrument Symbol','Symbol','Instrument','Ticker']:
                if candidate in df.columns:
                    syms = [str(x).strip() for x in df[candidate].dropna().tolist()]
                    return [s + ".TA" for s in syms if s]
            syms = [str(x).strip() for x in df.iloc[:,0].dropna().tolist()]
            return [s + ".TA" for s in syms if s]
    return []

def download_price_history(tickers: list[str], start="2016-01-01", end=None) -> dict[str, pd.DataFrame]:
    all_data = {}
    for t in tqdm(tickers, desc="Downloading"):
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                df = df.dropna()
                all_data[t] = df
        except Exception as e:
            print("Download error:", t, e)
    return all_data
