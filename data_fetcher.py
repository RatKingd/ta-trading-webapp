import requests, pandas as pd, yfinance as yf
from tqdm import tqdm

# ---- רשימות גיבוי קצרות (אפשר להרחיב בהמשך) ----
FALLBACK_TA35  = ["TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ICL.TA"]
FALLBACK_TA125 = FALLBACK_TA35  # אפשר להרחיב לרשימה גדולה יותר בהמשך

# כותרות כדי לא להיחסם ע"י TASE/CDN
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def get_tickers_from_tase(index_url: str) -> list[str]:
    """
    קורא את טבלת רכיבי המדד מ-TASE ומחזיר סימבולים בפורמט Yahoo (.TA).
    אם הקריאה נחסמת/נכשלת — חוזר לרשימת גיבוי.
    """
    try:
        r = requests.get(index_url, headers=HDRS, timeout=20)
        r.raise_for_status()
        dfs = pd.read_html(r.text)  # קורא את הטבלאות מה-HTML שהורדנו
        for df in dfs:
            cols = [str(c).lower() for c in df.columns]
            if any('symbol' in c or 'instrument' in c or 'ticker' in c for c in cols):
                for candidate in ['Instrument Symbol','Symbol','Instrument','Ticker']:
                    if candidate in df.columns:
                        syms = [str(x).strip() for x in df[candidate].dropna().tolist()]
                        out = [s + ".TA" for s in syms if s]
                        if out:
                            return out
                # נפילה לעמודה ראשונה אם אין שמות עמודות סטנדרטיים
                syms = [str(x).strip() for x in df.iloc[:,0].dropna().tolist()]
                out = [s + ".TA" for s in syms if s]
                if out:
                    return out
        raise RuntimeError("No component table found")
    except Exception as e:
        print("TASE fetch failed:", e)
        # גיבוי לפי ה-URL שנבחר
        if "index/142" in index_url:   # TA-35
            return FALLBACK_TA35
        if "index/168" in index_url:   # TA-125
            return FALLBACK_TA125
        return FALLBACK_TA35

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
