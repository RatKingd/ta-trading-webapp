# data_fetcher.py
import time
import requests
import pandas as pd
import yfinance as yf

# כותרות שיעזרו לעקוף חסימות/CDN של TASE
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# רשימות גיבוי אם TASE לא מחזיר כלום (TA-125 / TA-35)
FALLBACK_TA125 = ["TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ICL.TA"]
FALLBACK_TA35  = ["TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ICL.TA"]


def get_tickers_from_tase(index_url: str, timeout: int = 20) -> list[str]:
    """
    קורא את דף המדד מ-TASE ומחלץ סימבולים. במקרה של כישלון — יחזיר רשימת fallback.
    """
    try:
        r = requests.get(index_url, headers=HDRS, timeout=timeout)
        r.raise_for_status()
        dfs = pd.read_html(r.text)
        out: list[str] = []
        for df in dfs:
            cols = [str(c).lower() for c in df.columns]
            # עמודות אפשריות
            candidates = [c for c in df.columns if str(c).lower() in
                          ("instrument symbol", "symbol", "ticker", "סימול")]
            if candidates:
                symcol = candidates[0]
                syms = [str(x).strip() for x in df[symcol].dropna().tolist()]
                # הוסף .TA אם חסר
                out += [s if s.endswith(".TA") else f"{s}.TA" for s in syms if s]
        out = sorted(list({s for s in out if s.endswith(".TA")}))
        if out:
            return out[:100]  # נגביל קצת
    except Exception:
        pass

    # fallback לפי סוג הדף
    if "index/142" in index_url or "TA-35" in index_url:
        return FALLBACK_TA35
    return FALLBACK_TA125


def download_price_history(tickers: list[str], horizon: str = "daily") -> dict[str, pd.Series]:
    """
    מוריד היסטוריית מחירים מ-Yahoo עבור רשימת טיקרים ומחזיר {ticker: Series(Adj Close)}.
    תומך ביומי/שבועי/חודשי, כולל ריטריי ונרמול MultiIndex.
    """
    horizon_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    interval = horizon_map.get(horizon, "1d")

    out: dict[str, pd.Series] = {}

    df = pd.DataFrame()
    for _ in range(3):
        df = yf.download(
            tickers=tickers,
            period="3y",
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            break
        time.sleep(2)

    if df.empty:
        return out

    # MultiIndex: (TICKER, 'Adj Close')
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            col = (t, "Adj Close")
            if col in df.columns:
                ser = df[col].dropna()
                if not ser.empty:
                    out[t] = ser
    else:
        # יחיד
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        ser = df.get(col, pd.Series(dtype="float64")).dropna()
        if len(tickers) == 1 and not ser.empty:
            out[tickers[0]] = ser

    return out
