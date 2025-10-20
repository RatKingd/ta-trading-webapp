import requests
import pandas as pd
import yfinance as yf

# כותרות כדי להקטין סיכוי לחסימת אתר
_HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# רשימות נפילה (Fallback) אם שליפת רכיבי המדד נכשלת
FALLBACK_TA35 = ["TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ICL.TA"]
FALLBACK_TA125 = FALLBACK_TA35  # אפשר להרחיב בעתיד

def get_tickers_from_tase(index_url: str, timeout=20) -> list[str]:
    """
    מנסה לשלוף את טבלאות רכיבי המדד מאתר הבורסה (קישור רכיבי המדד).
    אם נכשל – מחזיר רשימת נפילה.
    """
    try:
        r = requests.get(index_url, headers=_HDRS, timeout=timeout)
        r.raise_for_status()
        dfs = pd.read_html(r.text)  # דורש lxml
        out = []
        for df in dfs:
            cols = [str(c).lower() for c in df.columns]
            # ננסה עמודות סבירות לשם הנייר
            for candidate in ("Instrument Symbol", "Symbol", "Ticker", "נייר"):
                if candidate in df.columns:
                    syms = [str(x).strip() for x in df[candidate].dropna().tolist()]
                    out = [s if s.endswith(".TA") else f"{s}.TA" for s in syms]
                    break
            if out:
                break

        if not out:
            raise ValueError("no symbols column found")

        # ננקה כפילויות/ריקים
        out = sorted({s for s in out if s and s != "nan"})
        # נגביל ל-100 כדי לא להעמיס
        return out[:100]
    except Exception as e:
        print("TASE fetch failed:", repr(e))
        # קבע עפ"י ה-URL איזה נפילה לבחור
        if "index/142" in index_url or "TA-35" in index_url:
            return FALLBACK_TA35
        return FALLBACK_TA125

def download_price_history(tickers: list[str], period="1y", interval="1d") -> dict[str, pd.DataFrame]:
    """
    מוריד היסטוריית מחירים מ-Yahoo. בלי פרמטר weekly/דומיו כדי למנוע שגיאה.
    מחזיר dict: ticker -> DataFrame עם עמודות ['Open','High','Low','Close','Adj Close','Volume'] ועמודת Date.
    """
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is None or df.empty:
                continue
            df = df.reset_index().rename(columns={"Date": "Date"})
            # נעגל נפחים/מחירים לשקיפות
            for c in ("Open", "High", "Low", "Close", "Adj Close"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            out[t] = df
        except Exception as e:
            print("yfinance failed for", t, ":", repr(e))
    return out
