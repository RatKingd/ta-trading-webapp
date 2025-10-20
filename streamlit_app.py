# data_fetcher.py
# ---------------
# שליפת טיקרים מאתר TASE + הורדת היסטוריית מחירים מ-Yahoo בפורמט Wide:
# עמודות בצורה MultiIndex: ('Adj Close','TEVA.TA'), ('Volume','TEVA.TA'), ...

from __future__ import annotations
import re
import time
from typing import List

import requests
import pandas as pd
import yfinance as yf

# כותרות "אנושיות" כדי לעקוף חסימות/CDN
_HDRS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _extract_symbols_from_tables(dfs: list[pd.DataFrame]) -> list[str]:
    """מקבל רשימת טבלאות שנקראו מ-HTML ומחזיר סמלים עם סיומת .TA"""
    out: list[str] = []
    candidates = {"Instrument Symbol", "Symbol", "Ticker", "Instrument", "תוך מסחר", "סימול"}

    for df in dfs:
        if df is None or df.empty:
            continue

        # normalize headers
        cols = [str(c).strip() for c in df.columns]
        lower = [c.lower() for c in cols]

        # חיפוש עמודת סמל אפשרית
        target_idx = None
        for i, c in enumerate(cols):
            if c in candidates:
                target_idx = i
                break
        if target_idx is None:
            for i, c in enumerate(lower):
                if "symbol" in c or "ticker" in c or "סימול" in c:
                    target_idx = i
                    break
        if target_idx is None:
            continue

        series = df.iloc[:, target_idx].dropna()
        for raw in series.astype(str).tolist():
            s = raw.strip().upper()
            # ניקוי לכל מקרה
            s = re.sub(r"[^A-Z\.]", "", s)
            if not s:
                continue
            # הוספת סיומת .TA אם חסר
            if not s.endswith(".TA"):
                s = s + ".TA"
            if s not in out:
                out.append(s)

    return out


def get_tase_tickers(index_url: str) -> List[str]:
    """
    קורא דף רכיבי מדד מה-TASE ומחזיר רשימת טיקרים בפורמט Yahoo (עם .TA).
    """
    resp = requests.get(index_url, headers=_HDRS, timeout=20)
    resp.raise_for_status()

    # pandas.read_html צפוי – אם עתידי ייעלם literal-html, נעטוף ב-StringIO אם צריך.
    dfs = pd.read_html(resp.text)
    symbols = _extract_symbols_from_tables(dfs)

    # ביטחון: מסנן כפולים ומחזיר רשימה
    symbols = pd.unique(pd.Series(symbols)).tolist()
    return symbols


def _to_field_first_wide(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    yfinance.download עם group_by='ticker' מחזיר MultiIndex ברמת (ticker, field).
    כאן נהפוך ל-(field, ticker) כדי שיהיה נוח: wide['Adj Close'][<TICKER>]
    תומך גם במקרה של טיקר יחיד.
    """
    if isinstance(df.columns, pd.MultiIndex):
        wide = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        # טיקר בודד – לבנות MultiIndex ידני
        t = tickers[0] if isinstance(tickers, list) and len(tickers) else "TICKER"
        wide = pd.concat({t: df}, axis=1)             # (ticker, field)
        wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return wide


def download_price_history(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    מוריד היסטוריית מחירים מ-Yahoo למספר טיקרים ומחזיר DataFrame Wide:
    עמודות MultiIndex: ('Adj Close','TEVA.TA'), ('Close','TEVA.TA'), ('Volume','TEVA.TA') ...
    """
    if not tickers:
        return pd.DataFrame()

    # yfinance לעיתים מחזיר None כשאין נתונים; ננסה שתי קריאות עם השהייה קצרה.
    for attempt in range(2):
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        if data is not None and not data.empty:
            break
        time.sleep(1.0)

    if data is None or data.empty:
        return pd.DataFrame()

    # ניקוי אינדקס תאריכים בעייתיים/כפולים
    data = data[~data.index.duplicated(keep="last")]

    wide = _to_field_first_wide(data, tickers)

    # שמירה רק על שדות נפוצים
    expected_fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    have_fields = [f for f in expected_fields if f in wide.columns.get_level_values(0)]
    if not have_fields:
        return pd.DataFrame()

    wide = wide.loc[:, wide.columns.get_level_values(0).isin(have_fields)]
    return wide
