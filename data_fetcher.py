# data_fetcher.py
# -*- coding: utf-8 -*-

"""
איסוף סמלים מהבורסה (TASE) והורדת היסטוריית מחירים מ-Yahoo Finance.
הקוד קשיח ל-Render: בקשות איטיות, ריטריי, ללא threads, וטיקר-אחרי-טיקר
כדי להימנע מהחזרות ריקות.

דרישות ספריות: requests, pandas, yfinance, lxml, tqdm (רשות).
"""

from __future__ import annotations

import time
import random
from typing import List, Dict, Optional

import requests
import pandas as pd
import yfinance as yf

# tqdm רשות בלבד: אם לא מותקן, נגדיר no-op
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x


# -------- תצורה וקבועים --------

# חלק מאתרים חוסמים ברירת מחדל של requests; נגדיר User-Agent "דפדפן" רגיל.
HDRS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# רשימות גיבוי (אם TASE לא מחזיר כלום)
FALLBACK_TA35 = ["TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ICL.TA"]
FALLBACK_TA125 = FALLBACK_TA35  # ניתן להרחיב אם רוצים


# -------- כלים פנימיים --------

def _ensure_ta_suffix(symbol: str) -> str:
    """מוודא שסיומת .TA קיימת (מנקה רווחים ותווים מיותרים)."""
    s = str(symbol).strip()
    if not s:
        return s
    if not s.endswith(".TA"):
        s = s.replace(".TA", "")  # אם באמצע
        s = s + ".TA"
    return s


def _sleep_jitter(base: float = 0.25, jitter: float = 0.2) -> None:
    """השהיה קצרה עם ג'יטר קטן למניעת Rate-Limit."""
    time.sleep(max(0.05, base + random.uniform(-jitter, jitter)))


# -------- שליפת סמלים מ-TASE --------

def get_tickers_from_tase(index_url: str, timeout: int = 20) -> List[str]:
    """
    קורא את עמוד המדד בבורסת ת"א ומחלץ את סמלי המסחר (Symbol/Instrument Symbol),
    מוסיף סיומת .TA ומחזיר רשימת טיקרים נקייה. אם נכשל — מחזיר רשימת גיבוי.

    Parameters
    ----------
    index_url : str
        קישור לעמוד רכיבי המדד (למשל TA-35/TA-125 ב-market.tase.co.il).
    timeout : int
        טיימאאוט לבקשה.

    Returns
    -------
    List[str]
    """
    try:
        r = requests.get(index_url, headers=HDRS, timeout=timeout)
        r.raise_for_status()

        # pandas.read_html מחייב lxml; התקן ב-requirements.txt: lxml
        dfs = pd.read_html(r.text)
        syms: List[str] = []

        for df in dfs:
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            # ננסה לאתר עמודת סמל: שמות אפשריים
            candidates = [
                "Instrument Symbol", "Symbol", "Ticker", "Instrument", "נייר",
                "מס׳ נייר", "מס סימול", "מס רישום"
            ]
            lower_cols = {c.lower(): c for c in df.columns}

            col_name: Optional[str] = None
            for c in candidates:
                key = c.lower()
                if key in lower_cols:
                    col_name = lower_cols[key]
                    break

            if col_name is None:
                # ננסה לפי טקסט "symbol" מרומז
                for c in df.columns:
                    if "symbol" in str(c).lower() or "ticker" in str(c).lower():
                        col_name = c
                        break

            if col_name is None:
                continue

            # ניקוי ערכים
            col = df[col_name].dropna().astype(str).str.strip()
            col = col[col != ""]
            if not col.empty:
                syms.extend([_ensure_ta_suffix(x) for x in col.tolist()])

        # סינון כפולים
        syms = sorted(set([s for s in syms if s.upper().endswith(".TA")]))

        if not syms:
            # אם אין תוצאה — החזר גיבוי סביר
            return FALLBACK_TA35 if "35" in index_url else FALLBACK_TA125

        # נחתוך ל-100 מקסימום כדי לא להיחסם בהורדה
        return syms[:100]

    except Exception:
        # כל כשל — נחזיר גיבוי
        return FALLBACK_TA35 if "35" in index_url else FALLBACK_TA125


# -------- הורדת היסטוריית מחירים מ-Yahoo --------

def download_price_history(tickers: List[str], horizon: str = "daily") -> Dict[str, pd.Series]:
    """
    מוריד היסטוריית מחירים מ-Yahoo עבור רשימת טיקרים ומחזיר {ticker: Series(Adj Close או Close)}.
    עובד טיקר-אחרי-טיקר עם ריטריי כדי להימנע מהחזרות ריקות/שגיאות.

    Parameters
    ----------
    tickers : List[str]
        רשימת טיקרים (עם .TA).
    horizon : str
        'daily' / 'weekly' / 'monthly' — קובע interval מתאים.

    Returns
    -------
    Dict[str, pd.Series]
        מפה מטיקר לסדרת מחירי הסגירה המנורמלים (Adj Close אם קיים, אחרת Close).
    """
    horizon_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    interval = horizon_map.get(horizon, "1d")

    out: Dict[str, pd.Series] = {}

    # נגביל בכל מקרה ל-100 כדי לא להיחסם
    for t in tqdm(tickers[:100], desc="Downloading from Yahoo", unit="sym"):
        t = _ensure_ta_suffix(t)
        ser: Optional[pd.Series] = None

        # --- נסיון 1: yf.download לטיקר יחיד ---
        for _ in range(2):
            try:
                df = yf.download(
                    tickers=t,
                    period="3y",
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,   # חשוב בסביבות מרובות-תהליכים (Render)
                    group_by=None,   # מחזיר DataFrame "שטוח"
                )

                if isinstance(df, pd.DataFrame) and not df.empty:
                    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
                    if col is not None:
                        s = df[col].dropna()
                        if not s.empty:
                            ser = s
                            break
                _sleep_jitter(0.6, 0.3)
            except Exception:
                _sleep_jitter(0.6, 0.3)

        # --- נסיון 2 (fallback): Ticker(...).history ---
        if ser is None or ser.empty:
            try:
                h = yf.Ticker(t).history(period="3y", interval=interval, auto_adjust=True)
                if isinstance(h, pd.DataFrame) and not h.empty:
                    col = "Close" if "Close" in h.columns else None
                    if col is not None:
                        s = h[col].dropna()
                        if not s.empty:
                            ser = s
            except Exception:
                pass

        if ser is not None and not ser.empty:
            out[t] = ser

        # הפסקה קטנה בין טיקרים כדי לא להיחסם
        _sleep_jitter(0.25, 0.15)

    return out
