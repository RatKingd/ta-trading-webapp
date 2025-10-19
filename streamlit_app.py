# streamlit_app.py

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import requests
import yfinance as yf

# ========= הגדרות ראשוניות =========
load_dotenv()
st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("תשבורת — TA-35 / TA-125 (Yahoo)")

# אימות אופציונלי בסיסמה דרך משתנה סביבה APP_PASSWORD
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD:
    pw = st.sidebar.text_input("🔑 סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.stop()

# ========= עזר: כותרות משנה =========
def subh(txt: str):
    st.subheader(txt)

# ========= הגדרות משתמש =========
with st.sidebar:
    st.markdown("### הגדרות:")
    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/142/components"  # TA-35
    )
    total_capital = st.number_input("סכום להשקעה (₪)", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    top_n = st.number_input("מספר פוזיציות", min_value=1, max_value=20, value=8, step=1)
    horizon = st.selectbox("אופק", options=["יומי", "שבועי"], index=0)
    bypass_cache = st.checkbox("להתעלם מה-cache בפעם הזו", value=False)

# ========= עזר: הבאת רשימת טיקרי TASE מהעמוד =========
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

@st.cache_data(ttl=60*60, show_spinner=False)
def get_tickers_from_tase(url: str) -> list[str]:
    """מנסה לחלץ סמלים מהעמוד; אם נופל – מחזיר fallback קצר."""
    FALLBACK_TA35 = ["TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ICL.TA"]
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        # pd.read_html דורש lxml מותקן (טופל ב-requirements + Dockerfile)
        dfs = pd.read_html(r.text)
        syms = []
        for df in dfs:
            cols = [str(c).lower() for c in df.columns]
            if any(c in cols for c in ["symbol", "instrument symbol", "ticker"]):
                for candidate in ["Instrument Symbol", "Symbol", "instrument", "Ticker"]:
                    if candidate in df.columns:
                        vals = df[candidate].dropna().astype(str).str.strip().tolist()
                        syms.extend(vals)
                        break
        # תקנון סיומת לתבנית Yahoo (.TA)
        syms = [s if s.endswith(".TA") else f"{s}.TA" for s in syms]
        # הסרה כפילויות וניקוי
        syms = sorted(set([s for s in syms if len(s) > 3]))
        return syms[:100] if syms else FALLBACK_TA35
    except Exception:
        return FALLBACK_TA35

# ========= הורדת היסטוריית מחירים =========
@st.cache_data(ttl=60*30, show_spinner=False)
def download_price_history(tickers: list[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """מחזיר DataFrame בפורמט wide: עמודות MultiIndex (field, ticker)."""
    # yfinance מאפשר להוריד מספר סמלים יחד
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        threads=True,
        progress=False
    )
    # במידה וקיבלנו Series/DF רזה – להמיר לפורמט אחיד
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

# ========= אינדיקטורים =========
def add_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """מוסיף RSI(14) ו-SMA(20/50) לכל נייר. קלט: wide MultiIndex -> פלט: long tidy"""
    # מצפה ל-MultiIndex: (field, ticker). נבחר 'Adj Close' אם קיים אחרת 'Close'
    fields = list(price_df.columns.get_level_values(0).unique()) if isinstance(price_df.columns, pd.MultiIndex) else []
    price_field = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else None)

    if price_field is None:
        raise ValueError("לא נמצאה עמודת Close/Adj Close בנתונים שהורדו.")

    close = price_df[price_field].copy()  # wide: Date × tickers

    # חישוב אינדיקטורים לכל טיקר
    def rsi(series, window=14):
        delta = series.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=series.index).rolling(window).mean()
        roll_down = pd.Series(down, index=series.index).rolling(window).mean()
        rs = roll_up / (roll_down + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    ind = {}
    for t in close.columns:
        s = close[t].dropna()
        df_t = pd.DataFrame({
            "Date": s.index,
            "Close": s.values
        })
        df_t["sma20"] = df_t["Close"].rolling(20).mean()
        df_t["sma50"] = df_t["Close"].rolling(50).mean()
        df_t["rsi"] = rsi(df_t["Close"], 14)
        df_t["ticker"] = t
        ind[t] = df_t

    out = pd.concat(ind.values(), ignore_index=True)
    return out  # long tidy: Date, Close, sma20, sma50, rsi, ticker

# ========= מודל פשוט + דירוג =========
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def score_and_rank(tidy_df: pd.DataFrame, horizon: str, top_n: int) -> tuple[pd.DataFrame, list]:
    """מייצר מטריצת תכונות לכל טיקר, מאמן מודל, מחשב הסתברות לעלייה ומחזיר TOP N."""
    # מגדירים תשואה עתידית לפי אופק
    tidy = tidy_df.sort_values(["ticker", "Date"]).copy()
    tidy["ret1"] = tidy.groupby("ticker")["Close"].pct_change().shift(-1 if horizon == "יומי" else -1)
    # בוחרים תכונות
    features = tidy[["sma20", "sma50", "rsi"]].fillna(method="ffill").fillna(0.0)
    y = (tidy["ret1"] > 0).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    # ניקוי רשומות חסרות
    mask = np.isfinite(features).all(axis=1) & y.notna()
    X = features[mask].values
    yv = y[mask].values

    scores = []
    try:
        cv_acc = cross_val_score(pipe, X, yv, cv=5, scoring="accuracy")
        cv_f1 = cross_val_score(pipe, X, yv, cv=5, scoring="f1")
        scores = [("דיוק", float(cv_acc.mean())), ("F1", float(cv_f1.mean()))]
    except Exception:
        scores = [("דיוק", np.nan), ("F1", np.nan)]

    # אימון סופי והפקת הסתברויות עבור התאריך האחרון לכל טיקר
    pipe.fit(X, yv)
    last_per_ticker = tidy.groupby("ticker").tail(1).copy()
    X_last = last_per_ticker[["sma20", "sma50", "rsi"]].fillna(0.0).values
    prob_up = pipe.predict_proba(X_last)[:, 1]
    picks = pd.DataFrame({
        "ticker": last_per_ticker["ticker"].values,
        "Adj Close": last_per_ticker["Close"].values,
        "prob_up": prob_up
    }).sort_values("prob_up", ascending=False).head(top_n).reset_index(drop=True)

    # חלוקת הקצאות שסוכמות ל-100% (softmax)
    if len(picks) > 0:
        w = np.exp(picks["prob_up"] - picks["prob_up"].max())
        w = w / w.sum()
        picks["allocation_%"] = (w * 100).round(2)
        picks["allocation_₪"] = (w * total_capital).round(0).astype(int)
    return picks, scores

# ========= כפתור הרצה =========
if st.button("הרץ המלצות"):
    # 1) שליפת טיקרים
    subh("1) שליפת רשימת טיקרים מה-TASE")
    st.write("בודק עד 100 סימבולים... (זמן קצר)")
    if bypass_cache:
        get_tickers_from_tase.clear()
    tickers = get_tickers_from_tase(index_url)[:100]
    if len(tickers) == 0:
        st.error("לא נמצאו טיקרים. בדוק את הקישור או נסה שוב.")
        st.stop()
    st.success(f"נמצאו {len(tickers)} סמלים. מצמצם ל־{min(100, len(tickers))} הראשונים.")

    # 2) הורדת מחירים וחישוב אינדיקטורים
    subh("2) חישוב אינדיקטורים")
    interval = "1d" if horizon == "יומי" else "1wk"   # ← תיקון 'weekly'
    period = "1y" if horizon == "יומי" else "5y"
    if bypass_cache:
        download_price_history.clear()
    try:
        price_df = download_price_history(tickers, period=period, interval=interval)
    except TypeError:
        # אם לגרסה קיימת של הפונקציה אין פרמטרים — fallback ישיר ל-yfinance
        price_df = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, threads=True, progress=False)

    # אם ריק – לעצור
    if price_df is None or len(price_df) == 0:
        st.error("הורדת המחירים נכשלה או חזרה ריקה.")
        st.stop()

    try:
        tidy = add_indicators(price_df)
    except Exception as e:
        st.exception(e)
        st.stop()

    # 3) אימון מודל ודירוג
    subh("3) אימון מודל והפקת ציון")
    picks, cv_scores = score_and_rank(tidy, horizon=horizon, top_n=int(top_n))
    st.write("תוצאות CV (דיוק, F1):", cv_scores)

    # 4) טבלת המלצות + הורדה
    subh("4) טבלת המלצות")
    st.dataframe(picks, use_container_width=True)
    if not picks.empty:
        csv = picks[["ticker", "Adj Close", "prob_up", "allocation_%", "allocation_₪"]].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ הורד CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

    # 5) גרף למניה נבחרת
    subh("5) גרף למניה נבחרת")
    if not picks.empty:
        sel = st.selectbox("בחר מניה", options=picks["ticker"].tolist())
        df_sel = tidy[tidy["ticker"] == sel].copy()
        # להבטיח Date כעמודה (ולא index) – חשוב ל-plotly
        if "Date" not in df_sel.columns:
            df_sel = df_sel.reset_index()
        # קו מחיר
        st.plotly_chart(
            px.line(df_sel.tail(250), x="Date", y="Close", title=f"{sel} — מחיר"),
            use_container_width=True
        )
        # RSI
        st.plotly_chart(
            px.line(df_sel.tail(250), x="Date", y="rsi", title="RSI (14)"),
            use_container_width=True
        )

    # 6) התראות טקסט (פשוט למסך; ניתן להרחיב ל-SMTP בהמשך)
    subh("6) התראות חזקות")
    alert_threshold = 0.70
    strong = picks[picks["prob_up"] >= alert_threshold].copy() if not picks.empty else pd.DataFrame()
    if len(strong) > 0:
        html = "<br>".join([f"{t} — {p:.2%}" for t, p in zip(strong["ticker"], strong["prob_up"])])
        st.success(f"🚨 נמצאו {len(strong)} סימבולים מעל סף {int(alert_threshold*100)}%:<br>{html}", icon="✅")
    else:
        st.info("אין סימבולים מעל סף ההתראה כרגע.")

    st.caption(f"ריצה הושלמה בזמן: {int(time.time())}")

else:
    st.caption("טיפ: אפשר להדביק קישור אחר לרשימת רכיבי TA-35/TA-125 (למשל index/168/components).")
