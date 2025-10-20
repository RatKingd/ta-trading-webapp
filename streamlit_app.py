# streamlit_app.py
import os
import time
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import get_tickers_from_tase, download_price_history

# ------------------ הגדרות כלליות ------------------
load_dotenv()
st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("תשבורת — TA-35 / TA-125 (Yahoo)")

# ------------------ סרגל צד ------------------
with st.sidebar:
    st.markdown("### הגדרות")
    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/142/components"  # TA-35 כברירת מחדל
    )
    total_capital = st.number_input("סכום להשקעה (₪)", min_value=0.0, value=100_000.0, step=1_000.0, format="%.2f")
    top_n = st.number_input("מספר פוזיציות", min_value=1, max_value=20, value=8, step=1)
    horizon_label = st.selectbox("אופק", options=["יומי", "שבועי", "חודשי"], index=0)
    horizon = {"יומי": "daily", "שבועי": "weekly", "חודשי": "monthly"}[horizon_label]
    ignore_cache = st.checkbox("בפעם הזו להתעלם מ-cache", value=False)
    run_btn = st.button("הרץ המלצות")

# ------------------ cache helpers ------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_tickers(url: str):
    return get_tickers_from_tase(url)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_prices(tickers: tuple[str, ...], horizon: str):
    return download_price_history(list(tickers), horizon=horizon)

def get_tickers(url: str):
    return get_tickers_from_tase(url) if ignore_cache else _cached_tickers(url)

def get_prices(tickers: list[str], horizon: str):
    key = tuple(sorted(tickers))
    return download_price_history(tickers, horizon) if ignore_cache else _cached_prices(key, horizon)

# ------------------ חישובי אינדיקטורים פשוטים ------------------
def compute_indicators(series: pd.Series) -> pd.DataFrame:
    """מקבל סדרת מחירים ומחזיר DataFrame עם אינדיקטורים בסיסיים."""
    df = series.to_frame("Adj Close").copy()
    df["ret1"] = df["Adj Close"].pct_change()
    df["sma20"] = df["Adj Close"].rolling(20).mean()
    df["sma50"] = df["Adj Close"].rolling(50).mean()
    # RSI בסיסי (14)
    delta = df["Adj Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    df["rsi"] = 100 - (100 / (1 + rs))
    return df.dropna()

def score_ticker(df: pd.DataFrame) -> float:
    """ציון בסיסי: שילוב מומנטום ויחסי ממוצעים (נורמליזציה 0..1)."""
    if df.empty:
        return 0.0
    last = df.iloc[-1]
    score = 0.0
    # מעל הממוצעים?
    score += 0.4 if last["Adj Close"] > last.get("sma50", 0) else 0.0
    score += 0.2 if last["Adj Close"] > last.get("sma20", 0) else 0.0
    # RSI באזור 50-70
    if not np.isnan(last.get("rsi", np.nan)):
        score += 0.4 * max(0.0, min((last["rsi"] - 50) / 20, 1.0))
    return float(np.clip(score, 0, 1))

# ------------------ הרצה ------------------
if run_btn:
    st.subheader("1) ‏שליפת רשימת טיקרים מ-TASE")
    tickers = get_tickers(index_url)
    if not tickers:
        st.error("לא נמצאו סמלים. בדוק את ה-URL או נסה שוב.")
        st.stop()
    st.success(f"נמצאו {min(len(tickers),100)} סמלים. בודק עד 100 (לשיקולי עומס).")

    st.subheader("2) הורדת נתונים וחישוב אינדיקטורים")
    prices_map = get_prices(tickers[:100], horizon=horizon)

    if not prices_map:
        st.error("לא נמצאו נתוני מחירים (Yahoo החזיר ריק). נסה שוב עם cache כבוי או פחות טיקרים.")
        st.stop()

    rows = []
    per_ticker_frames: dict[str, pd.DataFrame] = {}
    for t, ser in prices_map.items():
        df = compute_indicators(ser)
        per_ticker_frames[t] = df
        rows.append({"ticker": t, "score": score_ticker(df)})

    if not rows:
        st.error("לא נוצרו אינדיקטורים תקינים עבור הטיקרים.")
        st.stop()

    scores = pd.DataFrame(rows).sort_values("score", ascending=False)
    picks = scores.head(int(top_n)).copy()
    # הקצאה יחסית לציון
    s = picks["score"].sum()
    if s > 0:
        picks["allocation_%"] = (picks["score"] / s * 100).round(2)
    else:
        picks["allocation_%"] = (100 / len(picks)).round(2)
    picks["allocation_₪"] = (picks["allocation_%"] / 100 * total_capital).round(0)

    # לצורכי תצוגה נוסיף Adj Close האחרון אם קיים
    last_prices = []
    for t in picks["ticker"]:
        df = per_ticker_frames.get(t, pd.DataFrame())
        last_prices.append(float(df["Adj Close"].iloc[-1]) if not df.empty else np.nan)
    picks["Adj Close"] = last_prices

    st.success("חישוב הושלם.")

    st.subheader("3) אימון/ציון (בגרסת הבסיס – כבר חושב כ-score)")
    st.write("הציון מחושב משילוב מומנטום וממוצעים נעים (0–1).")

    st.subheader("4) טבלת המלצות")
    st.dataframe(
        picks[["ticker", "Adj Close", "score", "allocation_%", "allocation_₪"]]
            .reset_index(drop=True),
        use_container_width=True,
    )

    # הורדת CSV
    csv = picks[["ticker", "Adj Close", "score", "allocation_%", "allocation_₪"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ הורד CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

    st.subheader("5) גרף למניה נבחרת")
    sel = st.selectbox("בחר מניה", options=picks["ticker"].tolist())
    df = per_ticker_frames.get(sel, pd.DataFrame()).tail(250).copy()
    if not df.empty:
        fig = px.line(df.reset_index(), x="Date", y="Adj Close", title=f"{sel} — מחיר")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(df.reset_index(), x="Date", y="rsi", title="RSI (14)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("אין נתונים להצגה לגרף.")
else:
    st.info("הגדר פרמטרים ולחץ על **הרץ המלצות**.")
