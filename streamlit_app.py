# streamlit_app.py
# ----------------
# אפליקציית Streamlit להצגת מניות TA-35/125, הורדת נתוני מחירים מ-Yahoo,
# חישוב אינדיקטורים בסיסיים, יצירת טבלת המלצות וגרפים.

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv

# פונקציות שהוגדרו בקובץ data_fetcher.py (הגרסה המלאה שסיפקתי לך)
from data_fetcher import (
    get_tase_tickers,           # (index_url: str) -> list[str]
    download_price_history      # (tickers: list[str], period: str, interval: str) -> pd.DataFrame (MultiIndex)
)

# ────────────────────────────────────────────────────────────────────────────────
# הגדרות כלליות + טעינת סביבה
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(page_title="TA Trading WebApp", layout="wide")

# סטייל קטן לימין-לשמאל
st.markdown(
    """
    <style>
    .block-container {direction: rtl;}
    .stButton>button {direction: rtl;}
    </style>
    """,
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────────────────────────
# פונקציות עזר לאינדיקטורים
# ────────────────────────────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def enrich_indicators(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    wide_df בפורמט wide per ticker:
    עמודות לדוגמה: ('Adj Close','TEVA.TA'), ('Volume','TEVA.TA') וכו'.
    לאחר ההעשרה מוחזר DataFrame חדש שנוח ל-plot (per selected ticker).
    """
    return wide_df  # בפורמט Wide נשאיר כפי שהוא; חישובים נעשים פר-מניה למטה.


# ────────────────────────────────────────────────────────────────────────────────
# Sidebar — קלטים מהמשתמש
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("הגדרות")
index_url = st.sidebar.text_input(
    "קישור רכיבי מדד (TA-35/125)",
    value="https://market.tase.co.il/en/market_data/index/142/components"  # TA-35
)
total_capital = st.sidebar.number_input("סכום להשקעה (₪)", min_value=0.0, value=100_000.0, step=1000.0, format="%.2f")
num_positions = st.sidebar.number_input("מספר פוזיציות", min_value=1, max_value=30, value=8, step=1)

tf = st.sidebar.selectbox("אופק", ["יומי", "שבועי"])  # מיפוי אח״כ ל-interval
bypass_cache = st.sidebar.checkbox("בפעם הזו להתעלם מ-cache", value=False)

run = st.sidebar.button("הרץ המלצות")

# אפשרות לנעילת גישה בסיסית בסיסמה דרך ENV (לא חובה)
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD:
    pw = st.sidebar.text_input("סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Cache ניקוי לפי הצורך
# ────────────────────────────────────────────────────────────────────────────────
if bypass_cache:
    try:
        st.cache_data.clear()
        st.sidebar.success("ה-cache נוקה להפעלה זו.")
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────────────────
# גוף האפליקציה
# ────────────────────────────────────────────────────────────────────────────────
st.title("תשבורת — TA-35 / TA-125 (Yahoo)")

# (1) שליפת רשימת טיקרים
with st.expander("1) שליפת רשימת טיקרים מה TASE", expanded=True):
    try:
        tickers = get_tase_tickers(index_url)
        # כדי לא לעבור מגבלות של Yahoo, נגביל לגג 100 טיקרים לחישוב ראשוני
        tickers = tickers[:100]
        st.success(f"נמצאו {len(tickers)} סמלים. בדוק עד 100 (לשיקול עומס).")
        if st.checkbox("להציג את הטיקרים שאותרו", value=False):
            st.write(tickers)
    except Exception as e:
        st.error(f"שגיאה בשליפת טיקרים מהאתר. {e}")
        st.stop()

# אם לא לחצו "הרץ", לא נמשיך לחישובים כדי לא להכביד
if not run:
    st.info("לחץ/י על 'הרץ המלצות' כדי להמשיך.")
    st.stop()

# (2) הורדת נתונים וחישוב אינדיקטורים
with st.expander("2) הורדת נתונים וחישוב אינדיקטורים", expanded=True):
    interval = "1d" if tf == "יומי" else "1wk"
    try:
        # מורידים 6 חודשים כברירת מחדל — מספיק לחישובי SMA/RSI
        prices_wide = download_price_history(tickers, period="6mo", interval=interval)
        if prices_wide.empty:
            st.error("לא נמצאו נתוני מחירים מ-Yahoo (שוב). נסה/י כיבוי cache, שינוי אופק, או פחות סמלים.")
            st.stop()

        # בסיס חישוב: ניקח את מחירי ה-Adj Close לסיום ונסנן טורים שאין בהם נתונים
        adj_close = prices_wide["Adj Close"].copy()
        adj_close = adj_close.dropna(how="all", axis=1)

        if adj_close.empty:
            st.error("לא התקבלו מחירי Adj Close תקינים מהנתונים שהורדו.")
            st.stop()

        # בוחרים את N המניות/טיקרים לחישוב (כאן: נסנן טיקרים שאין להם מספיק תצפיות)
        min_points = 30 if interval == "1d" else 8  # דורש לפחות חודש + קצת, או 8 שבועות
        enough_history = [t for t in adj_close.columns if adj_close[t].dropna().shape[0] >= min_points]
        adj_close = adj_close[enough_history]

        if adj_close.empty:
            st.error("כל הטיקרים נפסלו בשל היסטוריה קצרה מדי. נסו אופק יומי/שבועי אחר או פחות טיקרים.")
            st.stop()

        st.success(f"נתונים הושגו עבור {adj_close.shape[1]} טיקרים.")
    except Exception as e:
        st.exception(e)
        st.stop()

    # ── חישוב אינדיקטורים בסיסיים לכל טיקר ───────────────────────────────────
    # נבנה DataFrame מסכם אחרון לשקלול "ציון"
    summary_rows = []

    for t in adj_close.columns:
        s = adj_close[t].dropna()
        if s.empty:
            continue

        sma20 = s.rolling(20).mean()
        sma50 = s.rolling(50).mean() if interval == "1d" else s.rolling(10).mean()  # הקבלה לשבועי
        rsi = calc_rsi(s, 14)

        last_price = float(s.iloc[-1])
        last_sma50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else np.nan
        last_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan

        # ציון פשוט: 0..1 — חצי על יחס למחיר מול SMA50 וחצי על RSI מנורמל
        comp1 = 1.0 if (not np.isnan(last_sma50) and last_price > last_sma50) else 0.0
        comp2 = 0.0 if np.isnan(last_rsi) else np.clip((last_rsi - 30) / (70 - 30), 0, 1)
        score = 0.5 * comp1 + 0.5 * comp2

        summary_rows.append(
            dict(ticker=t, last_price=last_price, rsi=last_rsi, score=score)
        )

    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    if summary.empty:
        st.error("לא ניתן לחשב אינדיקטורים/ציון. בדקו שהנתונים תקינים או נסו אופק אחר.")
        st.stop()

    # בוחרים את N המובילות לפי הציון
    picks = summary.head(int(num_positions)).reset_index(drop=True)

    # הקצאת משקלות פרופורציונלית לציון (נרמול ל-100%)
    weights = picks["score"].clip(lower=0)
    if weights.sum() == 0:
        weights[:] = 1.0
    alloc = (weights / weights.sum()) * 100.0
    picks["allocation_%"] = alloc.round(2)

    # הוספת Adj Close מה-wide
    picks["Adj Close"] = [adj_close[t].dropna().iloc[-1] if t in adj_close else np.nan
                          for t in picks["ticker"]]

    st.success(f"נבחרו {len(picks)} טיקרים מובילים לפי ציון משולב.")
    st.dataframe(
        picks[["ticker", "Adj Close", "score", "allocation_%"]],
        use_container_width=True
    )

    # כפתור הורדת CSV
    csv_bytes = picks[["ticker", "Adj Close", "score", "allocation_%"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ הורד CSV", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")

# (3) גרפים
with st.expander("3) גרף למניה נבחרת", expanded=True):
    ticker_options = picks["ticker"].tolist()
    if len(ticker_options) == 0:
        st.info("אין מניות להצגה.")
    else:
        sel = st.selectbox("בחר מניה", options=ticker_options)
        if sel:
            df = adj_close[[sel]].rename(columns={sel: "Adj Close"})
            df = df.dropna().reset_index().rename(columns={"Date": "Date"})
            # RSI נוסף לתצוגה:
            rsi_series = calc_rsi(df["Adj Close"], 14)
            out = pd.DataFrame({"Date": df["Date"], "Adj Close": df["Adj Close"], "rsi": rsi_series})

            st.plotly_chart(
                px.line(out, x="Date", y="Adj Close", title=f"{sel} — מחיר"),
                use_container_width=True
            )
            st.plotly_chart(
                px.line(out, x="Date", y="rsi", title="RSI (14)"),
                use_container_width=True
            )

# (4) חיווי זמן
st.caption(f"עודכן: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
