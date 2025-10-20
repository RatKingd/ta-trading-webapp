# streamlit_app.py
# ----------------
# TA Trading WebApp – UI נקי שמצייר מיידית,
# ומבצע הורדות/חישובים רק לאחר לחיצה על כפתור, עם spinner/סטטוס.

from __future__ import annotations
import os
import time
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import get_tase_tickers, download_price_history

# -------------------------------------------------
# הגדרות בסיסיות ל־Streamlit
# -------------------------------------------------
st.set_page_config(page_title="TA Trading WebApp", layout="wide")
load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# -------------------------------------------------
# UI בסיסי + אימות (מצויר מיידית)
# -------------------------------------------------
st.title("תשבורת — TA-35 / TA-125 (Yahoo)")
if APP_PASSWORD:
    pw = st.sidebar.text_input("🔐 סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.info("הכנס סיסמה כדי להמשיך.")
        st.stop()

with st.sidebar:
    st.header("הגדרות:")
    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/142/components",
        help="הדבק את קישור רכיבי המדד מה-TASE",
    )
    total_capital = st.number_input("סכום להשקעה (₪)", min_value=0.0, value=100000.0, step=1000.0)
    top_n = st.number_input("מספר פוזיציות", min_value=1, max_value=50, value=8, step=1)
    horizon = st.selectbox("אופק", ["יומי", "שבועי"], index=0)
    ignore_cache = st.checkbox("בפעם הזו להתעלם מ-cache", value=False)
    run_btn = st.button("הרץ המלצות")

# מקום לתוצרים (כדי שהדף לא יהיה ריק)
sec1 = st.container()
sec2 = st.container()
sec3 = st.container()
sec4 = st.container()
sec5 = st.container()

# -------------------------------------------------
# לוגיקה תרוץ רק לאחר לחיצה
# -------------------------------------------------
if not run_btn:
    st.caption("לחץ על “הרץ המלצות” כדי להתחיל.")
    st.stop()

# -------------------------------------------------
# 1) שליפת רשימת טיקרים מה-TASE
# -------------------------------------------------
with sec1:
    st.subheader("1) שליפת רשימת טיקרים מה-TASE")
    with st.status("בודק ומושך עד 100 סמלים ראשונים…", expanded=False) as s:
        tickers: List[str] = []
        ok = False
        try:
            # גם אם האתר איטי – get_tase_tickers מגדיר timeout פנימי.
            tickers = get_tase_tickers(index_url)[:100]
            if tickers:
                st.success(f"נמצאו {len(tickers)} סמלים. (בדוק עד 100 ראשונים)")
                ok = True
            else:
                st.warning("לא נמצאו סמלים. בדוק את הקישור.")
        except Exception as e:
            st.error(f"שגיאה בשליפת טיקרים: {e}")
        s.update(state="complete")

    if not ok:
        st.stop()

# -------------------------------------------------
# 2) הורדת נתונים וחישוב אינדיקטורים
# -------------------------------------------------
with sec2:
    st.subheader("2) הורדת נתונים וחישוב אינדיקטורים")
    # בחירת period/interval לפי אופק
    if horizon == "שבועי":
        period, interval = "2y", "1wk"
    else:
        period, interval = "6mo", "1d"

    # cache של הורדות – כדי להימנע ממסך לבן בין ריצות
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_download(tix: List[str], p: str, itv: str) -> pd.DataFrame:
        return download_price_history(tix, period=p, interval=itv)

    if ignore_cache:
        _cached_download.clear()

    with st.spinner("מוריד נתונים מ-Yahoo…"):
        df_all = _cached_download(tickers, period, interval)

    if df_all is None or df_all.empty:
        st.error("לא נמצאו נתוני מחירים (Yahoo). נסה שוב או הפחת מספר טיקרים.")
        st.stop()
    else:
        st.success(f"נתונים התקבלו בהצלחה – {df_all.shape[0]} שורות, {len(tickers)} סמלים (ייתכן שחלקם חסרים).")

# -------------------------------------------------
# 3) דוגמת “ניקוד” פשוטה + טבלה
#    (החלף כאן בהיגיון האמיתי שלך כשיהיה מוכן)
# -------------------------------------------------
with sec3:
    st.subheader("3) ניקוד פשוט והפקת רשימת Picks")
    # ניקוד נאיבי: שינוי יחסי של "Adj Close" על 30 הברות האחרונות
    field = "Adj Close"
    missing = [t for t in tickers if (field, t) not in df_all.columns]
    used = [t for t in tickers if (field, t) in df_all.columns]
    if len(used) == 0:
        st.error("אין עמודות 'Adj Close' זמינות לאף טיקר.")
        st.stop()

    closes = df_all.loc[:, df_all.columns.get_level_values(0) == field].copy()
    closes.columns = closes.columns.droplevel(0)  # להשאיר רק שמות טיקר
    # שינוי יחסי אחרון (פשוט להדגמה)
    scores = (closes.iloc[-1] / closes.iloc[-30].replace(0, pd.NA) - 1.0).dropna()
    picks = (
        pd.DataFrame({
            "ticker": scores.index,
            "score": scores.values,
        })
        .sort_values("score", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )

    st.write("טבלת Picks (מדגמית):")
    st.dataframe(picks, use_container_width=True)

# -------------------------------------------------
# 4) גרף למניה נבחרת
# -------------------------------------------------
with sec4:
    st.subheader("4) גרף למניה נבחרת")
    sel = st.selectbox("בחר מניה", options=picks["ticker"].tolist())
    data_sel = df_all.xs(key="Adj Close", level=0, axis=1).get(sel)
    if data_sel is not None and isinstance(data_sel, pd.Series):
        dfp = data_sel.dropna().reset_index()
        dfp.columns = ["Date", "Adj Close"]
        st.plotly_chart(px.line(dfp, x="Date", y="Adj Close", title=f"{sel} — מחיר"), use_container_width=True)
    else:
        st.info("אין נתונים לגרף עבור הבחירה.")

# -------------------------------------------------
# 5) זמן ריצה
# -------------------------------------------------
with sec5:
    st.caption(f"זמן הריצה (Client): ~{int(time.time()) % 1000} (אינדיקציה בלבד)")
