import os
import time
import pickle
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import get_tickers_from_tase, download_price_history
from features import add_indicators
from model import build_dataset, train_ensemble
from alerts import send_alert

load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# ===== הרשאות (אופציונלי) =====
if APP_PASSWORD:
    pw = st.sidebar.text_input("🔐 סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.stop()

st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("תשבורת — TA-35 / TA-125 (Yahoo)")

with st.sidebar:
    st.markdown("### הגדרות:")
    index_url = st.text_input("קישור רכיבי מדד (TA-35/125)", 
        value="https://market.tase.co.il/en/market_data/index/142/components")  # TA-35
    total_capital = st.number_input("סכום להשקעה (₪)", value=100000.0, step=1000.0, min_value=0.0)
    k = st.number_input("מספר פוזיציות", value=8, step=1, min_value=1, max_value=50)
    horizon = st.selectbox("אופק", ["יומי", "שבועי"])  # כרגע ההורדות תמיד '1d', ההחלטות מושפעות מאינדיקטורים
    ignore_cache = st.checkbox("בפעם הזו להתעלם מ-cache", value=False)
    run_btn = st.button("הרץ המלצות")

# ===== קאש ל-Render (זיכרון תהליך) =====
@st.cache_data(ttl=3600)
def _cache_tickers(url):
    return get_tickers_from_tase(url)

@st.cache_data(ttl=3600)
def _cache_prices(tickers: list[str]):
    return download_price_history(tickers, period="1y", interval="1d")

if run_btn:
    st.subheader("1) שליפת רשימת טיקרים מה-TASE")
    tickers = get_tickers_from_tase(index_url) if ignore_cache else _cache_tickers(index_url)
    st.success(f"נמצאו {min(100,len(tickers))} סמלים. בודק עד 100 (ליעילות).")
    tickers = tickers[:100]

    st.subheader("2) הורדת נתונים וחישוב אינדיקטורים")
    raw = download_price_history(tickers) if ignore_cache else _cache_prices(tickers)
    if not raw:
        st.error("לא נמצאו נתוני מחיר.")
        st.stop()

    # בונים פיצ'רים לכל נייר + מודל פשוט לחיזוי עלייה מחר
    results = []
    models_cache = {}
    for t, df in raw.items():
        df2 = add_indicators(df)
        ds = build_dataset(df2)
        if len(ds) < 80:
            continue
        model, cv_score = train_ensemble(ds)
        # ההסתברות לעלייה ביום הבא – מהשורה האחרונה
        X_last = ds.drop(columns=["target"]).iloc[[-1]]
        prob_up = float(model.predict_proba(X_last)[0][1])
        results.append({
            "ticker": t,
            "Adj Close": float(df2["Adj Close"].iloc[-1]),
            "prob_up": prob_up,
            "cv": cv_score,
        })
        models_cache[t] = (model, cv_score)

    if not results:
        st.error("לא נוצרו תוצאות (ייתכן שאין מספיק נתונים).")
        st.stop()

    picks = pd.DataFrame(results).sort_values("prob_up", ascending=False).reset_index(drop=True)
    picks = picks.head(int(k)).copy()
    weights = picks["prob_up"] / picks["prob_up"].sum()
    picks["allocation_%"] = (weights * 100).round(2)
    picks["allocation_₪"] = (weights * total_capital).round(0)

    st.subheader("3) אימון מודל והפקת ציון (CV)")
    st.write(f"מס' דגימות לכל נייר: ~{len(ds)} (משתנה בהתאם לנתונים)")

    st.subheader("4) טבלת המלצות")
    st.dataframe(picks[["ticker", "Adj Close", "prob_up", "allocation_%", "allocation_₪"]], use_container_width=True)

    st.download_button("⬇️ הורד CSV", data=picks.to_csv(index=False).encode("utf-8"),
                       file_name="recommendations.csv", mime="text/csv")

    st.subheader("5) גרף למניה נבחרת")
    sel = st.selectbox("בחר מניה", options=picks["ticker"].tolist())
    if sel in raw:
        df = add_indicators(raw[sel]).tail(250).copy()
        if "Date" not in df.columns:
            df = df.reset_index()
        st.plotly_chart(px.line(df, x="Date", y="Adj Close", title=f"{sel} — מחיר"), use_container_width=True)
        st.plotly_chart(px.line(df, x="Date", y="rsi", title="RSI (14)"), use_container_width=True)

    # התראות בסיסיות – אם יש מניות עם הסתברות גבוהה מאוד
    alert_threshold = 0.7
    strong = picks[picks["prob_up"] >= alert_threshold].copy()
    if len(strong) > 0:
        html = "<h3>TA Advisor — איתותים חזקים</h3>" + "<br>".join(
            [f"{r['ticker']}: {r['prob_up']:.2f}" for _, r in strong.iterrows()]
        )
        send_alert("TA Advisor — איתותים חזקים", html)
        st.success(f"נשלחה התראה (סטאב). נמצאו {len(strong)} מניות מעל {alert_threshold:.0%}.")

    st.caption(f"משך הריצה: ~{int(time.time())} שניות (כולל קאש).")
else:
    st.info("הזן קישור רכיבי מדד (TA-35/125) ולחץ על **הרץ המלצות**.")
