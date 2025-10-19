import os, pickle, time
import pandas as pd, numpy as np, plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import get_tickers_from_tase, download_price_history
from features import add_indicators
from model import build_dataset, train_ensemble, ensemble_predict, FEATURES
from alerts import send_alert

load_dotenv()

# -------- Auth (סיסמה פשוטה) --------
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD:
    pw = st.sidebar.text_input("🔒 סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.stop()

st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("TA-35 / TA-125 — דשבורד המלצות (Yahoo)")

with st.sidebar:
    st.markdown("**הגדרות:**")
    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/142/components"  # TA-35
    )
    total_capital = st.number_input("סכום להשקעה (₪)", value=100_000.0, step=1000.0)
    top_n = st.number_input("מספר פוזיציות", min_value=1, value=8, step=1)
    horizon = st.selectbox("אופק", options=["יומי", "שבועי"])
    thresh = 0.01 if horizon == "יומי" else 0.02
    alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.9"))

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

colA, colB = st.columns([1,1])
run = colA.button("הרץ המלצות")
refresh_cache = colB.checkbox("להתעלם מ-cache ולהוריד נתונים מחדש", value=False)

if run:
    t0 = time.time()
    st.subheader("1) שליפת רשימת טיקרים מה-TASE")
    tickers = get_tickers_from_tase(index_url)
    st.write(f"נמצאו {len(tickers)} סימבולים. נרוץ על עד 100 לניסיון ראשון.")
    tickers = tickers[:100]

    cache_file = os.path.join(CACHE_DIR, f"prices_{len(tickers)}.pkl")
    data = None
    if not refresh_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f: data = pickle.load(f)
        st.success(f"נטען מה-cache ({len(data)} סדרות)")
    if data is None:
        st.write("מוריד נתונים מ-Yahoo… (דקות ספורות)")
        data = download_price_history(tickers, start="2016-01-01")
        with open(cache_file, "wb") as f: pickle.dump(data, f)

    st.subheader("2) חישוב אינדיקטורים")
    clean = {}
    for t, df in list(data.items()):
        try:
            clean[t] = add_indicators(df)
        except Exception:
            pass
    data = clean
    st.write(f"תקין ל-{len(data)} סמלים.")

    st.subheader("3) אימון מודל והפקת ציון")
    df_all = build_dataset(data, threshold=thresh)
    st.write("שורות בדאטה:", len(df_all))
    models, scores = train_ensemble(df_all, n_splits=4)
    st.write("CV (דיוק, F1):", scores)

    last = df_all.groupby('ticker').tail(1)
    X_last = last[FEATURES].values
    probs = ensemble_predict(models, X_last)
    last = last.assign(prob_up=probs).sort_values("prob_up", ascending=False)

    st.subheader("4) טבלת המלצות")
    picks = last.head(int(top_n)).copy()
    weights = picks['prob_up'] / picks['prob_up'].sum()
    picks['allocation_%'] = (weights * 100).round(2)
    picks['allocation_₪'] = (weights * total_capital).round(0)
    st.dataframe(picks[['ticker','Adj Close','prob_up','allocation_%','allocation_₪']].reset_index(drop=True), use_container_width=True)

    csv = picks[['ticker','Adj Close','prob_up','allocation_%','allocation_₪']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ הורד CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

    st.subheader("5) גרף למניה נבחרת")
    sel = st.selectbox("בחר מניה", options=picks['ticker'].tolist())
    if sel in data:
     df = data[sel].tail(250).copy()

    # אם אין עמודת Date (כשהיא באינדקס), נכניס אותה כעמודה רגילה
    if 'Date' not in df.columns:
        df = df.reset_index()

    # גרף מחיר
    st.plotly_chart(
        px.line(df, x='Date', y='Adj Close', title=f"{sel} — מחיר"),
        use_container_width=True
    )

    # גרף RSI
    st.plotly_chart(
        px.line(df, x='Date', y='rsi', title="RSI (14)"),
        use_container_width=True
    )  

    # התראות במייל
    strong = last[last['prob_up'] >= alert_threshold].copy()
    if len(strong) > 0:
        html = "<h3>איתותים חזקים</h3>" + "<br>".join(f"{t}: {p:.2f}" for t,p in zip(strong['ticker'], strong['prob_up']))
        send_alert("TA Advisor — הזדמנויות חזקות", html)
        st.success(f"נשלחה התראה במייל עבור {len(strong)} סימבולים (אם מוגדר SMTP).")

    st.caption(f"זמן ריצה: {int(time.time()-t0)} שניות")

st.caption("טיפ: ל-TA-125 אפשר להדביק קישור רכיבי מדד אחר (למשל index/168/components).")
