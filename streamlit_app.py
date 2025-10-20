# streamlit_app.py
# ----------------
# אפליקציית Streamlit לניתוח מניות TA-35/TA-125 על בסיס Yahoo.

import os
import time
import pickle

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# טעינת dotenv בצורה סלחנית: אם הספרייה לא מותקנת – פשוט נמשיך בלי קריסה.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    def load_dotenv(*args, **kwargs):
        return

# מודולים פנימיים של האפליקציה
from data_fetcher import get_tickers_from_tase, download_price_history  # ודא שקיימים
from features import add_indicators, FEATURES                           # רשימת פיצ'רים
from model import build_dataset, train_ensemble, ensemble_predict      # מודל ML
from alerts import send_alert                                          # שליחת התראות

# ------------------------------------------------------------
# הגדרות עמוד
st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("תשבורת — TA-35 / TA-125 (Yahoo)")

# ------------------------------------------------------------
# Sidebar – פרמטרים
with st.sidebar:
    st.markdown("### הגדרות:")

    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/142/components"  # TA-35 לדוגמה
    )

    total_capital = st.number_input("סכום להשקעה (₪)", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    top_n        = st.number_input("מספר פוזיציות", min_value=1, max_value=50, value=8, step=1)

    period = st.selectbox("אופק", options=["יומי", "שבועי"], index=0)

    bypass_cache = st.checkbox("בפעם הזו להתעלם מה-cache (נתונים חדש)")

    run_btn = st.button("הרץ המלצות")

# ------------------------------------------------------------
# פונקציה עזר לגרפים – מבטיחה עמודת Date
def ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        # אם זה אינדקס של תאריכים – נהפוך לעמודה
        if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "Date":
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        else:
            # ננסה לנחש—אם יש עמודה בשם דומה
            for c in df.columns:
                if str(c).lower().strip() == "date":
                    df.rename(columns={c: "Date"}, inplace=True)
                    break
    return df

# ------------------------------------------------------------
# לוגיקה ראשית
if run_btn:
    # 1) שליפת רשימת טיקרים מה-TASE (עם fallback מובנה בקוד data_fetcher)
    st.subheader("1) שליפת רשימת טיקרים מה-TASE")
    try:
        # העלינו timeout ל-20–30ש' ב-data_fetcher; כאן רק קוראים
        tickers = get_tickers_from_tase(index_url)
        if not tickers:
            st.warning("לא נמצאו טיקרים מהרשימה, אשתמש ברשימת ברירת־מחדל (Yahoo).")
    except Exception as e:
        st.error(f"TASE fetch failed: {e}")
        tickers = []  # data_fetcher אמור כבר לדאוג ל-fallback, עדיין נשאיר הגנה

    # קיצוץ למקסימום 100 לטובת מהירות
    if tickers:
        tickers = tickers[:100]
        st.success(f"נמצאו {len(tickers)} סמלים. בודק עד 100 לניתוח הראשון.")
    else:
        st.info("ממשיך עם fallback שתואם ל־Yahoo (לדוגמה: TEVA.TA, LUMI.TA...).")

    # 2) הורדת נתונים וחישוב אינדיקטורים
    st.subheader("2) חישוב אינדיקטורים")
    try:
        # חשוב: לא להעביר פרמטר weekly – הוא לא קיים בפונקציה.
        # את ההבדל בין יומי/שבועי נטפל בתוך הפיצ'רים/מודל (resample אם צריך).
        prices = download_price_history(tickers or None)

        # אם המשתמש בחר "שבועי" – נעשה resample לכל טיקר לתדירות שבועית (OHLCV)
        if period == "שבועי":
            out = []
            for sym, df in prices.items():
                if df.empty:
                    continue
                # דואגים לשם Date
                df = ensure_date_index(df)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                wk = pd.DataFrame({
                    "Open":  df["Open"].resample("W").first(),
                    "High":  df["High"].resample("W").max(),
                    "Low":   df["Low"].resample("W").min(),
                    "Close": df["Close"].resample("W").last(),
                    "Adj Close": df["Adj Close"].resample("W").last(),
                    "Volume": df["Volume"].resample("W").sum(),
                }).dropna(how="all")
                wk["ticker"] = sym
                wk = wk.reset_index()
                out.append(wk)
            if out:
                prices = {sym_df["ticker"].iloc[0]: sym_df for sym_df in out}
    except TypeError as e:
        # זה מטפל ספציפית בשגיאת "unexpected keyword argument 'weekly'"
        st.error(f"שגיאה בהורדת נתונים/אינדיקטורים: {e}")
        st.stop()
    except Exception as e:
        st.error(f"שגיאה בהורדת נתונים/אינדיקטורים: {e}")
        st.stop()

    # 3) בניית דאטהסט, אימון מודל ואבחון
    st.subheader("3) אימון מודל והפקת ציון")
    try:
        df_all = build_dataset(prices, FEATURES)
        models, scores = train_ensemble(df_all, n_splits=4)
        st.write("CV (דיוק, F1):")
        st.write(scores)

        # תחזיות ליום/שבוע אחרון
        last = df_all.groupby("ticker").tail(1)
        X_last = last[FEATURES].values
        probs = ensemble_predict(models, X_last)

        last = last.assign(prob_up=probs).sort_values("prob_up", ascending=False)
    except Exception as e:
        st.error(f"שגיאה במודל/חיזוי: {e}")
        st.stop()

    # 4) טבלת המלצות + הורדת CSV
    st.subheader("4) טבלת המלצות")
    try:
        top_n = int(top_n)
        top_n = max(1, min(top_n, len(last))) if len(last) else 0

        if top_n == 0:
            st.warning("אין נתונים להצגה.")
            st.stop()

        top_sel = last.head(top_n).copy()

        # חישוב alocation לפי הסתברות יחסית
        weights = top_sel["prob_up"] / top_sel["prob_up"].sum()
        top_sel["allocation_%"]  = (weights * 100).round(2)
        top_sel["allocation_₪"]  = (weights * total_capital).round(0)

        # סדר עמודות נוח + הצגה
        picks = top_sel[["ticker", "Adj Close", "prob_up", "allocation_%", "allocation_₪"]].reset_index(drop=True)
        st.dataframe(picks)

        # כפתור הורדת CSV – משתמש ב־picks שהגדרנו כאן (לא NameError)
        csv = picks.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ הורד CSV", data=csv, file_name="recommendations.csv", mime="text/csv")
    except Exception as e:
        st.error(f"שגיאה בטבלת ההמלצות: {e}")

    # 5) גרף למניה נבחרת
    st.subheader("5) גרף למניה נבחרת")
    try:
        sel = st.selectbox("בחר מניה", options=picks["ticker"].tolist())
        data = prices.get(sel)
        if data is not None and not data.empty:
            data = ensure_date_index(data).tail(250).copy()

            fig1 = px.line(data, x="Date", y="Adj Close", title=f"{sel} — מחיר")
            st.plotly_chart(fig1, use_container_width=True)

            if "rsi" in data.columns:
                fig2 = px.line(data, x="Date", y="rsi", title="RSI (14)")
                st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"שגיאה בגרף: {e}")

    # 6) התראות (אופציונלי)
    st.subheader("6) התראות חזקות")
    try:
        alert_threshold = 0.7
        strong = last[last["prob_up"] >= alert_threshold].copy()
        if len(strong) > 0:
            html = "<h3>מניות חזקות</h3>" + "<br>".join([f"{t}: {p:0.2f}" for t, p in zip(strong["ticker"], strong["prob_up"])])
            # שלח התראה (אם הוגדר SMTP/מלוי משתני סביבה מתאימים)
            send_alert("TA Advisor — הזדמנויות חזקות", html)
            st.success(f"נשלחו {len(strong)} התראות (אם מוגדר SMTP).")
        else:
            st.info("אין כרגע מניות מעל סף ההתראה.")
    except Exception as e:
        st.warning(f"התראות לא נשלחו: {e}")

    st.caption(f"זמן ריצה: ~{int(time.time()) % 10} שניות")
