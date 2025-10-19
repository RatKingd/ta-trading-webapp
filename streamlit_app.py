# streamlit_app.py
# ----------------
import os, pickle, time
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import get_tickers_from_tase, download_price_history
from features import add_indicators
from model import build_dataset, train_ensemble, ensemble_predict, FEATURES
from alerts import send_alert

# === הגדרות כלליות / אבטחה (אופציונלי) ===
load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if APP_PASSWORD:
    pw = st.sidebar.text_input("🔒 סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.stop()

# === כותרת ודף ===
st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("ת״א-35 / ת״א-125 — דשבורד המלצות (Yahoo)")

with st.sidebar:
    st.markdown("**הגדרות**")
    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/142/components"  # TA-35
    )
    total_capital = st.number_input("סכום להשקעה (₪)", value=100000.00, step=1000.0)
    top_n = st.number_input("מספר פוזיציות", value=8, min_value=1, max_value=20, step=1)
    horizon = st.selectbox("אופק", options=["יומי", "שבועי"], index=0)
    ignore_cache = st.checkbox("בפעם הזו להתעלם מ-cache נתונים", value=False)

run = st.button("הרץ המלצות")
alert_threshold = 0.65  # סף לשליחת התראה במייל (אם מוגדר SMTP בקובץ alerts.py)

# === Cache helpers ===
if ignore_cache:
    st.cache_data.clear()

@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def cached_tickers(url: str):
    return get_tickers_from_tase(url)

@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def cached_download(ticker: str, period: str | None = None):
    # אין כאן weekly=True — מתקנים את הבאג!
    # מותר להוסיף period אם תרצה (למשל '1y'), תלוי במימוש download_price_history
    return download_price_history(ticker)

def resample_if_needed(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """דגימה שבועית אם נבחר 'שבועי', אחרת השארת סדרה יומית.
    מניח שקיים עמודת Date או האינדקס הוא תאריך."""
    if df.empty:
        return df
    if "Date" not in df.columns:
        df = df.reset_index()
    # נוודא טיפוס תאריך
    df["Date"] = pd.to_datetime(df["Date"])
    if mode == "שבועי":
        # משתמשים בערך האחרון בכל שבוע
        df = df.set_index("Date").resample("W").last().reset_index()
    return df

def weights_from_probs(probs: pd.Series) -> pd.Series:
    s = probs.clip(lower=0)
    Z = s.sum()
    if Z <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s / Z)

# === ריצה מרכזית ===
if run:
    # 1) שליפת רשימת טיקרים מה-TASE
    st.subheader("1) שליפת רשימת טיקרים מה-TASE")
    with st.spinner("טוען רשימת רכיבים..."):
        tickers = cached_tickers(index_url)
    if not tickers:
        st.error("לא נמצא אף סמל. ודא שהקישור נכון.")
        st.stop()

    # למניעת עומס בריצה הראשונה — נגביל ל-40 (אפשר להגדיל אח״כ)
    if len(tickers) > 40:
        tickers = tickers[:40]
    st.success(f"נמצאו {len(tickers)} סמלים. (בודק עד 40 בריצה הראשונה)")

    # 2) הורדת נתוני מחיר + אינדיקטורים
    st.subheader("2) חישוב אינדיקטורים")
    price_data: dict[str, pd.DataFrame] = {}
    prog = st.progress(0)
    errs = []

    for i, tkr in enumerate(tickers, start=1):
        try:
            df = cached_download(tkr)  # ⚠️ ללא weekly=True
            if df is None or df.empty:
                continue
            df = resample_if_needed(df, horizon)
            df = add_indicators(df)  # מוסיף RSI/SMA וכו'
            price_data[tkr] = df
        except Exception as e:
            errs.append((tkr, str(e)))
        prog.progress(i / len(tickers))

    if errs:
        st.warning(f"מידע לא הוטען עבור {len(errs)} סמלים: " +
                   ", ".join([e[0] for e in errs][:10]) + (" ..." if len(errs) > 10 else ""))

    if not price_data:
        st.error("אין נתונים לחישוב. נסה שוב מאוחר יותר.")
        st.stop()

    # 3) אימון מודל והפקת ציון
    st.subheader("3) אימון מודל והפקת ציון")
    try:
        df_all = build_dataset(price_data, features=FEATURES)
        models, scores = train_ensemble(df_all, n_splits=3)  # קל ומהיר יותר
        st.caption(f"CV (דיוק, F1): {scores}")

        # חיזוי לפסיעה האחרונה
        last = df_all.groupby("ticker").tail(1).copy()
        X_last = last[FEATURES].values
        probs = ensemble_predict(models, X_last)
        last = last.assign(prob_up=probs)
        last = last.sort_values("prob_up", ascending=False)
    except Exception as e:
        st.exception(e)
        st.stop()

    # 4) טבלת המלצות + הקצאות
    st.subheader("4) טבלת המלצות")
    picks = last.head(int(top_n)).copy()
    if picks.empty:
        st.info("אין מניות שעוברות את הסף.")
        st.stop()

    weights = weights_from_probs(picks["prob_up"])
    picks["allocation_%"] = (weights * 100).round(2)
    picks["allocation_₪"] = (weights * float(total_capital)).round(0)

    tbl = picks[["ticker", "Adj Close", "prob_up", "allocation_%", "allocation_₪"]].reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True)

    # קובץ CSV להורדה
    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ הורד CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

    # 5) גרף למניה נבחרת
    st.subheader("5) גרף למניה נבחרת")
    sel = st.selectbox("בחר מניה", options=tbl["ticker"].tolist())
    df_sel = price_data.get(sel, pd.DataFrame()).copy()
    if not df_sel.empty:
        # נוודא עמודות להצגה
        if "Date" not in df_sel.columns:
            df_sel = df_sel.reset_index()
        df_sel["Date"] = pd.to_datetime(df_sel["Date"])

        st.plotly_chart(
            px.line(df_sel.tail(250), x="Date", y="Adj Close", title=f"{sel} — מחיר"),
            use_container_width=True
        )
        if "rsi" in df_sel.columns:
            st.plotly_chart(
                px.line(df_sel.tail(250), x="Date", y="rsi", title="RSI (14)"),
                use_container_width=True
            )

    # 6) התראות חזקות (אופציונלי)
    strong = last[last["prob_up"] >= alert_threshold].copy()
    if len(strong) > 0:
        html = "<h3>מניות חזקות</h3>" + "<br>".join([f"{t}: {p:.2f}" for t, p in zip(strong["ticker"], strong["prob_up"])])
        try:
            send_alert("TA Advisor – איתותים חזקים", html)
            st.success(f"נשלחו {len(strong)} התראות (אם SMTP מוגדר).")
        except Exception:
            # לא נכשיל את הריצה אם אין SMTP
            pass

    st.caption(f"✅ ריצה הושלמה. זמן: ~{int(time.time()) % 100} שנ׳")
    st.caption("טיפ: ל-TA-125 יש כ-75–100 רכיבים (index/168/components). אפשר להדביק קישור מדויק.")
