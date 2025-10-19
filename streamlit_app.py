# -*- coding: utf-8 -*-
import os
import time
import pickle
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# מודולים פנימיים
from data_fetcher import get_tickers_from_tase, download_price_history
from features import add_indicators
from model import build_dataset, train_ensemble, ensemble_predict, FEATURES
from alerts import send_alert

# -----------------------
# הגדרות כלליות
# -----------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))  # סף איתות ברירת מחדל

st.set_page_config(page_title="TA Trading WebApp", layout="wide")
st.title("דשבורד — TA-35 / TA-125 (Yahoo)")

with st.sidebar:
    st.markdown("### הגדרות")
    index_url = st.text_input(
        "קישור רכיבי מדד (TA-35/125)",
        value="https://market.tase.co.il/en/market_data/index/168/components",  # TA-125 (אפשר להדביק קישור TA-35)
    )
    total_capital = st.number_input("סכום להשקעה (₪)", value=100000.0, step=1000.0, format="%.2f")
    top_n = st.number_input("מספר פוזיציות", min_value=1, max_value=30, value=8, step=1)
    horizon = st.selectbox("אופק", options=["יומי", "שבועי"], index=0)
    ignore_cache = st.checkbox("להתעלם מ-cache בפעם הזו", value=False)

# הגנת סיסמה (אופציונלי)
if APP_PASSWORD:
    pw = st.sidebar.text_input("🔐 סיסמה", type="password")
    if pw != APP_PASSWORD:
        st.stop()

# -----------------------
# כפתור הרצה
# -----------------------
run = st.button("הרץ המלצות")

# מטמון מקומי (בזיכרון ריצה) לפי פרמטרים
cache_key = f"{index_url}|{horizon}"
_mem_cache = st.session_state.setdefault("_mem_cache", {})

if run:
    st.write("**1) שליפת רשימת טיקרים מה-TASE**")
    # קבצי Cache: רשימת טיקרים + נתונים
    tickers = None
    data = None

    # 1. טיקרים
    try:
        tickers = get_tickers_from_tase(index_url)
        if not tickers:
            raise RuntimeError("לא נמצאו סמלים מהרכיב שנבחר.")
        st.success(f"נמצאו {len(tickers)} סמלים. בודק עד 100 בלחיצה ראשונה.")
        # כדי להריץ מהר בפעם הראשונה – נגביל ל-100
        if len(tickers) > 100:
            tickers = tickers[:100]
    except Exception as e:
        st.error(f"שגיאה בשליפת טיקרים: {e}")
        st.stop()

    # 2. הורדת נתונים ויצירת אינדיקטורים
    st.write("**2) חישוב אינדיקטורים**")
    try:
        use_weekly = (horizon == "שבועי")
        # Cache בזיכרון עבור אותה ריצה
        if (not ignore_cache) and (cache_key in _mem_cache) and ("data" in _mem_cache[cache_key]):
            data = _mem_cache[cache_key]["data"]
        else:
            data = download_price_history(tickers, weekly=use_weekly)
            # הוספת אינדיקטורים לכל מניה
            for t in list(data.keys()):
                try:
                    df = add_indicators(data[t].copy())
                    # מסנן דאטה קצר מדי
                    if df.shape[0] < 90:
                        data.pop(t, None)
                    else:
                        data[t] = df
                except Exception:
                    data.pop(t, None)

            _mem_cache[cache_key] = {"data": data}

        st.success(f"התקבלו נתונים עבור {len(data)} מניות לאחר ניקוי.")

    except Exception as e:
        st.error(f"שגיאה בהורדת נתונים/אינדיקטורים: {e}")
        st.stop()

    # 3. בניית דטה-סט ואימון/תחזית
    st.write("**3) אימון מודל והפקת ציון**")
    try:
        df_all, last = build_dataset(data)        # df_all: היסטוריה מאוחדת; last: שורה אחרונה לכל טיקר
        models, scores = train_ensemble(df_all, n_splits=4)
        st.caption(f"CV (דיוק, F1): {scores}")

        X_last = last[FEATURES].values
        probs = ensemble_predict(models, X_last)

        last = last.assign(prob_up=probs).sort_values("prob_up", ascending=False)
        picks = last.head(int(top_n)).copy()

        # משקולות הקצאה
        weights = picks["prob_up"] / picks["prob_up"].sum()
        picks["allocation_%"] = (weights * 100).round(2)
        picks["allocation_₪"] = (weights * total_capital).round(0)

        st.subheader("4) טבלת המלצות")
        show_cols = ["ticker", "Adj Close", "prob_up", "allocation_%", "allocation_₪"]
        show_cols = [c for c in show_cols if c in picks.columns]
        st.dataframe(picks[show_cols].reset_index(drop=True), use_container_width=True)

        # כפתור הורדת CSV
        csv = picks[["ticker", "Adj Close", "prob_up", "allocation_%", "allocation_₪"]].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ הורד CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

    except Exception as e:
        st.error(f"שגיאה באימון/חיזוי: {e}")
        st.stop()

    # 4. גרף למניה נבחרת — עמיד לכל המבנים
    st.subheader("5) גרף למניה נבחרת")

    # גיבוי: אם picks ריק/לא קיים—נבחר מתוך data
    if isinstance(locals().get("picks", None), pd.DataFrame) and not picks.empty:
        options = picks["ticker"].tolist()
    else:
        options = sorted(list(data.keys()))[:min(20, len(data))]
        if not options:
            st.info("אין מניות להצגה.")
            st.stop()

    sel = st.selectbox("בחר מניה", options=options)

    if sel in data:
        df = data[sel].tail(250).copy()

        # אם התאריך יושב באינדקס—להכניס כעמודה
        if "Date" not in df.columns:
            df = df.reset_index()

        # אם יש MultiIndex בעמודות (למשל ('Adj Close','POLI.TA')) – נשטח לשמות פשוטים
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join([str(x) for x in tup if str(x) != ""]).strip()
                for tup in df.columns
            ]

        # למצוא עמודת מחיר מתאימה (Adj Close או Adj Close <סימבול>)
        cand_cols = [c for c in df.columns if c.lower().startswith("adj close")]
        y_col = cand_cols[0] if cand_cols else ("Adj Close" if "Adj Close" in df.columns else None)

        # דאגה לעמודת Date
        if "Date" not in df.columns:
            if "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            else:
                df["Date"] = range(len(df))  # גיבוי

        # גרף מחיר (אם נמצאה עמודת יעד)
        if y_col:
            st.plotly_chart(
                px.line(df, x="Date", y=y_col, title=f"{sel} — מחיר"),
                use_container_width=True,
            )
        else:
            st.info("לא נמצאה עמודת מחיר לציור.")

        # גרף RSI — רק אם קיים
        rsi_col = [c for c in df.columns if str(c).lower().startswith("rsi")]
        if rsi_col:
            st.plotly_chart(
                px.line(df, x="Date", y=rsi_col[0], title="RSI (14)"),
                use_container_width=True,
            )

    # 6) התראות במייל (אופציונלי)
    try:
        strong = last[last["prob_up"] >= ALERT_THRESHOLD].copy()
        if len(strong) > 0:
            html = "<h3>מניות בעוצמה</h3><br>" + "<br>".join(
                f"{t}: {p:.2f}" for t, p in zip(strong["ticker"], strong["prob_up"])
            )
            send_alert("TA Advisor — איתותים חזקים", html)  # עובד רק אם SMTP מוגדר ב-.env
            st.success(f"נשלחו {len(strong)} התראות (אם SMTP מוגדר).")
    except Exception:
        pass

    st.caption(f"משך ריצה (שניות): {int(time.time() - st.session_state.get('_t0', time.time()))}")

# חיווי build/commit מהריצה בענן (Render)
st.caption(f"Build: {os.getenv('RENDER_GIT_COMMIT', '')[:7]} | Branch: {os.getenv('RENDER_GIT_BRANCH', '')}")
