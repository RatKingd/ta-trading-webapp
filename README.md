# TA Trading WebApp (TASE)

אפליקציית Streamlit שמביאה רשימת רכיבי מדד TA-35/TA-125 (מתוך אתר הבורסה),
מורידה נתוני מחיר מ-Yahoo Finance, מוסיפה אינדיקטורים, מאמנת מודל פשוט,
ומציגה טבלת הקצאות + גרפים.

## ריצה מקומית
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py

## דיפלוי ב-Render (Docker)
- חברו את הרפו
- אין צורך ב-Start Command (מוגדר ב-Dockerfile)
- Auto Deploy: On Commit
