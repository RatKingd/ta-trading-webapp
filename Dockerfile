# בסיס
FROM python:3.11-slim

# התקנת ספריות מערכת הנדרשות ל-lxml ו-html5lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית עבודה
WORKDIR /app

# העתקת קבצי הפרויקט
COPY . /app

# התקנת תלויות
RUN pip install --no-cache-dir -r requirements.txt

# פתיחת הפורט של Streamlit
EXPOSE 8501

# הפעלת האפליקציה
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
