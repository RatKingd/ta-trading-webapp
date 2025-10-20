# בסיס קליל
FROM python:3.11-slim

# הכנות למודולים שבונים wheels (lxml ו-pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# התקנת תלויות
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# קבצי האפליקציה
COPY . .

# סטרימליט ירוץ על הפורט של Render
ENV PORT=8501
EXPOSE 8501

# הרצה
CMD ["bash", "-lc", "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"]
