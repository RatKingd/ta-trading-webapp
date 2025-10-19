import smtplib, ssl, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(subject: str, html_body: str):
    if os.getenv("ALERTS_ENABLED","false").lower() != "true":
        return
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT","587"))
    user = os.getenv("SMTP_USERNAME")
    pwd = os.getenv("SMTP_PASSWORD")
    to = os.getenv("ALERTS_TO")
    if not all([host,port,user,pwd,to]):  # אם חסר משהו — אל תשלח
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=ctx)
        server.login(user, pwd)
        server.sendmail(user, [to], msg.as_string())
