def send_alert(subject: str, html: str) -> None:
    """
    סטאב. אם תרצה מיילים בעתיד נוסיף SMTP.
    כרגע רק 'מדפיס' ללוג של השרת.
    """
    print("[ALERT]", subject)
    print(html[:500], "...")
