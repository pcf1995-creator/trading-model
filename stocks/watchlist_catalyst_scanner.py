"""
stocks/watchlist_catalyst_scanner.py — Idea Watchlist catalyst/date alert.

Invoked by GitHub Actions daily (same schedule as the SMA scanner).

Scans the free-text "notes" field of every idea_watchlist entry for mentions
of upcoming dates (earnings, conferences, buybacks, announcements — whatever
you pasted in when you logged the ticker) and emails a reminder for anything
landing within the next CATALYST_LOOKAHEAD_DAYS. This is intentionally a
reminder, not a one-time alert — a date inside the window shows up every day
until it passes, same as a calendar reminder.

Required env vars (same secrets as sma_scanner.py):
  SUPABASE_URL, SUPABASE_KEY
  SMTP_USER, SMTP_APP_PASSWORD
"""

import os
import re
import smtplib
import sys
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_STOCKS_DIR = Path(__file__).parent
for _p in (_REPO_ROOT, _STOCKS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import db  # noqa: E402  → stocks/db.py

ALERT_EMAIL = "pf@hyperspaceventures.com"
CATALYST_LOOKAHEAD_DAYS = 7

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))
_DATE_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)


def extract_dates(text: str, today: date) -> list[dict]:
    """Find month/day (optionally year) mentions in free text, resolve them to
    real dates, and return each with a short surrounding-context snippet."""
    if not text:
        return []
    found = []
    for m in _DATE_RE.finditer(text):
        month = _MONTHS[m.group(1).lower()]
        try:
            day = int(m.group(2))
        except ValueError:
            continue
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        # No explicit year given and the date's already well in the past this
        # year → assume it means next year (these are pasted as "upcoming").
        if not m.group(3) and d < today - timedelta(days=45):
            try:
                d = date(year + 1, month, day)
            except ValueError:
                continue
        start = max(0, m.start() - 40)
        end   = min(len(text), m.end() + 40)
        snippet = text[start:end].strip().replace("\n", " ")
        found.append({"date": d, "snippet": snippet})
    return found


def send_email(hits: list[dict]) -> None:
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD", "")
    if not smtp_user or not smtp_pass:
        print("[watchlist_catalyst] SMTP credentials not configured — skipping email")
        return

    today = date.today().isoformat()
    n = len(hits)
    subject = f"Watchlist Catalysts: {n} upcoming in the next {CATALYST_LOOKAHEAD_DAYS} days — {today}"

    rows = ""
    for h in hits:
        days_out = (h["date"] - date.today()).days
        urgency = "red" if days_out <= 1 else "darkorange" if days_out <= 3 else "#555"
        rows += (
            f"<tr>"
            f"<td><b>{h['ticker']}</b></td>"
            f"<td>{h['date'].isoformat()}</td>"
            f"<td style='color:{urgency}'>{days_out}d</td>"
            f"<td>{h['snippet']}</td>"
            f"</tr>"
        )

    html = f"""
    <html><body style='font-family:sans-serif'>
    <h2 style='margin-bottom:4px'>Idea Watchlist — Upcoming Catalysts</h2>
    <p style='color:gray;margin-top:0'>Dates parsed from your watchlist notes, next {CATALYST_LOOKAHEAD_DAYS} days</p>
    <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse'>
    <tr style='background:#f0f0f0'><th>Ticker</th><th>Date</th><th>In</th><th>Context</th></tr>
    {rows}
    </table>
    <p style='color:gray;font-size:11px;margin-top:16px'>
    This is a reminder, not a one-time alert — a date stays listed every day until it passes.
    Parsed automatically from free text; verify actual dates before trading around them.
    </p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = ALERT_EMAIL
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    print(f"[watchlist_catalyst] Email sent → {ALERT_EMAIL}: {subject}")


def main() -> None:
    today = date.today()
    print(f"[watchlist_catalyst] Scanning watchlist notes for catalysts — {today}")

    items = db.load_watchlist()
    print(f"[watchlist_catalyst] {len(items)} watchlist items")

    hits = []
    for item in items:
        ticker = item.get("ticker", "?")
        for d in extract_dates(item.get("notes") or "", today):
            days_out = (d["date"] - today).days
            if 0 <= days_out <= CATALYST_LOOKAHEAD_DAYS:
                hits.append({"ticker": ticker, "date": d["date"], "snippet": d["snippet"]})
                print(f"[watchlist_catalyst] {ticker}: {d['date']} ({days_out}d) — {d['snippet']}")

    hits.sort(key=lambda h: h["date"])

    if hits:
        send_email(hits)
    else:
        print("[watchlist_catalyst] No upcoming catalysts in window — no email sent")


if __name__ == "__main__":
    main()
