"""
stocks/sma_scanner.py — Daily 200-day SMA crossunder scanner.

Invoked by GitHub Actions at 10 AM ET on weekdays.

Loads the scanner universe from the Supabase sma_scanner_universe table,
fetches ~1 year of daily closes for each ticker via yfinance, computes the
200-day SMA, and emails:

  🔴 Crossunders  — closed ABOVE SMA yesterday, at/BELOW today (first signal)
  🟡 Approaching  — within 2% above the SMA but not yet crossed

Required env vars (set as GitHub Actions secrets):
  SUPABASE_URL, SUPABASE_KEY
  SMTP_USER          — Gmail address (sender and recipient)
  SMTP_APP_PASSWORD  — Gmail app password
"""

import os
import smtplib
import sys
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).parent.parent
_STOCKS_DIR = Path(__file__).parent
for _p in (_REPO_ROOT, _STOCKS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.db_common import _get_client

ALERT_EMAIL = "pf@hyperspaceventures.com"
SMA_PERIOD  = 200


def load_universe() -> list[str]:
    client = _get_client()
    if client:
        try:
            resp = client.table("sma_scanner_universe").select("ticker").execute()
            tickers = [r["ticker"] for r in (resp.data or [])]
            if tickers:
                print(f"[sma_scanner] Loaded {len(tickers)} tickers from Supabase")
                return tickers
        except Exception as e:
            print(f"[sma_scanner] Supabase load failed: {e}")
    # Fallback: ticker_summary.csv if it exists in the repo
    csv = _STOCKS_DIR / "ticker_summary.csv"
    if csv.exists():
        print("[sma_scanner] Falling back to ticker_summary.csv")
        return pd.read_csv(csv)["Ticker"].tolist()
    raise RuntimeError("No scanner universe available — run refresh_sma_universe.py first")


def fetch_sma_data(ticker: str) -> dict | None:
    """Fetch 200-day SMA data for ticker. Returns None on insufficient history."""
    try:
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        closes = hist["Close"].dropna()
        if len(closes) < SMA_PERIOD + 1:
            return None
        sma = closes.rolling(SMA_PERIOD).mean()
        prev_close  = float(closes.iloc[-2])
        today_close = float(closes.iloc[-1])
        prev_sma    = float(sma.iloc[-2])
        today_sma   = float(sma.iloc[-1])
        return {
            "ticker":      ticker,
            "prev_close":  prev_close,
            "today_close": today_close,
            "prev_sma":    prev_sma,
            "today_sma":   today_sma,
            "pct_vs_sma":  (today_close - today_sma) / today_sma * 100,
        }
    except Exception as e:
        print(f"[sma_scanner] {ticker} fetch failed: {e}")
        return None


def _table_rows(rows: list[dict]) -> str:
    out = ""
    for r in rows:
        color = "red" if r["pct_vs_sma"] <= 0 else "darkorange"
        out += (
            f"<tr>"
            f"<td><b>{r['ticker']}</b></td>"
            f"<td>${r['today_close']:.2f}</td>"
            f"<td>${r['today_sma']:.2f}</td>"
            f"<td style='color:{color}'>{r['pct_vs_sma']:+.2f}%</td>"
            f"</tr>"
        )
    return out


def send_email(crossunders: list[dict], approaching: list[dict]) -> None:
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD", "")
    if not smtp_user or not smtp_pass:
        print("[sma_scanner] SMTP credentials not configured — skipping email")
        return

    today = date.today().isoformat()
    n = len(crossunders)
    subject = (
        f"SMA Alert {today}: {n} ticker{'s' if n != 1 else ''} crossed below 200-day SMA"
        if crossunders
        else f"SMA Scan {today}: {len(approaching)} approaching 200d SMA"
        if approaching
        else f"SMA Scan {today}: No alerts"
    )

    cross_section = (
        "<p>None today.</p>"
        if not crossunders
        else f"""
        <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse'>
        <tr style='background:#f0f0f0'><th>Ticker</th><th>Close</th><th>200d SMA</th><th>vs SMA</th></tr>
        {_table_rows(crossunders)}
        </table>"""
    )
    approach_section = (
        "<p>None today.</p>"
        if not approaching
        else f"""
        <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse'>
        <tr style='background:#f0f0f0'><th>Ticker</th><th>Close</th><th>200d SMA</th><th>vs SMA</th></tr>
        {_table_rows(approaching)}
        </table>"""
    )

    html = f"""
    <html><body style='font-family:sans-serif'>
    <h2 style='margin-bottom:4px'>200-Day SMA Scanner — {today}</h2>
    <p style='color:gray;margin-top:0'>S&amp;P 500 constituents with market cap &ge; $10B</p>

    <h3>🔴 Crossed Below 200-Day SMA ({len(crossunders)})</h3>
    {cross_section}

    <h3>🟡 Approaching 200-Day SMA — within 2% above ({len(approaching)})</h3>
    {approach_section}

    <p style='color:gray;font-size:11px;margin-top:24px'>
    Crossunder = closed above 200d SMA yesterday, at/below today (first-signal only).
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
    print(f"[sma_scanner] Email sent → {ALERT_EMAIL}: {subject}")


def main() -> None:
    today = date.today()
    print(f"[sma_scanner] Running scan for {today}")

    universe = load_universe()
    print(f"[sma_scanner] Scanning {len(universe)} tickers…")

    crossunders: list[dict] = []
    approaching: list[dict] = []

    for ticker in universe:
        data = fetch_sma_data(ticker)
        if data is None:
            continue
        # First crossunder: was above SMA yesterday, at/below today
        if data["prev_close"] > data["prev_sma"] and data["today_close"] <= data["today_sma"]:
            crossunders.append(data)
            print(f"[sma_scanner] CROSSUNDER {ticker}: close={data['today_close']:.2f} sma={data['today_sma']:.2f} ({data['pct_vs_sma']:+.2f}%)")
        # Approaching: within 2% above SMA, not yet crossed
        elif 0 < data["pct_vs_sma"] <= 2.0:
            approaching.append(data)
            print(f"[sma_scanner] APPROACHING {ticker}: {data['pct_vs_sma']:+.2f}% above SMA")

    crossunders.sort(key=lambda r: r["pct_vs_sma"])
    approaching.sort(key=lambda r: r["pct_vs_sma"])

    print(f"[sma_scanner] {len(crossunders)} crossunders, {len(approaching)} approaching")

    if crossunders or approaching:
        send_email(crossunders, approaching)
    else:
        print("[sma_scanner] No alerts — no email sent")


if __name__ == "__main__":
    main()
