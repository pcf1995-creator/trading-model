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
    # 1. Supabase curated universe (populated by refresh_sma_universe.py)
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

    # 2. Fetch full S&P 500 from Wikipedia (works on GH Actions / Python 3.11 + lxml)
    print("[sma_scanner] Fetching S&P 500 from Wikipedia…")
    try:
        sp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp_df  = pd.read_html(sp_url)[0]
        tickers = sp_df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"[sma_scanner] Fetched {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"[sma_scanner] Wikipedia fetch failed: {e}")

    # 3. Last resort: ticker_summary.csv already in the repo
    csv = _STOCKS_DIR / "ticker_summary.csv"
    if csv.exists():
        print("[sma_scanner] Falling back to ticker_summary.csv")
        return pd.read_csv(csv)["Ticker"].tolist()

    raise RuntimeError("No scanner universe available")


SMA_FAST = 50


def fetch_all_sma(tickers: list[str]) -> dict[str, dict]:
    """Batch-download 1yr of closes for every ticker at once and compute 50/200d SMAs.

    Far faster and more robust than per-ticker requests (which hang/rate-limit on
    ~500 names and get the GitHub Actions job cancelled).
    """
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False, threads=True)
    if data.empty:
        return {}
    closes = data["Close"] if "Close" in data.columns else data.xs("Close", axis=1, level=0)

    out: dict[str, dict] = {}
    for tk in tickers:
        try:
            s = closes[tk].dropna() if tk in closes.columns else None
            if s is None or len(s) < SMA_PERIOD + 1:
                continue
            sma200 = s.rolling(SMA_PERIOD).mean()
            sma50  = s.rolling(SMA_FAST).mean()
            today_close = float(s.iloc[-1])
            today_sma   = float(sma200.iloc[-1])
            out[tk] = {
                "ticker":       tk,
                "prev_close":   float(s.iloc[-2]),
                "today_close":  today_close,
                "prev_sma":     float(sma200.iloc[-2]),
                "today_sma":    today_sma,
                "prev_sma50":   float(sma50.iloc[-2]),
                "today_sma50":  float(sma50.iloc[-1]),
                "pct_vs_sma":   (today_close - today_sma) / today_sma * 100,
            }
        except Exception as e:
            print(f"[sma_scanner] {tk} parse failed: {e}")
            continue
    return out


def _table_rows(rows: list[dict]) -> str:
    out = ""
    for r in rows:
        color = "red" if r["pct_vs_sma"] <= 0 else "darkorange"
        uptrend = r.get("today_sma50", 0) > r.get("today_sma", 0)
        trend = (
            "<span style='color:green'>🟢 dip in uptrend</span>"
            if uptrend
            else "<span style='color:#b00'>🔴 downtrend — avoid</span>"
        )
        out += (
            f"<tr>"
            f"<td><b>{r['ticker']}</b></td>"
            f"<td>${r['today_close']:.2f}</td>"
            f"<td>${r['today_sma']:.2f}</td>"
            f"<td style='color:{color}'>{r['pct_vs_sma']:+.2f}%</td>"
            f"<td>{trend}</td>"
            f"</tr>"
        )
    return out


def _golden_rows(rows: list[dict]) -> str:
    """Rows for the golden-cross table: 50d SMA, 200d SMA, close."""
    out = ""
    for r in rows:
        out += (
            f"<tr>"
            f"<td><b>{r['ticker']}</b></td>"
            f"<td>${r['today_close']:.2f}</td>"
            f"<td>${r['today_sma50']:.2f}</td>"
            f"<td>${r['today_sma']:.2f}</td>"
            f"</tr>"
        )
    return out


def send_email(golden: list[dict], crossunders: list[dict], approaching: list[dict]) -> None:
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD", "")
    if not smtp_user or not smtp_pass:
        print("[sma_scanner] SMTP credentials not configured — skipping email")
        return

    today = date.today().isoformat()
    g = len(golden)
    subject = (
        f"SMA Alert {today}: {g} golden cross{'es' if g != 1 else ''} 🟢"
        if golden
        else f"SMA Alert {today}: {len(crossunders)} crossed below 200d SMA"
        if crossunders
        else f"SMA Scan {today}: {len(approaching)} approaching 200d SMA"
        if approaching
        else f"SMA Scan {today}: No alerts"
    )

    golden_section = (
        "<p>None today.</p>"
        if not golden
        else f"""
        <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse'>
        <tr style='background:#e6f4ea'><th>Ticker</th><th>Close</th><th>50d SMA</th><th>200d SMA</th></tr>
        {_golden_rows(golden)}
        </table>"""
    )
    cross_section = (
        "<p>None today.</p>"
        if not crossunders
        else f"""
        <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse'>
        <tr style='background:#f0f0f0'><th>Ticker</th><th>Close</th><th>200d SMA</th><th>vs SMA</th><th>50d vs 200d</th></tr>
        {_table_rows(crossunders)}
        </table>"""
    )
    approach_section = (
        "<p>None today.</p>"
        if not approaching
        else f"""
        <table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse'>
        <tr style='background:#f0f0f0'><th>Ticker</th><th>Close</th><th>200d SMA</th><th>vs SMA</th><th>50d vs 200d</th></tr>
        {_table_rows(approaching)}
        </table>"""
    )

    html = f"""
    <html><body style='font-family:sans-serif'>
    <h2 style='margin-bottom:4px'>200-Day SMA Scanner — {today}</h2>
    <p style='color:gray;margin-top:0'>S&amp;P 500 constituents with market cap &ge; $10B</p>

    <h3>🟢 Golden Cross — 50d crossed above 200d ({len(golden)})</h3>
    <p style='color:gray;margin-top:0;font-size:12px'>Bullish trend-confirmation buy signal — the names to act on.</p>
    {golden_section}

    <h3>🔴 Crossed Below 200-Day SMA ({len(crossunders)}) — watch list, not a buy</h3>
    <p style='color:gray;margin-top:0;font-size:12px'>Only a dip-buy candidate when the 50d is still
    above the 200d (🟢). If the 50d is below the 200d (🔴), it's a downtrend — likely a falling knife.</p>
    {cross_section}

    <h3>🟡 Approaching 200-Day SMA — within 2% above ({len(approaching)})</h3>
    {approach_section}

    <p style='color:gray;font-size:11px;margin-top:24px'>
    <b>Golden cross</b> = 50d SMA closed at/below the 200d yesterday and above it today (first-signal only) —
    the evidence-backed buy.<br>
    <b>Crossunder / approaching</b> = price at the 200d. On its own this is buying into weakness; treat it as a
    watch list, and only as a buy when the 50d is above the 200d (a pullback within an uptrend).
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

    all_data = fetch_all_sma(universe)
    print(f"[sma_scanner] Computed SMAs for {len(all_data)} tickers")

    golden: list[dict] = []
    crossunders: list[dict] = []
    approaching: list[dict] = []

    for ticker, data in all_data.items():
        # Golden cross: 50d SMA closed at/below 200d yesterday, above it today
        if data["prev_sma50"] <= data["prev_sma"] and data["today_sma50"] > data["today_sma"]:
            golden.append(data)
            print(f"[sma_scanner] GOLDEN CROSS {ticker}: 50d={data['today_sma50']:.2f} > 200d={data['today_sma']:.2f}")
        # First crossunder: was above SMA yesterday, at/below today
        if data["prev_close"] > data["prev_sma"] and data["today_close"] <= data["today_sma"]:
            crossunders.append(data)
            print(f"[sma_scanner] CROSSUNDER {ticker}: close={data['today_close']:.2f} sma={data['today_sma']:.2f} ({data['pct_vs_sma']:+.2f}%)")
        # Approaching: within 2% above SMA, not yet crossed
        elif 0 < data["pct_vs_sma"] <= 2.0:
            approaching.append(data)
            print(f"[sma_scanner] APPROACHING {ticker}: {data['pct_vs_sma']:+.2f}% above SMA")

    golden.sort(key=lambda r: r["ticker"])
    crossunders.sort(key=lambda r: r["pct_vs_sma"])
    approaching.sort(key=lambda r: r["pct_vs_sma"])

    print(f"[sma_scanner] {len(golden)} golden crosses, {len(crossunders)} crossunders, {len(approaching)} approaching")

    if golden or crossunders or approaching:
        send_email(golden, crossunders, approaching)
    else:
        print("[sma_scanner] No alerts — no email sent")


if __name__ == "__main__":
    main()
