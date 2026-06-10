"""
stocks/overnight_cron.py — Overnight drift paper-trade automation.

Invoked by two GitHub Actions cron jobs:

  --action buy   (4:30 PM ET Mon-Fri, after market close)
      Scans the ticker universe, selects qualifying names by overnight-drift
      criteria, fetches today's official close price, and opens a paper position.

  --action sell  (10:00 AM ET Mon-Fri, after market open)
      Closes every open overnight paper position at today's official open price
      and records the P&L.

Both actions write to the Supabase `overnight_paper_trades` table via db.py.
The Streamlit app reads from that same table to display the trade history.
"""

import argparse
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).parent.parent
_STOCKS_DIR = Path(__file__).parent
for _p in (_REPO_ROOT, _STOCKS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import db as _db
import overnight as _on_mod

# ── Selection thresholds ───────────────────────────────────────────────────────
MIN_T_STAT    = 2.0
MIN_NET_SHARPE = 0.30
MIN_WIN_RATE  = 52.0
COST_BPS      = 3.0   # per side (6 bps round-trip)

# ── Position sizing ────────────────────────────────────────────────────────────
DOLLARS_PER_TRADE = 1_000.0
MAX_POSITIONS     = 5


def _get_price(ticker: str, field: str) -> float | None:
    """
    Fetch today's `field` price ('Close' or 'Open') from yfinance.
    Returns None if data is unavailable.
    """
    try:
        hist = yf.Ticker(ticker).history(period="2d", auto_adjust=True)
        if hist.empty:
            return None
        today = date.today()
        todays = hist[hist.index.date == today]
        if not todays.empty:
            return float(todays[field].iloc[-1])
        if field == "Close":
            return float(hist["Close"].iloc[-1])
        return None
    except Exception:
        return None


def run_buy() -> None:
    """Open paper positions at today's close price for qualifying tickers."""
    today_str = str(date.today())
    print(f"[overnight_cron] BUY — {today_str}")

    _summary_csv = _STOCKS_DIR / "ticker_summary.csv"
    tickers = (
        pd.read_csv(_summary_csv)["Ticker"].tolist()
        if _summary_csv.exists()
        else _on_mod.DEFAULT_TICKERS
    )
    print(f"[overnight_cron] Scanning {len(tickers)} tickers…")

    scan_df = _on_mod.scan_tickers(tickers, period_days=1825, cost_bps=COST_BPS)
    if scan_df.empty:
        print("[overnight_cron] No scan results — aborting.")
        return

    qualifying = scan_df[
        (scan_df["t_stat"]      >= MIN_T_STAT)
        & (scan_df["net_sharpe"] >= MIN_NET_SHARPE)
        & (scan_df["win_rate"]   >= MIN_WIN_RATE)
        & (scan_df["breakeven_bps"] >= COST_BPS)
    ].head(MAX_POSITIONS)

    if qualifying.empty:
        print("[overnight_cron] No qualifying tickers after filtering.")
        return

    print(f"[overnight_cron] {len(qualifying)} qualifying: {qualifying['ticker'].tolist()}")

    open_trades      = _db.load_overnight_paper_trades()
    already_open     = {t["ticker"] for t in open_trades if t.get("status") == "open"}

    new_count = 0
    for _, row in qualifying.iterrows():
        ticker = row["ticker"]
        if ticker in already_open:
            print(f"[overnight_cron] {ticker}: position already open, skipping.")
            continue

        close_px = _get_price(ticker, "Close")
        if not close_px or close_px <= 0:
            print(f"[overnight_cron] {ticker}: no close price, skipping.")
            continue

        shares = round(DOLLARS_PER_TRADE / close_px, 4)
        trade  = {
            "id"          : str(uuid.uuid4()),
            "ticker"      : ticker,
            "entry_date"  : today_str,
            "entry_price" : round(close_px, 4),
            "shares"      : shares,
            "dollars"     : round(close_px * shares, 2),
            "status"      : "open",
            "t_stat"      : float(row.get("t_stat")      or 0),
            "net_sharpe"  : float(row.get("net_sharpe")  or 0),
            "win_rate"    : float(row.get("win_rate")    or 0),
            "placed_at"   : datetime.now(timezone.utc).isoformat(),
        }
        _db.add_overnight_paper_trade(trade)
        new_count += 1
        print(
            f"[overnight_cron] {ticker}: BUY {shares:.4f} sh @ ${close_px:.4f} "
            f"= ${trade['dollars']:.2f}"
        )

    print(f"[overnight_cron] Done. {new_count} positions opened.")


def run_sell() -> None:
    """Close all open overnight paper positions at today's open price."""
    today_str = str(date.today())
    print(f"[overnight_cron] SELL — {today_str}")

    open_trades = [t for t in _db.load_overnight_paper_trades() if t.get("status") == "open"]
    if not open_trades:
        print("[overnight_cron] No open positions.")
        return

    print(f"[overnight_cron] Closing {len(open_trades)} positions…")
    closed = 0
    for trade in open_trades:
        ticker   = trade["ticker"]
        open_px  = _get_price(ticker, "Open")
        if not open_px or open_px <= 0:
            print(f"[overnight_cron] {ticker}: no open price, skipping.")
            continue

        entry_px = float(trade["entry_price"])
        shares   = float(trade.get("shares") or 0)
        pnl_pct  = (open_px - entry_px) / entry_px if entry_px > 0 else 0.0
        pnl_usd  = round((open_px - entry_px) * shares, 2)

        _db.close_overnight_paper_trade(
            trade_id  = trade["id"],
            exit_price = round(open_px, 4),
            exit_date  = today_str,
            pnl_pct    = round(pnl_pct * 100, 4),
            pnl_dollars = pnl_usd,
        )
        closed += 1
        print(
            f"[overnight_cron] {ticker}: SELL @ ${open_px:.4f} | "
            f"entry ${entry_px:.4f} | P&L {pnl_pct*100:+.2f}% (${pnl_usd:+.2f})"
        )

    print(f"[overnight_cron] Done. {closed} positions closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight drift paper trade cron")
    parser.add_argument("--action", choices=["buy", "sell"], required=True,
                        help="'buy' runs after close; 'sell' runs after open")
    args = parser.parse_args()

    if args.action == "buy":
        run_buy()
    else:
        run_sell()
