"""
stocks/refresh_sma_universe.py — Quarterly SMA scanner universe refresh.

Fetches the current S&P 500 constituent list from Wikipedia, checks
market cap for each via yfinance, and upserts tickers with market cap
>= $10B into the Supabase sma_scanner_universe table.

Run manually ~4x per year (or whenever the S&P 500 composition changes):

    SUPABASE_URL=... SUPABASE_KEY=... python3 stocks/refresh_sma_universe.py

Requires sma_scanner_universe table — create it once with:

    CREATE TABLE IF NOT EXISTS sma_scanner_universe (
        ticker       TEXT PRIMARY KEY,
        name         TEXT,
        sector       TEXT,
        market_cap_b NUMERIC,
        updated_at   TIMESTAMPTZ DEFAULT NOW()
    );
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).parent.parent
_STOCKS_DIR = Path(__file__).parent
for _p in (_REPO_ROOT, _STOCKS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.db_common import _get_client

MIN_MARKET_CAP_B = 10.0


def get_sp500_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector"]]


def get_market_cap_b(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).info
        mc = info.get("marketCap")
        return round(mc / 1e9, 2) if mc else None
    except Exception:
        return None


def main() -> None:
    client = _get_client()
    if not client:
        raise RuntimeError("Supabase not available — set SUPABASE_URL and SUPABASE_KEY")

    print("Fetching S&P 500 constituent list from Wikipedia…")
    sp500 = get_sp500_constituents()
    print(f"  {len(sp500)} tickers found\n")

    qualified: list[dict] = []
    skipped: list[str]   = []

    for i, row in sp500.iterrows():
        ticker = row["ticker"]
        mc = get_market_cap_b(ticker)
        if mc is None:
            print(f"  ? {ticker}: no market cap data, skipping")
            skipped.append(ticker)
        elif mc >= MIN_MARKET_CAP_B:
            qualified.append({
                "ticker":       ticker,
                "name":         row["name"],
                "sector":       row["sector"],
                "market_cap_b": mc,
                "updated_at":   datetime.now(timezone.utc).isoformat(),
            })
            print(f"  ✓ {ticker}: ${mc:.1f}B")
        else:
            print(f"  ✗ {ticker}: ${mc:.1f}B (below threshold)")
        # Rate-limit yfinance to avoid getting blocked
        if (i + 1) % 25 == 0:
            time.sleep(2)

    print(f"\n{len(qualified)} tickers qualify (market cap >= ${MIN_MARKET_CAP_B}B)")
    if skipped:
        print(f"{len(skipped)} tickers skipped (no data): {', '.join(skipped)}")

    # Upsert qualified tickers
    for row in qualified:
        try:
            client.table("sma_scanner_universe").upsert(row, on_conflict="ticker").execute()
        except Exception as e:
            print(f"  Upsert failed for {row['ticker']}: {e}")

    # Remove tickers that no longer qualify
    qualified_set = {r["ticker"] for r in qualified}
    try:
        existing = client.table("sma_scanner_universe").select("ticker").execute()
        removed = []
        for r in (existing.data or []):
            if r["ticker"] not in qualified_set:
                client.table("sma_scanner_universe").delete().eq("ticker", r["ticker"]).execute()
                removed.append(r["ticker"])
        if removed:
            print(f"\nRemoved {len(removed)} tickers no longer qualifying: {', '.join(removed)}")
    except Exception as e:
        print(f"  Cleanup failed: {e}")

    print("\nUniverse refresh complete.")


if __name__ == "__main__":
    main()
