"""
kalshi_crypto_weekly.py — Lightweight weekly-only crypto scan for Kalshi

Stripped-down version of kalshi_crypto.py that:
- Only loads daily model (no intraday)
- Only scores markets with >24h to expiry (weekly bucket)
- Uses full 5 years of history for accuracy
- Significantly less memory than full scan (no hourly/minute data)

To restore full scan (all buckets + intraday), revert to kalshi_crypto.py --scan
"""

import argparse
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from kalshi_api import KalshiClient
from kalshi_crypto import (
    CRYPTO_ASSETS,
    KALSHI_SERIES,
    STRIKE_OFFSETS,
    CALIBRATION_PATH,
    RF_TREES,
    RF_MAX_DEPTH,
    RF_MIN_LEAF,
    RANDOM_STATE,
    MIN_TRAIN_FRAC,
    MIN_EV,
    MIN_EDGE,
    MAX_KELLY,
    DEFAULT_BANKROLL,
    compute_kelly,
    score_contract,
    load_calibration,
    load_crypto_models,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Minimal config (weekly only) ────────────────────────────────────────────
INFERENCE_PERIOD = "5y"  # Full 5 years for model accuracy
# ────────────────────────────────────────────────────────────────────────────


def download_crypto_minimal(symbol: str, period: str = INFERENCE_PERIOD) -> pd.DataFrame:
    """Download daily OHLC (minimal history)."""
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def main():
    import sys
    print("SCAN_START: kalshi_crypto_weekly.py starting", flush=True)
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description="Weekly-only Kalshi crypto scan")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--min-ev", type=float, default=MIN_EV)
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--auto-save-db", action="store_true",
                        help="Save results to Supabase")
    args = parser.parse_args()
    print(f"SCAN_ARGS: parsed args OK", flush=True)
    sys.stdout.flush()

    # ── Kalshi client ──
    print("SCAN_INIT: Initializing Kalshi client...", flush=True)
    sys.stdout.flush()
    client = KalshiClient()
    print("SCAN_INIT: Kalshi client created", flush=True)
    sys.stdout.flush()
    if not client.dry_run:
        print("SCAN_LOGIN: Logging in to Kalshi...", flush=True)
        sys.stdout.flush()
        client.login()
        print("SCAN_LOGIN: Login successful", flush=True)
        sys.stdout.flush()
    try:
        print("SCAN_BALANCE: Fetching balance...", flush=True)
        sys.stdout.flush()
        balance_cents = client.get_balance().get("balance", 0)
        print(f"SCAN_BALANCE: Balance = {balance_cents}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"SCAN_BALANCE_ERROR: {e}", flush=True)
        sys.stdout.flush()
        balance_cents = 0
    bankroll = args.bankroll or balance_cents / 100 or DEFAULT_BANKROLL
    mode = "DRY RUN" if client.dry_run else "LIVE"

    print(f"\n{'='*65}")
    print(f"  Kalshi Weekly Scan  [{mode}]  —  {date.today()}")
    print(f"  Bankroll: ${bankroll:,.2f}")
    print(f"{'='*65}")

    # ── Load models (daily only) ──
    models = load_crypto_models()
    if models.get("daily") is None:
        logger.error("No daily model found. Skipping scan.")
        return

    logger.info("Model loaded — daily only (intraday skipped)")

    # ── Asset data (minimal: daily only) ──
    asset_dfs_by_symbol = {}
    for symbol in CRYPTO_ASSETS:
        logger.info(f"Fetching {symbol} daily data ({INFERENCE_PERIOD})...")
        daily_df = download_crypto_minimal(symbol)
        asset_dfs_by_symbol[symbol] = {"daily": daily_df, "hourly": None, "minute": []}

    # ── Score contracts (weekly only) ──
    recommendations = []
    all_results = []

    for symbol, series in KALSHI_SERIES.items():
        markets = client.get_markets(series_ticker=series, status="open")
        logger.info(f"{series}: {len(markets)} open contracts")
        asset_dfs = asset_dfs_by_symbol[symbol]

        for market in markets:
            # Score both sides
            for result in score_contract(market, models, asset_dfs):
                # Only keep >24hr (weekly) markets
                if result["hours_to_expiry"] > 24:
                    all_results.append(result)
                    if result["ev"] >= args.min_ev and result["edge"] >= args.min_edge:
                        recommendations.append(result)

    # ── Results ──
    print(f"\n[WEEKLY MARKETS ONLY (>24h to expiry)]")
    print(f"  Total scored: {len(all_results)}")
    print(f"  Recommendations (EV ≥ {args.min_ev}, edge ≥ {args.min_edge*100:.0f}pp): {len(recommendations)}")

    if not recommendations:
        print("  No contracts meet the threshold today.")
    else:
        recommendations.sort(key=lambda x: x["ev"], reverse=True)
        print(f"\n{'Ticker':<35} {'Side':>4} {'Price':>6} {'Cal%':>6} {'Edge':>6} {'EV':>6} {'Kelly':>6}")
        print(f"{'-'*90}")
        for r in recommendations:
            print(f"  {r['ticker']:<35} {r['side']:>4}  {r['price']:>4}¢  "
                  f"{r['calibrated_prob']*100:>5.1f}%  {r['edge']*100:>+5.1f}%  "
                  f"{r['ev']:>+.3f}  {r['kelly_pct']:>5.1f}%")

    # ── Auto-save to DB ──
    if args.auto_save_db and recommendations:
        try:
            import db
            added = 0
            # Per-bucket Kelly sizing (dollars-first). This file only handles
            # weekly markets, so BUCKET_BUDGETS["weekly"] is the cap.
            from kalshi_crypto import _size_contracts
            for rec in recommendations:
                try:
                    contracts, bet_dollars = _size_contracts(
                        rec["kelly_pct"], rec["price"], "weekly"
                    )
                    db.add_paper_trade({
                        "ticker": rec["ticker"],
                        "side": rec["side"],
                        "status": "open",
                        "price_cents": rec["price"],
                        "contracts": contracts,
                        "bet_dollars": bet_dollars,
                        "ev": rec["ev"],
                        "kelly_pct": rec["kelly_pct"],
                        "cal_prob": rec["calibrated_prob"],
                        "hours_to_exp": rec["hours_to_expiry"],
                        "close_time": rec["expiry"],
                        "bucket": "weekly",
                        "placed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    })
                    added += 1
                except Exception as e:
                    logger.error(f"Failed to save {rec['ticker']}: {e}")

            logger.info(f"Saved {added} trades to Supabase")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
