#!/usr/bin/env python3
"""
Recover Kalshi closed trades from API and populate Supabase.
Fetches all fills since March 1, 2026 and reconstructs closed positions.
"""

import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

# Ensure kalshi/ (siblings) and repo root (shared/) are importable.
_KALSHI_DIR = Path(__file__).parent
sys.path.insert(0, str(_KALSHI_DIR))
sys.path.insert(0, str(_KALSHI_DIR.parent))

from kalshi_api import KalshiClient
import db

print("="*70)
print("KALSHI TRADE RECOVERY - Fetching fills since March 1, 2026")
print("="*70)

# Initialize Kalshi client
client = KalshiClient()
if client.dry_run:
    print("ERROR: Running in dry-run mode - no Kalshi credentials found!")
    sys.exit(1)

print(f"Connected to Kalshi API (dry_run={client.dry_run})")

# Fetch all fills since March 1
since_ts = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
print(f"\nFetching fills since March 1, 2026 (timestamp: {since_ts} ms)...")

fills = client.get_fills(limit=1000, min_ts=since_ts)
print(f"✓ Fetched {len(fills)} fills")

if not fills:
    print("ERROR: No fills returned from API!")
    sys.exit(1)

# Group fills by ticker
fills_by_ticker = defaultdict(list)
for fill in fills:
    ticker = fill.get("market_ticker") or fill.get("ticker", "")
    if ticker and (ticker.startswith("KXBTC") or ticker.startswith("KXETH")):
        fills_by_ticker[ticker].append(fill)

print(f"✓ Grouped into {len(fills_by_ticker)} tickers")

# Reconstruct positions
def price_dollars(fill, price_key="yes_price"):
    """Convert fixed-point price to dollars."""
    price_fp = fill.get(price_key, 0)
    return price_fp / 10000 if price_fp else 0

def fill_count(fill):
    """Get contract count from fill."""
    return fill.get("quantity", fill.get("contracts", 0))

def fill_action(fill):
    """Get action from fill (buy/sell)."""
    action = fill.get("action", "")
    if action in ("buy", "sell"):
        return action
    side = fill.get("side", "yes")
    return "buy" if side == "yes" else "sell"

positions = []
for ticker in sorted(fills_by_ticker.keys()):
    ticker_fills = sorted(
        fills_by_ticker[ticker],
        key=lambda f: f.get("ts") or f.get("created_time", 0)
    )

    # Process fills to reconstruct positions
    for side in ["yes", "no"]:
        side_fills = [f for f in ticker_fills if f.get("side", "yes") == side]
        if not side_fills:
            continue

        contracts = 0
        buy_cost = 0.0
        sell_proceeds = 0.0
        latest_ts = 0
        entry_cents = None
        exit_cents = None
        result = None

        for fill in side_fills:
            ts = fill.get("ts") or fill.get("created_time", 0)
            if ts > latest_ts:
                latest_ts = ts

            cnt = fill_count(fill)
            action = fill_action(fill)

            if action == "buy":
                if contracts == 0:
                    entry_cents = round(price_dollars(fill, "yes_price") * 100)
                buy_cost += cnt * price_dollars(fill, "yes_price")
                contracts += cnt
            elif action == "sell":
                sell_proceeds += cnt * price_dollars(fill, "yes_price")
                contracts -= cnt

        if contracts == 0 and buy_cost > 0 and sell_proceeds > 0:
            # Fully closed position
            pnl = round(sell_proceeds - buy_cost, 2)
            exit_cents = round(sell_proceeds / max(fill_count(side_fills[0]), 1) * 100) if side_fills else None

            # Check if we can determine result (won/lost)
            # This requires checking if the market resolved
            # For now, infer from PnL
            result = "yes" if pnl > 0 else "no" if pnl < 0 else None

            positions.append({
                "ticker": ticker,
                "side": side,
                "contracts": fill_count(side_fills[0]),
                "entry_cents": entry_cents,
                "exit_cents": exit_cents,
                "buy_cost": round(buy_cost, 2),
                "sell_proceeds": round(sell_proceeds, 2),
                "pnl": pnl,
                "result": result,
                "settled_at": datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc).isoformat() if latest_ts else None,
            })

print(f"✓ Reconstructed {len(positions)} closed positions")

# Store in Supabase
if positions:
    count = db.upsert_kalshi_trades(positions)
    print(f"✓ Stored {count} positions in Supabase kalshi_trades table")
    print("\n" + "="*70)
    print("RECOVERY COMPLETE - Performance tab should now show your trades")
    print("="*70)
else:
    print("WARNING: No closed positions found to recover!")
    sys.exit(1)
