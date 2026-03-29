"""
monitor.py — Position tracker and stop-loss monitor for Kalshi contracts

Usage:
  python3 monitor.py log   --ticker KXBTCD-26MAR2117-T69899.99 --contracts 10 --entry 58
  python3 monitor.py check                  # check all open positions, trigger stops
  python3 monitor.py list                   # show all open positions

Alert logic:
  If current bid price drops 50% or more below your entry price,
  the monitor flags it as a manual review alert.

Config:
  STOP_LOSS_PCT = 0.50  → alert if contract loses 50% of entry value
                           e.g. entered at 58¢, alert if price drops to 29¢
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from kalshi_api import KalshiClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path("positions_kalshi.json")
STOP_LOSS_PCT  = 0.50   # alert if price drops 50% from entry


# ── Positions file helpers ─────────────────────────────────────────────────────
def load_positions() -> list[dict]:
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def save_positions(positions: list[dict]) -> None:
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


# ── Commands ──────────────────────────────────────────────────────────────────
def log_position(ticker: str, contracts: int, entry_cents: int) -> None:
    """Record a manually placed trade."""
    positions = load_positions()
    position  = {
        "ticker"      : ticker,
        "contracts"   : contracts,
        "entry_cents" : entry_cents,
        "stop_cents"  : round(entry_cents * (1 - STOP_LOSS_PCT)),
        "entered_at"  : datetime.now(timezone.utc).isoformat(),
        "status"      : "open",
    }
    positions.append(position)
    save_positions(positions)
    print(f"\nLogged position:")
    print(f"  Ticker    : {ticker}")
    print(f"  Contracts : {contracts}")
    print(f"  Entry     : {entry_cents}¢")
    print(f"  Stop-loss : {position['stop_cents']}¢  "
          f"(triggers if price drops below this)")
    print(f"  Max loss  : ~${contracts * entry_cents / 100:.2f}")


def list_positions() -> None:
    """Show all open positions."""
    positions = [p for p in load_positions() if p["status"] == "open"]
    if not positions:
        print("No open positions.")
        return

    print(f"\n{'Ticker':<40} {'Contracts':>9} {'Entry':>6} {'Stop':>5} {'Entered'}")
    print("-" * 85)
    for p in positions:
        entered = p["entered_at"][:16].replace("T", " ")
        print(f"{p['ticker']:<40} {p['contracts']:>9} "
              f"{p['entry_cents']:>5}¢  {p['stop_cents']:>4}¢  {entered}")


def sync_positions(client: KalshiClient) -> None:
    """Pull current positions from Kalshi API and update local file."""
    raw = client.get_positions()
    if not raw:
        print("No positions found on Kalshi (or running in dry-run mode).")
        return

    existing   = load_positions()
    ex_tickers = {p["ticker"] for p in existing if p["status"] == "open"}
    added      = 0

    for p in raw:
        ticker   = p.get("ticker", "")
        # position > 0 = long YES, position < 0 = long NO
        net_pos  = p.get("position", 0) or round(float(p.get("position_fp", 0)))
        if not ticker or net_pos == 0:
            continue
        side      = "yes" if net_pos > 0 else "no"
        contracts = abs(int(round(net_pos)))
        # Only track crypto contracts
        series = ticker.split("-")[0].upper()
        if not any(x in series for x in ("KXBTCD", "KXETHD")):
            continue
        if ticker in ex_tickers:
            continue                            # already tracked

        # Best guess at entry: use current bid for the held side as proxy
        market = client.get_market(ticker)
        if side == "no":
            # NO bid = 100 - yes_ask
            ya = (market.get("yes_ask_dollars") or market.get("yes_ask_fp")
                  or market.get("yes_ask"))
            if market.get("yes_ask_dollars") is not None:
                entry_cents = max(0, 100 - round(float(market["yes_ask_dollars"]) * 100))
            elif market.get("yes_ask_fp") is not None:
                entry_cents = max(0, 100 - round(market["yes_ask_fp"] / 100))
            elif market.get("yes_ask") is not None:
                entry_cents = max(0, 100 - int(market["yes_ask"]))
            else:
                entry_cents = 50
        else:
            if market.get("yes_bid_dollars") is not None:
                entry_cents = round(float(market["yes_bid_dollars"]) * 100)
            elif market.get("yes_bid_fp") is not None:
                entry_cents = round(market["yes_bid_fp"] / 100)
            elif market.get("yes_bid") is not None:
                entry_cents = int(market["yes_bid"])
            else:
                entry_cents = 50

        close_time = market.get("close_time", "")
        position = {
            "ticker"      : ticker,
            "side"        : side,
            "contracts"   : contracts,
            "entry_cents" : entry_cents,
            "stop_cents"  : round(entry_cents * (1 - STOP_LOSS_PCT)),
            "close_time"  : close_time,
            "entered_at"  : datetime.now(timezone.utc).isoformat(),
            "status"      : "open",
            "synced"      : True,
        }
        existing.append(position)
        added += 1
        print(f"  Added: {ticker}  {side.upper()}  {contracts} contracts  "
              f"entry ~{entry_cents}¢  stop {position['stop_cents']}¢")

    save_positions(existing)
    print(f"\nSync complete — {added} new position(s) added.")
    if added:
        print("  Note: entry prices are estimated from current bid. "
              "Update manually if needed.")


def check_positions(client: KalshiClient, dry_run_sell: bool = True,
                    urgent_only: bool = False) -> None:
    """
    Check current prices against stop-loss levels.
    urgent_only=True: only check contracts expiring within 6 hours.
    """
    positions = load_positions()
    open_pos  = [p for p in positions if p["status"] == "open"]

    if not open_pos:
        print("No open positions to check.")
        return

    print(f"\nChecking {len(open_pos)} open position(s) ...")
    changed = False

    for p in open_pos:
        ticker = p["ticker"]
        # Only monitor crypto contracts
        series = ticker.split("-")[0].upper()
        if not any(x in series for x in ("KXBTCD", "KXETHD")):
            continue

        # If urgent_only, skip contracts with more than 6 hours remaining
        if urgent_only:
            close_time_str = p.get("close_time", "")
            if close_time_str:
                try:
                    close_dt  = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                    hours_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_left > 6:
                        continue
                except Exception:
                    pass

        market  = client.get_market(ticker)
        if not market:
            print(f"  {ticker}: could not fetch market data")
            continue

        # Current best exit price for the held side
        side = p.get("side", "yes")
        if side.lower() == "no":
            # NO bid = 100 - yes_ask
            yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask_fp") or market.get("yes_ask")
            if yes_ask is None:
                print(f"  {ticker}: no ask price available for NO bid")
                continue
            if market.get("yes_ask_dollars") is not None:
                current_cents = max(0, 100 - round(float(market["yes_ask_dollars"]) * 100))
            elif market.get("yes_ask_fp") is not None:
                current_cents = max(0, 100 - round(market["yes_ask_fp"] / 100))
            else:
                current_cents = max(0, 100 - int(market["yes_ask"]))
        else:
            yes_bid_dollars = market.get("yes_bid_dollars")
            yes_bid_fp      = market.get("yes_bid_fp")
            yes_bid         = market.get("yes_bid")
            if yes_bid_dollars is not None:
                current_cents = round(float(yes_bid_dollars) * 100)
            elif yes_bid_fp is not None:
                current_cents = round(yes_bid_fp / 100)
            elif yes_bid is not None:
                current_cents = int(yes_bid)
            else:
                print(f"  {ticker}: no bid price available")
                continue

        entry   = p["entry_cents"]
        stop    = p["stop_cents"]
        pnl_pct = (current_cents - entry) / entry * 100 if entry > 0 else 0.0

        status_str = f"{current_cents}¢  (entry {entry}¢  {pnl_pct:+.1f}%)"

        # Bid of 0 means contract has settled/expired — mark closed, not stopped
        if current_cents == 0:
            print(f"  SETTLED  {ticker:<42} — marking closed")
            p["status"]     = "settled"
            p["closed_at"]  = datetime.now(timezone.utc).isoformat()
            p["exit_cents"] = 0
            changed = True
            continue

        if current_cents <= stop:
            side  = p.get("side", "yes")
            count = p["contracts"]
            print(f"\n  *** STOP-LOSS TRIGGERED: {ticker}")
            print(f"      Current bid : {current_cents}¢  (stop {stop}¢  entry {entry}¢)")
            print(f"      P&L         : {pnl_pct:+.1f}%")
            print(f"      Selling     : {count} {side.upper()} contracts at {current_cents}¢ ...")
            if not dry_run_sell:
                try:
                    result = client.sell_position(ticker, side, count, current_cents)
                    print(f"      Order placed: {result}")
                    p["status"]     = "stopped"
                    p["closed_at"]  = datetime.now(timezone.utc).isoformat()
                    p["exit_cents"] = current_cents
                    changed = True
                except Exception as e:
                    print(f"      ERROR placing sell order: {e}")
            else:
                print(f"      (dry run — pass --execute to place real order)")
        else:
            print(f"  OK  {ticker:<42} {status_str}")

    if changed:
        save_positions(positions)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Kalshi position monitor")
    sub    = parser.add_subparsers(dest="command")

    # log
    log_p = sub.add_parser("log", help="Record a new position")
    log_p.add_argument("--ticker",    required=True)
    log_p.add_argument("--contracts", required=True, type=int)
    log_p.add_argument("--entry",     required=True, type=int,
                       help="Entry price in cents (e.g. 58 for 58¢)")

    # sync
    sub.add_parser("sync", help="Pull positions from Kalshi API")

    # check
    check_p = sub.add_parser("check", help="Check positions against stop-loss levels")
    check_p.add_argument("--urgent", action="store_true",
                         help="Only check contracts expiring within 6 hours")
    check_p.add_argument("--execute", action="store_true",
                         help="Place real sell orders when stop-loss triggers (default: dry run)")

    # list
    sub.add_parser("list", help="List all open positions")

    args   = parser.parse_args()
    client = KalshiClient()

    if args.command == "log":
        log_position(args.ticker, args.contracts, args.entry)
    elif args.command == "sync":
        sync_positions(client)
    elif args.command == "check":
        check_positions(client,
                        dry_run_sell=not getattr(args, "execute", False),
                        urgent_only=getattr(args, "urgent", False))
    elif args.command == "list":
        list_positions()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
