"""
analyze.py — Trade history analysis for Kalshi crypto positions

Reconstructs P&L from fills since the positions API only returns current positions.
For settled contracts, fetches market result to calculate settlement payout.

Usage:
  python3 analyze.py
  python3 analyze.py --raw
  python3 analyze.py --backtest   # compare model predictions vs actual outcomes
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kalshi_api import KalshiClient

PREDICTIONS_LOG = Path("predictions_log.jsonl")

CRYPTO_SERIES = ("KXBTCD", "KXETHD")


def fetch_all_fills(client: KalshiClient) -> list[dict]:
    fills, cursor = [], None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        data   = client._request("GET", "/portfolio/fills", params=params)
        batch  = data.get("fills", [])
        fills.extend(batch)
        cursor = data.get("cursor") or ""
        if not cursor or not batch:
            break
    return fills


def get_market_result(client: KalshiClient, ticker: str) -> str | None:
    """Returns 'yes', 'no', or None if still open."""
    try:
        market = client._request("GET", f"/markets/{ticker}").get("market", {})
        return market.get("result")  # 'yes', 'no', or None
    except Exception:
        return None


def analyze(fills: list[dict], client: KalshiClient) -> None:
    crypto = [f for f in fills
              if any(s in f.get("ticker", "").upper() for s in CRYPTO_SERIES)]

    if not crypto:
        print("No crypto fills found.")
        return

    # Aggregate per ticker
    by_ticker = defaultdict(lambda: {
        "bought": 0.0, "sold": 0.0, "cost": 0.0,
        "manual_revenue": 0.0, "fees": 0.0,
        "asset": "", "date": ""
    })

    for f in crypto:
        ticker = f["ticker"]
        t      = by_ticker[ticker]
        count  = float(f.get("count_fp", 0))
        side   = f.get("side", "yes")
        action = f.get("action", "buy")
        fee    = float(f.get("fee_cost", 0) or 0)
        price  = float(f.get("yes_price_dollars", 0) if side == "yes"
                       else f.get("no_price_dollars", 0))

        t["asset"] = t["asset"] or ("BTC" if "BTC" in ticker.upper() else "ETH")
        t["date"]  = t["date"]  or f.get("created_time", "")[:10]
        t["fees"] += fee

        if action == "buy":
            t["bought"] += count
            t["cost"]   += count * price
        else:
            t["sold"]           += count
            t["manual_revenue"] += count * price

    # Fetch settlement results for each ticker
    print("Fetching market results for settled contracts...")
    trades = []
    total_api_calls = 0

    for ticker, t in by_ticker.items():
        held_to_settlement = t["bought"] - t["sold"]
        settlement_payout  = 0.0
        status = "open"

        if held_to_settlement > 0:
            result = get_market_result(client, ticker)
            total_api_calls += 1
            if result == "yes":
                settlement_payout = held_to_settlement * 1.0  # $1 per contract
                status = "won"
            elif result == "no":
                settlement_payout = 0.0
                status = "lost"
            else:
                status = "open"
        else:
            status = "exited"  # fully sold before settlement

        total_revenue = t["manual_revenue"] + settlement_payout
        pnl           = total_revenue - t["cost"] - t["fees"]

        trades.append({
            "ticker"   : ticker,
            "asset"    : t["asset"],
            "date"     : t["date"],
            "bought"   : int(t["bought"]),
            "sold"     : int(t["sold"]),
            "cost"     : round(t["cost"], 2),
            "revenue"  : round(total_revenue, 2),
            "fees"     : round(t["fees"], 2),
            "pnl"      : round(pnl, 2),
            "status"   : status,
        })

    trades.sort(key=lambda x: x["pnl"], reverse=True)

    closed  = [t for t in trades if t["status"] != "open"]
    open_t  = [t for t in trades if t["status"] == "open"]
    wins    = [t for t in closed if t["pnl"] > 0]
    losses  = [t for t in closed if t["pnl"] <= 0]

    total_cost    = sum(t["cost"]    for t in trades)
    total_revenue = sum(t["revenue"] for t in trades)
    total_fees    = sum(t["fees"]    for t in trades)
    total_pnl     = sum(t["pnl"]     for t in trades)
    open_exposure = sum(t["cost"]    for t in open_t)
    invested_closed = total_cost - open_exposure

    print(f"\n{'='*60}")
    print(f"  Kalshi Crypto Trade Analysis  ({len(trades)} contracts)")
    print(f"{'='*60}")
    print(f"\n  Win / Loss       : {len(wins)}W / {len(losses)}L")
    if len(wins) + len(losses) > 0:
        print(f"  Win rate         : {len(wins)/(len(wins)+len(losses))*100:.0f}%")
    print(f"  Open positions   : {len(open_t)}")
    print(f"  Total invested   : ${total_cost:.2f}")
    print(f"  Open exposure    : ${open_exposure:.2f}")
    print(f"  Fees paid        : ${total_fees:.2f}")
    print(f"  Net P&L          : ${total_pnl:+.2f}")
    if invested_closed > 0:
        print(f"  ROI (closed)     : {total_pnl/invested_closed*100:+.1f}%")

    # By asset
    print(f"\n  {'─'*20} By Asset {'─'*20}")
    for asset in ("BTC", "ETH"):
        t_list   = [t for t in trades if t["asset"] == asset]
        cl_list  = [t for t in t_list if t["status"] != "open"]
        if not t_list:
            continue
        a_cost   = sum(t["cost"] for t in t_list)
        a_pnl    = sum(t["pnl"]  for t in t_list)
        a_fees   = sum(t["fees"] for t in t_list)
        a_wins   = sum(1 for t in cl_list if t["pnl"] > 0)
        a_losses = sum(1 for t in cl_list if t["pnl"] <= 0)
        a_exp    = sum(t["cost"] for t in t_list if t["status"] == "open")
        roi      = a_pnl / (a_cost - a_exp) * 100 if (a_cost - a_exp) > 0 else 0
        print(f"  {asset}: {a_wins}W/{a_losses}L  |  "
              f"Invested ${a_cost:.2f}  |  Fees ${a_fees:.2f}  |  "
              f"P&L ${a_pnl:+.2f}  |  ROI {roi:+.1f}%")

    # All contracts
    print(f"\n  {'─'*15} All Contracts (best → worst) {'─'*15}")
    print(f"  {'Ticker':<42} {'Date':>10} {'Ctrs':>4} {'Cost':>7} {'P&L':>7}  {'Status'}")
    print(f"  {'-'*82}")
    for t in trades:
        roi_str = f"{t['pnl']/t['cost']*100:+.0f}%" if t["cost"] > 0 else ""
        print(f"  {t['ticker']:<42} {t['date']:>10} "
              f"{t['bought']:>4} ${t['cost']:>6.2f}  ${t['pnl']:>+6.2f}  "
              f"{t['status'].upper()} {roi_str}")

    print()


def load_predictions() -> list[dict]:
    if not PREDICTIONS_LOG.exists():
        return []
    entries = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def backtest(fills: list[dict], client) -> None:
    """Join model predictions against actual trade outcomes."""
    predictions = load_predictions()
    if not predictions:
        print("No predictions log found. Run kalshi_crypto.py first to start logging.")
        return

    # Build actual outcome map from fills: ticker → {outcome, avg_entry_cents}
    CRYPTO_SERIES = ("KXBTCD", "KXETHD")
    crypto = [f for f in fills if any(s in f.get("ticker", "").upper() for s in CRYPTO_SERIES)]

    by_ticker = defaultdict(lambda: {"bought": 0.0, "sold": 0.0, "cost": 0.0, "revenue": 0.0})
    for f in crypto:
        ticker = f["ticker"]
        t      = by_ticker[ticker]
        count  = float(f.get("count_fp", 0))
        side   = f.get("side", "yes")
        action = f.get("action", "buy")
        price  = float(f.get("yes_price_dollars", 0) if side == "yes" else f.get("no_price_dollars", 0))
        if action == "buy":
            t["bought"] += count
            t["cost"]   += count * price
        else:
            t["sold"]    += count
            t["revenue"] += count * price

    # Fetch settlement results for tickers we held to expiry
    outcome_map = {}   # ticker → "won" | "lost" | "exited" | "open"
    for ticker, t in by_ticker.items():
        held = t["bought"] - t["sold"]
        if held <= 0:
            outcome_map[ticker] = "exited"
        else:
            result = None
            try:
                market = client._request("GET", f"/markets/{ticker}").get("market", {})
                result = market.get("result")
            except Exception:
                pass
            if result == "yes":
                outcome_map[ticker] = "won"
            elif result == "no":
                outcome_map[ticker] = "lost"
            else:
                outcome_map[ticker] = "open"

    # Deduplicate predictions: keep the one closest to expiry for each ticker
    # (i.e. most recent run before expiry)
    best_pred = {}
    for p in predictions:
        ticker = p["ticker"]
        if ticker not in best_pred or p["run_ts"] > best_pred[ticker]["run_ts"]:
            best_pred[ticker] = p

    # Join
    joined = []
    for ticker, pred in best_pred.items():
        if ticker not in outcome_map:
            continue
        outcome = outcome_map[ticker]
        if outcome == "open":
            continue  # skip, not yet resolved
        actual = 1 if outcome == "won" else 0
        joined.append({
            "ticker"          : ticker,
            "outcome"         : outcome,
            "actual"          : actual,
            "cal_prob"        : pred["calibrated_prob"],
            "market_price"    : pred["market_price"],
            "ev"              : pred["ev"],
            "edge"            : pred["edge"],
            "recommended"     : pred["recommended"],
            "hours_left"      : pred["hours_left"],
        })

    if not joined:
        print("No resolved trades found with matching predictions yet.")
        print(f"  Predictions logged: {len(best_pred)} tickers")
        print(f"  Resolved trades:    {sum(1 for v in outcome_map.values() if v != 'open')}")
        return

    wins       = [j for j in joined if j["actual"] == 1]
    losses     = [j for j in joined if j["actual"] == 0]
    rec        = [j for j in joined if j["recommended"]]
    rec_wins   = [j for j in rec if j["actual"] == 1]

    avg_cal    = sum(j["cal_prob"] for j in joined) / len(joined)
    actual_wr  = len(wins) / len(joined)

    print(f"\n{'='*60}")
    print(f"  Model Back-test  ({len(joined)} resolved contracts with predictions)")
    print(f"{'='*60}")
    print(f"\n  Avg predicted cal%  : {avg_cal*100:.1f}%")
    print(f"  Actual win rate     : {actual_wr*100:.1f}%")
    print(f"  Calibration gap     : {(avg_cal - actual_wr)*100:+.1f}pp "
          f"({'overconfident' if avg_cal > actual_wr else 'underconfident'})")
    if rec:
        print(f"\n  Recommended trades  : {len(rec)}  →  {len(rec_wins)}W / {len(rec)-len(rec_wins)}L  "
              f"({len(rec_wins)/len(rec)*100:.0f}% win rate)")

    print(f"\n  {'Ticker':<42} {'Cal%':>5} {'Mkt':>4} {'EV':>6} {'Outcome'}")
    print(f"  {'-'*72}")
    for j in sorted(joined, key=lambda x: -x["cal_prob"]):
        flag = "★" if j["recommended"] else " "
        print(f"  {flag} {j['ticker']:<41} "
              f"{j['cal_prob']*100:>4.0f}% "
              f"{j['market_price']:>3}¢ "
              f"{j['ev']:>+.2f}  "
              f"{j['outcome'].upper()}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",      action="store_true")
    parser.add_argument("--backtest", action="store_true",
                        help="Compare model predictions vs actual outcomes")
    args = parser.parse_args()

    client = KalshiClient()
    if client.dry_run:
        print("No credentials — cannot fetch trade history.")
        return

    print("Fetching fills...")
    fills = fetch_all_fills(client)
    print(f"Found {len(fills)} fills.\n")

    if args.raw:
        crypto = [f for f in fills
                  if any(s in f.get("ticker", "").upper() for s in CRYPTO_SERIES)]
        print(json.dumps(crypto[:5], indent=2))
        return

    if args.backtest:
        backtest(fills, client)
        return

    analyze(fills, client)


if __name__ == "__main__":
    main()
