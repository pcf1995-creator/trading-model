"""
kalshi/portfolio_guard.py — Correlation-aware position filter.

BTC and ETH weekly contracts are ~0.80 correlated. Simultaneously holding
YES on BTC-above-X and YES on ETH-above-Y is nearly the same directional
bet and doubles concentration risk.

Two rules applied before saving recommendations to the DB:

  1. Correlation cap
     ──────────────
     Total dollar exposure for BTC+ETH in the same direction (both YES or
     both NO) may not exceed MAX_CORR_MULTIPLIER × per-asset bucket budget.
     For weekly: $200 × 1.5 = $300 max same-direction BTC+ETH exposure.
     The highest-EV trade is always preferred; excess positions are reduced
     to fit remaining capacity rather than outright rejected when possible.

  2. Drawdown brake
     ──────────────
     If the current open portfolio drawdown exceeds BRAKE_THRESHOLD_PCT of
     bankroll (default 10% = $50 on $500), all new Kelly fractions are
     multiplied by BRAKE_FACTOR (0.5). The brake lifts automatically when
     the portfolio recovers above the threshold.

Usage (from kalshi_crypto_weekly.py):
    from portfolio_guard import apply_portfolio_guard
    open_trades = db.load_paper_trades()
    open_trades = [t for t in open_trades if t.get("status") == "open"]
    recommendations = apply_portfolio_guard(
        recommendations, open_trades, bankroll=bankroll, bucket="weekly"
    )
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Risk parameters
MAX_CORR_MULTIPLIER = 1.5   # BTC+ETH same direction ≤ 1.5× per-asset budget
BRAKE_THRESHOLD_PCT = 0.10  # drawdown fraction that triggers the Kelly brake
BRAKE_FACTOR        = 0.50  # Kelly multiplier when brake is active


def _estimate_open_pnl(open_positions: list[dict]) -> float:
    """
    Rough mark-to-market P&L on open positions.

    Uses `current_price_cents` if the caller stamped it in the position dict;
    otherwise uses entry price vs. the midpoint of yes_bid/yes_ask from the
    market snapshot if present.  Falls back to zero when no current price is
    available (conservative: doesn't assume unrealised gains).
    """
    total = 0.0
    for pos in open_positions:
        if pos.get("status") != "open":
            continue
        contracts   = int(pos.get("contracts", 1))
        entry_cents = float(pos.get("price_cents", 50))

        current_cents = pos.get("current_price_cents")
        if current_cents is None:
            continue
        pnl = (float(current_cents) - entry_cents) * contracts / 100.0
        total += pnl
    return total


def _asset_from_ticker(ticker: str) -> str:
    """Extract 'BTC' or 'ETH' from a Kalshi ticker like KXBTCD-..."""
    t = ticker.upper()
    if "BTC" in t:
        return "BTC"
    if "ETH" in t:
        return "ETH"
    return "UNK"


def apply_portfolio_guard(
    recommendations: list[dict],
    open_positions: list[dict],
    bankroll: float = 500.0,
    bucket: str = "weekly",
    bucket_budget: float | None = None,
) -> list[dict]:
    """
    Filter and resize recommendations before DB insertion.

    Parameters
    ----------
    recommendations : list of score_contract() result dicts
    open_positions  : list of open paper_trade rows from db.load_paper_trades()
    bankroll        : total bankroll in dollars (for drawdown brake)
    bucket          : time-horizon bucket ("weekly", "intraday_short", etc.)
    bucket_budget   : per-asset dollar cap; defaults to BUCKET_BUDGETS[bucket]

    Returns
    -------
    Filtered list (some removed, some with reduced kelly_pct).
    Always sorted by EV descending so highest-conviction trades are preferred.
    """
    from kalshi_crypto import BUCKET_BUDGETS, _size_contracts

    budget = bucket_budget if bucket_budget is not None else BUCKET_BUDGETS.get(bucket, 200.0)
    max_combined = MAX_CORR_MULTIPLIER * budget  # e.g. $300 for weekly

    # ── Drawdown brake ──────────────────────────────────────────────────────
    open_pnl = _estimate_open_pnl(open_positions)
    drawdown_frac = max(0.0, -open_pnl / bankroll) if bankroll > 0 else 0.0
    brake_on = drawdown_frac > BRAKE_THRESHOLD_PCT
    if brake_on:
        logger.info(
            f"[portfolio_guard] Drawdown brake ACTIVE — "
            f"open P&L ${open_pnl:+.2f} ({drawdown_frac:.1%} of bankroll); "
            f"Kelly × {BRAKE_FACTOR}"
        )

    # ── Build deployed-capital map from open positions ──────────────────────
    # {("BTC", "YES"): dollars_deployed, ("ETH", "NO"): dollars_deployed, …}
    deployed: dict[tuple[str, str], float] = {}
    for pos in open_positions:
        if pos.get("status") != "open" or pos.get("bucket") != bucket:
            continue
        asset  = _asset_from_ticker(pos.get("ticker", ""))
        side   = (pos.get("side", "YES") or "YES").upper()
        dollars = float(pos.get("bet_dollars", 0))
        key = (asset, side)
        deployed[key] = deployed.get(key, 0.0) + dollars

    # ── Filter / resize ─────────────────────────────────────────────────────
    recs_sorted = sorted(recommendations, key=lambda r: r.get("ev", 0), reverse=True)
    filtered: list[dict] = []

    for rec in recs_sorted:
        ticker = rec.get("ticker", "")
        asset  = _asset_from_ticker(ticker) or rec.get("asset", "UNK")
        side   = (rec.get("side", "YES") or "YES").upper()

        kelly = float(rec.get("kelly_pct", 0.0))

        # Apply drawdown brake
        if brake_on:
            kelly = kelly * BRAKE_FACTOR

        # Compute tentative bet size
        contracts, bet_dollars = _size_contracts(kelly, rec["price"], bucket)

        # Correlation cap: check combined BTC+ETH same-direction exposure
        same_dir_deployed = (
            deployed.get(("BTC", side), 0.0)
            + deployed.get(("ETH", side), 0.0)
        )
        if same_dir_deployed + bet_dollars > max_combined:
            remaining = max_combined - same_dir_deployed
            price_dollars = rec["price"] / 100.0
            if remaining >= price_dollars:
                contracts   = max(1, int(remaining / price_dollars))
                bet_dollars = round(contracts * price_dollars, 2)
                kelly       = bet_dollars / budget * 100.0
                logger.info(
                    f"[portfolio_guard] Correlation cap: {ticker} {side} → "
                    f"{contracts} contracts (${bet_dollars:.2f})"
                )
            else:
                logger.info(
                    f"[portfolio_guard] Correlation cap: skipping {ticker} {side} "
                    f"(${same_dir_deployed:.0f}+${bet_dollars:.0f} > ${max_combined:.0f} cap)"
                )
                continue

        # Accept this recommendation with adjusted sizing
        deployed[(asset, side)] = deployed.get((asset, side), 0.0) + bet_dollars
        filtered.append({**rec, "kelly_pct": round(kelly, 1)})

    if brake_on or len(filtered) < len(recs_sorted):
        logger.info(
            f"[portfolio_guard] {len(recs_sorted)} recs → {len(filtered)} passed "
            f"(brake={'ON' if brake_on else 'OFF'})"
        )

    return filtered
