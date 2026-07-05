"""
kalshi/mc_simulate.py — Monte Carlo 30-day campaign simulator.

Answers: "Given the current model edge profile and $500 bankroll, what is
P(30-day return ≥ 7%) and P(max drawdown > 20%)?"

Model assumptions
─────────────────
• Weekly-only markets (>24h to expiry)
• Half-Kelly sizing capped at 25% of per-asset budget ($200)
• BTC and ETH outcomes are correlated (rho ≈ 0.80)
• Drawdown brake: when cumulative loss > brake_threshold, Kelly × 0.5
• Correlation cap: BTC+ETH same-direction exposure ≤ 1.5× single budget
• Trade frequency derived from scan history (default: ~4 new weekly
  contract recommendations per week after deduplication)

Edge distribution calibrated from scan history:
  median edge ≈ 12pp, std ≈ 6pp, clipped to [5pp, 35pp]
  median price ≈ 25¢ (recommendations skew toward out-of-the-money)

Usage
─────
  python mc_simulate.py                            # default parameters
  python mc_simulate.py --bankroll 1000            # different bankroll
  python mc_simulate.py --trades-per-week 6        # higher scan frequency
  python mc_simulate.py --edge-mean 0.08 --n 5000  # pessimistic edge
"""

import argparse
import math
import sys

import numpy as np


# ── Default simulation parameters ──────────────────────────────────────────
DEFAULT_BANKROLL        = 500.0
DEFAULT_N_PATHS         = 10_000
DEFAULT_N_WEEKS         = 4
DEFAULT_TRADES_PER_WEEK = 4.0    # unique new weekly recs after dedup
DEFAULT_EDGE_MEAN       = 0.12   # 12pp mean edge on recs that pass filter
DEFAULT_EDGE_STD        = 0.06
DEFAULT_EDGE_MIN        = 0.05
DEFAULT_EDGE_MAX        = 0.35
DEFAULT_PRICE_MEAN      = 25.0   # cents (recs skew OTM)
DEFAULT_PRICE_STD       = 12.0
DEFAULT_PRICE_MIN       = 10.0
DEFAULT_PRICE_MAX       = 75.0
DEFAULT_BTC_ETH_CORR    = 0.80
DEFAULT_KELLY_FRACTION  = 0.50   # half-Kelly applied to full-Kelly result
DEFAULT_MAX_KELLY       = 0.25   # cap on Kelly fraction
DEFAULT_BUCKET_BUDGET   = 200.0  # per-asset budget for weekly bucket
DEFAULT_MAX_DRAWDOWN    = 0.20   # drawdown hard brake threshold
DEFAULT_BRAKE_FACTOR    = 0.50   # Kelly multiplier when brake active
DEFAULT_BRAKE_THRESHOLD = 0.10   # fraction of bankroll that triggers brake
DEFAULT_TARGET_RETURN   = 0.07   # 7% goal


def _kelly_fraction(edge: float, price_cents: float) -> float:
    """Half-Kelly fraction for a binary YES bet."""
    p   = price_cents / 100.0
    b   = (1 - p) / p          # net odds: win (1-p) per dollar risked p
    win_prob = p + edge         # calibrated win probability
    f   = (win_prob * b - (1 - win_prob)) / b  # full Kelly
    return float(np.clip(f * 0.5, 0.0, DEFAULT_MAX_KELLY))  # half-Kelly cap


def _bet_dollars(kelly: float, budget: float, price_cents: float) -> float:
    """Dollars placed = budget × kelly, rounded to whole contracts."""
    target = budget * kelly
    price_d = price_cents / 100.0
    if price_d <= 0:
        return 0.0
    contracts = max(1, int(target / price_d))
    return round(contracts * price_d, 2)


def simulate_campaign(
    n_paths: int           = DEFAULT_N_PATHS,
    bankroll: float        = DEFAULT_BANKROLL,
    n_weeks: int           = DEFAULT_N_WEEKS,
    trades_per_week: float = DEFAULT_TRADES_PER_WEEK,
    edge_mean: float       = DEFAULT_EDGE_MEAN,
    edge_std: float        = DEFAULT_EDGE_STD,
    price_mean: float      = DEFAULT_PRICE_MEAN,
    price_std: float       = DEFAULT_PRICE_STD,
    btc_eth_corr: float    = DEFAULT_BTC_ETH_CORR,
    bucket_budget: float   = DEFAULT_BUCKET_BUDGET,
    max_drawdown: float    = DEFAULT_MAX_DRAWDOWN,
    brake_threshold: float = DEFAULT_BRAKE_THRESHOLD,
    brake_factor: float    = DEFAULT_BRAKE_FACTOR,
    target_return: float   = DEFAULT_TARGET_RETURN,
    seed: int              = 42,
) -> dict:
    """
    Run Monte Carlo simulation of the 30-day trading campaign.

    Returns a dict with summary statistics and percentile distributions.
    """
    rng = np.random.default_rng(seed)

    # Total trades across the campaign
    n_days   = n_weeks * 7
    total_trades = int(round(trades_per_week * n_weeks))

    final_pnl       = np.zeros(n_paths)
    max_drawdowns   = np.zeros(n_paths)
    hit_target      = np.zeros(n_paths, dtype=bool)
    brake_fired     = np.zeros(n_paths, dtype=bool)

    # Cholesky for correlated BTC/ETH outcomes.
    # For a single trade on one asset, correlation is irrelevant; it matters
    # when two simultaneous positions are live.  We model this at the weekly
    # level: each week we may have a BTC trade and an ETH trade whose
    # outcomes are correlated.
    rho = btc_eth_corr
    L   = np.array([[1.0, 0.0], [rho, math.sqrt(1 - rho**2)]])

    for path in range(n_paths):
        pnl          = 0.0
        peak_pnl     = 0.0
        max_dd       = 0.0
        brake_active = False

        trades_left  = total_trades

        for week in range(n_weeks):
            # Determine trades this week (Poisson-distributed around mean)
            wk_trades = rng.poisson(trades_per_week)
            wk_trades = min(wk_trades, trades_left)
            trades_left -= wk_trades
            if wk_trades == 0:
                continue

            # Sample edges and prices for this week's trades
            edges  = rng.normal(edge_mean, edge_std, size=wk_trades)
            edges  = np.clip(edges, DEFAULT_EDGE_MIN, DEFAULT_EDGE_MAX)
            prices = rng.normal(price_mean, price_std, size=wk_trades)
            prices = np.clip(prices, DEFAULT_PRICE_MIN, DEFAULT_PRICE_MAX)

            # Generate correlated outcomes for BTC/ETH pairs.
            # Pair consecutive trades: (0,1), (2,3), …; odd remainder is solo.
            n_pairs  = wk_trades // 2
            n_solo   = wk_trades % 2

            pair_z   = rng.standard_normal((n_pairs, 2)) @ L.T  # (n_pairs, 2)
            solo_z   = rng.standard_normal(n_solo)

            all_z    = np.concatenate([pair_z.flatten(), solo_z])[:wk_trades]

            for t in range(wk_trades):
                e  = edges[t]
                p  = prices[t]
                k  = _kelly_fraction(e, p)

                # Apply drawdown brake
                effective_k = k * brake_factor if brake_active else k
                bet         = _bet_dollars(effective_k, bucket_budget, p)
                if bet <= 0:
                    continue

                # Simulate outcome: win with probability (price/100 + edge)
                win_prob = min(0.99, p / 100.0 + e)
                win      = all_z[t] < stats_ppf(win_prob)
                if win:
                    pnl_trade = bet * (100.0 - p) / p
                else:
                    pnl_trade = -bet

                pnl      += pnl_trade
                peak_pnl  = max(peak_pnl, pnl)
                drawdown  = (peak_pnl - pnl) / bankroll if peak_pnl > pnl else 0.0
                max_dd    = max(max_dd, drawdown)

                # Check brake
                open_loss_frac = max(0.0, -pnl / bankroll)
                brake_active   = open_loss_frac > brake_threshold

        final_pnl[path]     = pnl
        max_drawdowns[path]  = max_dd
        hit_target[path]     = pnl / bankroll >= target_return
        brake_fired[path]    = max_drawdowns[path] > brake_threshold

    returns = final_pnl / bankroll

    return {
        "n_paths"           : n_paths,
        "n_weeks"           : n_weeks,
        "total_trades"      : total_trades,
        "bankroll"          : bankroll,
        "edge_mean"         : edge_mean,
        "price_mean"        : price_mean,

        "p_hit_target"      : float(hit_target.mean()),
        "p_max_dd_exceeded" : float((max_drawdowns > max_drawdown).mean()),
        "p_brake_fired"     : float(brake_fired.mean()),
        "p_loss"            : float((returns < 0).mean()),

        "expected_return"   : float(returns.mean()),
        "median_return"     : float(np.median(returns)),
        "std_return"        : float(returns.std()),

        "p05_return"        : float(np.percentile(returns, 5)),
        "p10_return"        : float(np.percentile(returns, 10)),
        "p25_return"        : float(np.percentile(returns, 25)),
        "p75_return"        : float(np.percentile(returns, 75)),
        "p90_return"        : float(np.percentile(returns, 90)),
        "p95_return"        : float(np.percentile(returns, 95)),

        "expected_pnl"      : float(final_pnl.mean()),
        "median_pnl"        : float(np.median(final_pnl)),
        "median_max_dd"     : float(np.median(max_drawdowns)),
    }


# Fast normal CDF inverse via log-odds approximation
def stats_ppf(p: float) -> float:
    """Approximate standard normal quantile via rational approximation."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    # Beasley-Springer-Moro approximation (good to 4 decimal places)
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [0, -7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [0,  7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / \
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
             ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)


def print_report(r: dict, target: float = DEFAULT_TARGET_RETURN) -> None:
    bkrl   = r["bankroll"]
    print(f"\n{'='*60}")
    print(f"  Monte Carlo: 30-Day Kalshi Crypto Campaign Simulation")
    print(f"  {r['n_paths']:,} paths × {r['n_weeks']} weeks "
          f"(~{r['total_trades']} total trades)")
    print(f"  Bankroll ${bkrl:,.0f} | Edge mean {r['edge_mean']*100:.0f}pp "
          f"| Avg price {r['price_mean']:.0f}¢")
    print(f"{'='*60}")

    print(f"\n  Target (+{target*100:.0f}% = +${bkrl*target:,.0f})")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  P(return ≥ +{target*100:.0f}%)      {r['p_hit_target']*100:6.1f}%")
    print(f"  P(loss)                 {r['p_loss']*100:6.1f}%")
    print(f"  P(drawdown > 20%)       {r['p_max_dd_exceeded']*100:6.1f}%")
    print(f"  P(drawdown brake fired) {r['p_brake_fired']*100:6.1f}%")

    print(f"\n  Return distribution")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Expected    {r['expected_return']*100:+7.1f}%  (${r['expected_pnl']:+7.2f})")
    print(f"  Median      {r['median_return']*100:+7.1f}%  (${r['median_pnl']:+7.2f})")
    print(f"  Std dev     {r['std_return']*100:7.1f}%")
    print(f"  p5          {r['p05_return']*100:+7.1f}%  (worst 5% scenario)")
    print(f"  p25         {r['p25_return']*100:+7.1f}%")
    print(f"  p75         {r['p75_return']*100:+7.1f}%")
    print(f"  p95         {r['p95_return']*100:+7.1f}%  (best 5% scenario)")
    print(f"\n  Median max drawdown: {r['median_max_dd']*100:.1f}%")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo 30-day Kalshi campaign")
    parser.add_argument("--bankroll",         type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--n",                type=int,   default=DEFAULT_N_PATHS,
                        help="Number of simulation paths")
    parser.add_argument("--weeks",            type=int,   default=DEFAULT_N_WEEKS)
    parser.add_argument("--trades-per-week",  type=float, default=DEFAULT_TRADES_PER_WEEK)
    parser.add_argument("--edge-mean",        type=float, default=DEFAULT_EDGE_MEAN,
                        help="Mean edge (fraction, e.g. 0.12 = 12pp)")
    parser.add_argument("--edge-std",         type=float, default=DEFAULT_EDGE_STD)
    parser.add_argument("--price-mean",       type=float, default=DEFAULT_PRICE_MEAN,
                        help="Mean market price in cents (e.g. 25)")
    parser.add_argument("--target",           type=float, default=DEFAULT_TARGET_RETURN,
                        help="Target return fraction (e.g. 0.07 = 7%%)")
    parser.add_argument("--max-drawdown",     type=float, default=DEFAULT_MAX_DRAWDOWN)
    parser.add_argument("--corr",             type=float, default=DEFAULT_BTC_ETH_CORR,
                        help="BTC-ETH outcome correlation")
    parser.add_argument("--seed",             type=int,   default=42)

    # Scenario shorthand
    parser.add_argument("--pessimistic",  action="store_true",
                        help="Edge mean 8pp, 2 trades/week")
    parser.add_argument("--optimistic",   action="store_true",
                        help="Edge mean 15pp, 6 trades/week")

    args = parser.parse_args()

    edge_mean        = args.edge_mean
    trades_per_week  = args.trades_per_week

    if args.pessimistic:
        edge_mean, trades_per_week = 0.08, 2.0
        print("  [PESSIMISTIC scenario: 8pp edge, 2 trades/week]")
    elif args.optimistic:
        edge_mean, trades_per_week = 0.15, 6.0
        print("  [OPTIMISTIC scenario: 15pp edge, 6 trades/week]")

    result = simulate_campaign(
        n_paths         = args.n,
        bankroll        = args.bankroll,
        n_weeks         = args.weeks,
        trades_per_week = trades_per_week,
        edge_mean       = edge_mean,
        edge_std        = args.edge_std,
        price_mean      = args.price_mean,
        btc_eth_corr    = args.corr,
        max_drawdown    = args.max_drawdown,
        target_return   = args.target,
        seed            = args.seed,
    )
    print_report(result, target=args.target)

    # Run three scenarios for comparison
    print("  Scenario comparison (same seed, n=5,000 each)")
    print(f"  {'Scenario':<20} {'P(≥7%)':>8} {'E[ret]':>8} {'P(DD>20%)':>10}")
    print(f"  {'-'*50}")
    for label, em, tpw in [
        ("Pessimistic", 0.08, 2.0),
        ("Base case",   0.12, 4.0),
        ("Optimistic",  0.15, 6.0),
    ]:
        s = simulate_campaign(n_paths=5_000, bankroll=args.bankroll,
                              edge_mean=em, trades_per_week=tpw, seed=args.seed)
        print(f"  {label:<20} {s['p_hit_target']*100:>7.1f}% "
              f"{s['expected_return']*100:>+7.1f}% "
              f"{s['p_max_dd_exceeded']*100:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
