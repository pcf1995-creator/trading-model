"""
sports/scanner.py — EV scanner for college football and NFL betting markets.

Usage:
  python scanner.py                   # scan both NCAAF + NFL, current week
  python scanner.py --sport ncaaf     # college football only
  python scanner.py --sport nfl       # NFL only
  python scanner.py --week 3          # specific week
  python scanner.py --min-ev 0.03     # lower edge threshold (3pp)
  python scanner.py --save            # save recommendations to Supabase

Edge thresholds (configurable via CLI):
  NFL spreads/totals : 5pp (--min-edge-nfl, default 0.05)
  CFB Power 4        : 5pp
  CFB Group of 5     : 4pp (--min-edge-cfb, default 0.04)

These are intentionally tiered — more efficient markets need a higher bar.
The scanner doesn't hard-exclude any conference; it just needs to clear the
relevant threshold naturally.

Required env vars (see cfbd_client.py and odds_client.py):
  CFBD_API_KEY    — from collegefootballdata.com
  ODDS_API_KEY    — from the-odds-api.com
  SUPABASE_URL    — for --save (reuses kalshi db infrastructure)
  SUPABASE_KEY    — for --save
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
_SPORTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SPORTS_DIR.parent))
sys.path.insert(0, str(_SPORTS_DIR))

import pandas as pd

from cfbd_client import get_games, get_upcoming_games
from odds_client  import get_parsed_odds, american_to_decimal, american_to_prob, no_vig_prob
from efficiency   import (
    build_ratings, predict_game, total_model_prob, spread_model_prob,
    compute_ev_total, compute_ev_spread, compute_kelly_sports,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Conference tiers (for tiered edge thresholds) ───────────────────────────
# Populated from CFBD conference listings; Power 4 lines are more efficient.
POWER4_CONFERENCES = {
    "SEC", "Big Ten", "ACC", "Big 12",
    "Pac-12",          # legacy (programs now in B12 etc. but data may use old name)
}

# ── Thresholds ───────────────────────────────────────────────────────────────
DEFAULT_MIN_EV_NFL  = 0.05   # 5pp edge for NFL
DEFAULT_MIN_EV_P4   = 0.05   # 5pp for Power 4
DEFAULT_MIN_EV_G5   = 0.04   # 4pp for Group of 5 / mid-major
DEFAULT_MAX_KELLY   = 0.05   # cap at 5% of bankroll per bet

# ── Bankroll (paper-trading default) ─────────────────────────────────────────
DEFAULT_BANKROLL = 500.0


def _min_ev_for_game(conference: str | None, nfl: bool) -> float:
    if nfl:
        return DEFAULT_MIN_EV_NFL
    if conference in POWER4_CONFERENCES:
        return DEFAULT_MIN_EV_P4
    return DEFAULT_MIN_EV_G5


def _conference_from_game(game: dict, year: int) -> str | None:
    """Best-effort conference lookup from game dict (CFBD includes this field)."""
    return game.get("home_conference") or game.get("conference")


def score_game(parsed_odds: dict, ratings, year: int,
               nfl: bool = False) -> list[dict]:
    """
    Score one game across totals and spreads.

    parsed_odds: output of odds_client.parse_game_lines()
    Returns list of recommendation dicts (may be empty if no edge found above threshold).
    """
    home = parsed_odds["home"]
    away = parsed_odds["away"]

    # Get prediction from efficiency model
    try:
        prediction = predict_game(home, away, ratings, nfl=nfl)
    except Exception as e:
        logger.debug(f"predict_game failed for {home} vs {away}: {e}")
        return []

    results = []
    game_id = parsed_odds["id"]
    commence = parsed_odds.get("commence_time", "")

    base = {
        "game_id"    : game_id,
        "home"       : home,
        "away"       : away,
        "commence"   : commence,
        "sport"      : "nfl" if nfl else "ncaaf",
        "model_total": prediction["expected_total"],
        "model_spread": prediction["expected_spread"],
        "home_games" : prediction["home_games"],
        "away_games" : prediction["away_games"],
        "weather_adj": prediction["weather_adj"],
    }

    min_ev = DEFAULT_MIN_EV_G5 if not nfl else DEFAULT_MIN_EV_NFL

    # ── Totals ──────────────────────────────────────────────────────────────
    over_line  = parsed_odds["total"]["over"]
    under_line = parsed_odds["total"]["under"]

    if over_line and under_line:
        market_total = over_line["point"]
        if market_total:
            probs = total_model_prob(prediction, float(market_total))
            for side, line, model_prob in [
                ("Over",  over_line,  probs["prob_over"]),
                ("Under", under_line, probs["prob_under"]),
            ]:
                odds = line["price"]
                ev   = compute_ev_total(model_prob, odds)
                kelly = compute_kelly_sports(model_prob, odds, DEFAULT_MAX_KELLY)
                edge = model_prob - american_to_prob(odds)
                if ev >= min_ev:
                    results.append({
                        **base,
                        "market"    : "total",
                        "side"      : side,
                        "model_prob": round(model_prob, 4),
                        "market_prob": round(american_to_prob(odds), 4),
                        "edge"      : round(edge, 4),
                        "ev"        : round(ev, 4),
                        "kelly_pct" : round(kelly * 100, 2),
                        "odds"      : odds,
                        "line"      : market_total,
                        "book"      : line["book"],
                        "model_vs_market": probs["model_vs_market"],
                    })

    # ── Spreads ─────────────────────────────────────────────────────────────
    home_spread = parsed_odds["spread"]["home"]
    away_spread = parsed_odds["spread"]["away"]

    if home_spread and away_spread:
        market_spread = home_spread["point"]   # negative = home favourite
        if market_spread is not None:
            probs = spread_model_prob(prediction, float(market_spread))
            for side, line, model_prob in [
                (f"{home} {'+' if market_spread > 0 else ''}{market_spread}",
                 home_spread, probs["prob_home_cover"]),
                (f"{away} {'+' if -market_spread > 0 else ''}{-market_spread}",
                 away_spread, probs["prob_away_cover"]),
            ]:
                odds  = line["price"]
                ev    = compute_ev_spread(model_prob, odds)
                kelly = compute_kelly_sports(model_prob, odds, DEFAULT_MAX_KELLY)
                edge  = model_prob - american_to_prob(odds)
                if ev >= min_ev:
                    results.append({
                        **base,
                        "market"    : "spread",
                        "side"      : side,
                        "model_prob": round(model_prob, 4),
                        "market_prob": round(american_to_prob(odds), 4),
                        "edge"      : round(edge, 4),
                        "ev"        : round(ev, 4),
                        "kelly_pct" : round(kelly * 100, 2),
                        "odds"      : odds,
                        "line"      : market_spread,
                        "book"      : line["book"],
                        "model_vs_market": probs["model_vs_market"],
                    })

    # ── Moneyline ────────────────────────────────────────────────────────────
    # Derive from spread model (simpler: home win = home covers a spread of 0)
    for team_name, prob_cover in [
        (home, spread_model_prob(prediction, 0.0)["prob_home_cover"]),
        (away, 1 - spread_model_prob(prediction, 0.0)["prob_home_cover"]),
    ]:
        ml_line = (parsed_odds["moneyline"]["home"]
                   if team_name == home
                   else parsed_odds["moneyline"]["away"])
        if not ml_line:
            continue
        odds  = ml_line["price"]
        ev    = compute_ev_spread(prob_cover, odds)
        kelly = compute_kelly_sports(prob_cover, odds, DEFAULT_MAX_KELLY)
        edge  = prob_cover - american_to_prob(odds)
        if ev >= min_ev:
            results.append({
                **base,
                "market"    : "moneyline",
                "side"      : team_name,
                "model_prob": round(prob_cover, 4),
                "market_prob": round(american_to_prob(odds), 4),
                "edge"      : round(edge, 4),
                "ev"        : round(ev, 4),
                "kelly_pct" : round(kelly * 100, 2),
                "odds"      : odds,
                "line"      : None,
                "book"      : ml_line["book"],
                "model_vs_market": round(prediction["expected_spread"]
                                         if team_name == home
                                         else -prediction["expected_spread"], 2),
            })

    return results


def run_scan(sport: str = "both",
             year: int | None = None,
             week: int | None = None,
             min_ev: float | None = None,
             bankroll: float = DEFAULT_BANKROLL,
             save: bool = False) -> list[dict]:
    """
    Full scan: fetch odds → build ratings → score every game → filter by EV.
    """
    year = year or date.today().year
    all_recs: list[dict] = []

    sports_to_scan = (
        ["ncaaf", "nfl"] if sport == "both"
        else [sport]
    )

    for sp in sports_to_scan:
        nfl = (sp == "nfl")
        logger.info(f"\n{'='*55}")
        logger.info(f"  Scanning {sp.upper()}  (year={year}, week={week or 'all'})")
        logger.info(f"{'='*55}")

        # Build ratings once per sport
        logger.info("Building team efficiency ratings...")
        ratings = build_ratings(year, through_week=week, nfl=nfl)

        if nfl:
            # CFBD does not cover the NFL — ratings will always be empty from
            # this source until an NFL-specific data feed (nfl_client.py) is
            # integrated. Skip gracefully rather than raising an error.
            if len(ratings) == 0:
                logger.warning("NFL ratings unavailable from CFBD (expected — NFL data source not yet integrated). Skipping NFL.")
                continue
        else:
            # For NCAAF, empty or all-zero ratings means the CFBD key is missing
            # or the API returned nothing. Refuse to scan rather than emit noise.
            if len(ratings) == 0:
                raise RuntimeError(
                    "No NCAAF ratings loaded — CFBD_API_KEY is missing or the API "
                    "returned no data. Check that CFBD_API_KEY is set in Streamlit secrets."
                )
            sample = [ratings.get(t) for t in ratings.teams()[:20]]
            all_zero = all(r["sp_off"] == 0 and r["sp_def"] == 0 for r in sample)
            if all_zero:
                raise RuntimeError(
                    "NCAAF ratings are all zero — CFBD_API_KEY may be invalid or "
                    "the API returned no data for this year. Verify your key and try again."
                )

        # Fetch odds
        logger.info("Fetching market odds...")
        games = get_parsed_odds(sp)
        logger.info(f"  {len(games)} games with odds found")

        # Deduplicate games by id (Odds API occasionally returns duplicates)
        seen_ids: set[str] = set()
        unique_games = []
        for g in games:
            if g["id"] not in seen_ids:
                seen_ids.add(g["id"])
                unique_games.append(g)
        games = unique_games

        for game in games:
            recs = score_game(game, ratings, year, nfl=nfl)
            if min_ev is not None:
                recs = [r for r in recs if r["ev"] >= min_ev]
            all_recs.extend(recs)

    # Sort by EV
    all_recs.sort(key=lambda r: r["ev"], reverse=True)

    # ── Print results ────────────────────────────────────────────────────────
    _print_results(all_recs, bankroll)

    # ── Save to Supabase ─────────────────────────────────────────────────────
    if save and all_recs:
        _save_recommendations(all_recs, bankroll)

    return all_recs


def _print_results(recs: list[dict], bankroll: float) -> None:
    if not recs:
        print("\n  No bets cleared the edge threshold today.\n")
        return

    # Group by market type
    totals   = [r for r in recs if r["market"] == "total"]
    spreads  = [r for r in recs if r["market"] == "spread"]
    mls      = [r for r in recs if r["market"] == "moneyline"]

    def _table(rows: list[dict], label: str) -> None:
        if not rows:
            return
        print(f"\n  [{label}]")
        print(f"  {'Game':<32} {'Side':<22} {'Odds':>5} {'Line':>6} "
              f"{'Mdl%':>6} {'Mkt%':>6} {'Edge':>6} {'EV':>6} {'Kelly':>6} {'Book':<12}")
        print(f"  {'-'*120}")
        for r in rows:
            game_str  = f"{r['away'][:14]} @ {r['home'][:14]}"
            bet_d     = bankroll * r["kelly_pct"] / 100
            line_str  = f"{r['line']:+.1f}" if r["line"] is not None else "—"
            print(
                f"  {game_str:<32} {r['side']:<22} "
                f"{r['odds']:>+5}  {line_str:>6}  "
                f"{r['model_prob']*100:>5.1f}%  {r['market_prob']*100:>5.1f}%  "
                f"{r['edge']*100:>+5.1f}pp  {r['ev']:>+.3f}  "
                f"{r['kelly_pct']:>5.2f}%  {r['book']:<12}"
                f"  (${bet_d:.2f})"
            )

    print(f"\n{'='*65}")
    print(f"  Sports EV Scanner — {date.today()}")
    print(f"  {len(recs)} recommendations  |  Bankroll: ${bankroll:,.0f}")
    print(f"{'='*65}")
    _table(totals,  "TOTALS")
    _table(spreads, "SPREADS")
    _table(mls,     "MONEYLINES")
    print(f"\n{'='*65}\n")


def _save_recommendations(recs: list[dict], bankroll: float) -> None:
    """Persist recommendations to Supabase sports_bets table."""
    try:
        from shared.db_common import _get_client
        client = _get_client()
        if not client:
            logger.warning("No Supabase client — skipping save")
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for r in recs:
            bet_dollars = round(bankroll * r["kelly_pct"] / 100, 2)
            rows.append({
                "game_id"     : r["game_id"],
                "sport"       : r["sport"],
                "home"        : r["home"],
                "away"        : r["away"],
                "commence"    : r["commence"],
                "market"      : r["market"],
                "side"        : r["side"],
                "odds"        : r["odds"],
                "line"        : r["line"],
                "book"        : r["book"],
                "model_prob"  : r["model_prob"],
                "market_prob" : r["market_prob"],
                "edge"        : r["edge"],
                "ev"          : r["ev"],
                "kelly_pct"   : r["kelly_pct"],
                "bet_dollars" : bet_dollars,
                "status"      : "open",
                "result"      : None,
                "pnl_dollars" : None,
                "scanned_at"  : now,
            })
        client.table("sports_bets").insert(rows).execute()
        logger.info(f"Saved {len(rows)} recommendations to sports_bets table")
    except Exception as e:
        logger.error(f"Save to Supabase failed: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Sports EV scanner — NCAAF + NFL")
    parser.add_argument("--sport",    default="both",
                        choices=["ncaaf", "nfl", "both"])
    parser.add_argument("--week",     type=int,   default=None)
    parser.add_argument("--year",     type=int,   default=None)
    parser.add_argument("--min-ev",   type=float, default=None,
                        help="Override minimum EV threshold (e.g. 0.03)")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--save",     action="store_true",
                        help="Save recommendations to Supabase sports_bets table")
    args = parser.parse_args()

    run_scan(
        sport    = args.sport,
        year     = args.year,
        week     = args.week,
        min_ev   = args.min_ev,
        bankroll = args.bankroll,
        save     = args.save,
    )


if __name__ == "__main__":
    main()
