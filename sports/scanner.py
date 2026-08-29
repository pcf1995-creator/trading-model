"""
sports/scanner.py — EV scanner for college football and NFL betting markets.

Usage:
  python scanner.py                   # scan NCAAF (NFL needs a data source)
  python scanner.py --sport ncaaf     # college football only
  python scanner.py --week 3          # specific week
  python scanner.py --min-ev 0.03     # lower edge threshold (3pp)
  python scanner.py --save            # save recommendations to Supabase

Edge thresholds (configurable via CLI):
  NFL spreads/totals : 5pp
  CFB Power 4        : 5pp
  CFB Group of 5     : 4pp

These are intentionally tiered — more efficient markets need a higher bar.
No conference is hard-excluded; a game just has to clear its threshold.

Games are skipped (not predicted) when either team is unresolvable or has no
ratings — typically FCS opponents, which S&P+ does not cover. Predicting those
from a zero-filled default is what previously produced ~1000 bogus edges.

Required env vars (see cfbd_client.py and odds_client.py):
  CFBD_API_KEY    — from collegefootballdata.com
  ODDS_API_KEY    — from the-odds-api.com
  SUPABASE_URL    — for --save (reuses kalshi db infrastructure)
  SUPABASE_KEY    — for --save
"""

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
_SPORTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SPORTS_DIR.parent))
sys.path.insert(0, str(_SPORTS_DIR))

from odds_client import get_parsed_odds, american_to_prob, no_vig_prob
from efficiency  import (
    build_ratings, predict_game, total_model_prob, spread_model_prob,
    moneyline_model_prob, compute_ev_total, compute_ev_spread,
    compute_kelly_sports, UnknownTeamError, UnratedTeamError,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Conference tiers (for tiered edge thresholds) ───────────────────────────
POWER4_CONFERENCES = {
    "SEC", "Big Ten", "ACC", "Big 12",
    "Pac-12",          # legacy name; data may still use it
}

# ── Thresholds ───────────────────────────────────────────────────────────────
DEFAULT_MIN_EV_NFL  = 0.05   # 5pp edge for NFL
DEFAULT_MIN_EV_P4   = 0.05   # 5pp for Power 4
DEFAULT_MIN_EV_G5   = 0.04   # 4pp for Group of 5 / mid-major
DEFAULT_MAX_KELLY   = 0.05   # cap at 5% of bankroll per bet

# Longshot moneylines are where a normal-distribution model is least
# trustworthy (the tails are far too thin to price a +5000 dog). Skip them.
MAX_ML_ODDS = 600

# ── Bankroll (paper-trading default) ─────────────────────────────────────────
DEFAULT_BANKROLL = 500.0

# Diagnostics from the most recent run_scan(), for the dashboard to surface.
LAST_SCAN_STATS: dict = {}


def _min_ev_for_game(conference: str | None, nfl: bool) -> float:
    if nfl:
        return DEFAULT_MIN_EV_NFL
    if conference in POWER4_CONFERENCES:
        return DEFAULT_MIN_EV_P4
    return DEFAULT_MIN_EV_G5


def score_game(parsed_odds: dict, ratings, year: int,
               nfl: bool = False) -> list[dict]:
    """
    Score one game across totals, spreads, and moneylines.

    parsed_odds: output of odds_client.parse_game_lines()
    Raises UnknownTeamError / UnratedTeamError if the game cannot be modelled;
    the caller decides how to account for those.
    """
    home = parsed_odds["home"]
    away = parsed_odds["away"]

    prediction = predict_game(home, away, ratings, nfl=nfl)

    home_c = ratings.resolve(home)
    min_ev = _min_ev_for_game(ratings.get(home_c).get("conference"), nfl)

    results: list[dict] = []
    base = {
        "game_id"     : parsed_odds["id"],
        "home"        : home,
        "away"        : away,
        "commence"    : parsed_odds.get("commence_time", ""),
        "sport"       : "nfl" if nfl else "ncaaf",
        "model_total" : prediction["expected_total"],
        "model_spread": prediction["expected_spread"],
        "home_games"  : prediction["home_games"],
        "away_games"  : prediction["away_games"],
        "weather_adj" : prediction["weather_adj"],
    }

    def _add(market: str, side: str, line_obj: dict, model_prob: float,
             novig_prob: float, point, delta: float) -> None:
        odds  = line_obj["price"]
        ev    = compute_ev_spread(model_prob, odds)
        if ev < min_ev:
            return
        kelly = compute_kelly_sports(model_prob, odds, DEFAULT_MAX_KELLY)
        if kelly <= 0:
            return
        results.append({
            **base,
            "market"         : market,
            "side"           : side,
            "model_prob"     : round(model_prob, 4),
            # Vig removed, so "edge" is against the book's true price, not
            # against its margin. Otherwise every bet looks ~2pp better.
            "market_prob"    : round(novig_prob, 4),
            "edge"           : round(model_prob - novig_prob, 4),
            "ev"             : round(ev, 4),
            "kelly_pct"      : round(kelly * 100, 2),
            "odds"           : odds,
            "line"           : point,
            "book"           : line_obj["book"],
            "model_vs_market": round(delta, 2),
        })

    # ── Totals ──────────────────────────────────────────────────────────────
    over_line  = parsed_odds["total"]["over"]
    under_line = parsed_odds["total"]["under"]
    if over_line and under_line and over_line.get("point") is not None:
        nv_over, nv_under = no_vig_prob(american_to_prob(over_line["price"]),
                                        american_to_prob(under_line["price"]))
        for side, line_obj, novig in (("Over",  over_line,  nv_over),
                                      ("Under", under_line, nv_under)):
            point = line_obj.get("point")
            if point is None:
                continue
            probs = total_model_prob(prediction, float(point), nfl=nfl)
            model_prob = probs["prob_over"] if side == "Over" else probs["prob_under"]
            _add("total", side, line_obj, model_prob, novig, point,
                 probs["model_vs_market"])

    # ── Spreads ─────────────────────────────────────────────────────────────
    home_spread = parsed_odds["spread"]["home"]
    away_spread = parsed_odds["spread"]["away"]
    if home_spread and away_spread and home_spread.get("point") is not None:
        nv_home, nv_away = no_vig_prob(american_to_prob(home_spread["price"]),
                                       american_to_prob(away_spread["price"]))
        for team, line_obj, novig, is_home in (
            (home, home_spread, nv_home, True),
            (away, away_spread, nv_away, False),
        ):
            point = line_obj.get("point")
            if point is None:
                continue
            # Each side is priced at its own book's number, so evaluate each
            # against the point actually being offered.
            probs = spread_model_prob(prediction, float(point) if is_home
                                      else -float(point), nfl=nfl)
            model_prob = (probs["prob_home_cover"] if is_home
                          else probs["prob_away_cover"])
            delta = (probs["model_vs_market"] if is_home
                     else -probs["model_vs_market"])
            _add("spread", f"{team} {float(point):+g}", line_obj,
                 model_prob, novig, point, delta)

    # ── Moneyline ───────────────────────────────────────────────────────────
    ml_home = parsed_odds["moneyline"]["home"]
    ml_away = parsed_odds["moneyline"]["away"]
    if ml_home and ml_away:
        nv_home, nv_away = no_vig_prob(american_to_prob(ml_home["price"]),
                                       american_to_prob(ml_away["price"]))
        # Shrink toward the margin the market implies, not toward a pick'em.
        # Prefer the spread market; otherwise invert the no-vig moneyline.
        if home_spread and home_spread.get("point") is not None:
            anchor = -float(home_spread["point"])
        else:
            from scipy.stats import norm
            from efficiency import SIGMA_SPREAD_NFL, SIGMA_SPREAD_CFB
            sig = SIGMA_SPREAD_NFL if nfl else SIGMA_SPREAD_CFB
            anchor = float(norm.ppf(min(max(nv_home, 1e-4), 1 - 1e-4)) * sig)

        win_probs = moneyline_model_prob(prediction, anchor, nfl=nfl)
        for team, line_obj, novig, prob in (
            (home, ml_home, nv_home, win_probs["prob_home_win"]),
            (away, ml_away, nv_away, win_probs["prob_away_win"]),
        ):
            if abs(line_obj["price"]) > MAX_ML_ODDS:
                continue
            delta = (prediction["expected_spread"] if team == home
                     else -prediction["expected_spread"])
            _add("moneyline", team, line_obj, prob, novig, None, delta)

    return results


def run_scan(sport: str = "both",
             year: int | None = None,
             week: int | None = None,
             min_ev: float | None = None,
             bankroll: float = DEFAULT_BANKROLL,
             save: bool = False) -> list[dict]:
    """Full scan: build ratings → fetch odds → score every game → filter by EV."""
    global LAST_SCAN_STATS
    year = year or date.today().year
    all_recs: list[dict] = []
    stats = {"games": 0, "scored": 0, "skipped_unknown": 0,
             "skipped_unrated": 0, "skipped_nfl": False, "unknown_names": []}

    sports_to_scan = ["ncaaf", "nfl"] if sport == "both" else [sport]

    for sp in sports_to_scan:
        nfl = (sp == "nfl")
        logger.info(f"\n{'='*55}")
        logger.info(f"  Scanning {sp.upper()}  (year={year}, week={week or 'all'})")
        logger.info(f"{'='*55}")

        logger.info("Building team efficiency ratings...")
        ratings = build_ratings(year, through_week=week, nfl=nfl)

        if nfl:
            # CFBD is college-only. Skip rather than error until an NFL data
            # source (nfl_client.py) is wired up.
            logger.warning("NFL ratings unavailable (no NFL data source yet) — skipping NFL.")
            stats["skipped_nfl"] = True
            continue

        if len(ratings.rated_teams()) == 0:
            raise RuntimeError(
                "No NCAAF ratings loaded — CFBD_API_KEY is missing or invalid, or "
                "the API returned no data for this season. Check the key in "
                "Streamlit secrets (Settings → Secrets)."
            )

        logger.info("Fetching market odds...")
        games = get_parsed_odds(sp)

        seen_ids: set[str] = set()
        for game in games:
            gid = game.get("id")
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            stats["games"] += 1
            try:
                recs = score_game(game, ratings, year, nfl=nfl)
            except UnknownTeamError as e:
                stats["skipped_unknown"] += 1
                if len(stats["unknown_names"]) < 10:
                    stats["unknown_names"].append(str(e))
                logger.debug(f"skip {game['away']} @ {game['home']}: {e}")
                continue
            except UnratedTeamError as e:
                stats["skipped_unrated"] += 1
                logger.debug(f"skip {game['away']} @ {game['home']}: {e}")
                continue
            stats["scored"] += 1
            if min_ev is not None:
                recs = [r for r in recs if r["ev"] >= min_ev]
            all_recs.extend(recs)

        logger.info(f"  {stats['games']} games | {stats['scored']} scored | "
                    f"{stats['skipped_unknown']} unresolved names | "
                    f"{stats['skipped_unrated']} unrated (FCS)")

        # If we couldn't resolve essentially anything, the alias table is
        # broken — surface that instead of silently returning nothing.
        if stats["games"] > 0 and stats["scored"] == 0 and stats["skipped_unknown"] > 0:
            raise RuntimeError(
                f"None of the {stats['games']} games could be matched to CFBD "
                f"teams (e.g. {stats['unknown_names'][:3]}). The team-name alias "
                "table failed to load — check CFBD_API_KEY."
            )

    all_recs.sort(key=lambda r: r["ev"], reverse=True)
    LAST_SCAN_STATS = stats

    _print_results(all_recs, bankroll)
    if save and all_recs:
        _save_recommendations(all_recs, bankroll)
    return all_recs


def _print_results(recs: list[dict], bankroll: float) -> None:
    if not recs:
        print("\n  No bets cleared the edge threshold today.\n")
        return

    totals  = [r for r in recs if r["market"] == "total"]
    spreads = [r for r in recs if r["market"] == "spread"]
    mls     = [r for r in recs if r["market"] == "moneyline"]

    def _table(rows: list[dict], label: str) -> None:
        if not rows:
            return
        print(f"\n  [{label}]")
        print(f"  {'Game':<32} {'Side':<22} {'Odds':>5} {'Line':>6} "
              f"{'Mdl%':>6} {'Mkt%':>6} {'Edge':>6} {'EV':>6} {'Kelly':>6} {'Book':<12}")
        print(f"  {'-'*120}")
        for r in rows:
            game_str = f"{r['away'][:14]} @ {r['home'][:14]}"
            bet_d    = bankroll * r["kelly_pct"] / 100
            line_str = f"{r['line']:+.1f}" if r["line"] is not None else "—"
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
    """Persist recommendations to the Supabase sports_bets table."""
    try:
        from shared.db_common import _get_client
        client = _get_client()
        if not client:
            logger.warning("No Supabase client — skipping save")
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [{
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
            "bet_dollars" : round(bankroll * r["kelly_pct"] / 100, 2),
            "status"      : "open",
            "result"      : None,
            "pnl_dollars" : None,
            "scanned_at"  : now,
        } for r in recs]
        client.table("sports_bets").insert(rows).execute()
        logger.info(f"Saved {len(rows)} recommendations to sports_bets")
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

    run_scan(sport=args.sport, year=args.year, week=args.week,
             min_ev=args.min_ev, bankroll=args.bankroll, save=args.save)


if __name__ == "__main__":
    main()
