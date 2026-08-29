"""
sports/efficiency.py — Team efficiency ratings and game total/spread model.

Pipeline:
  1. Ratings: S&P+ preseason ratings (points space) as the cold-start prior,
     blended into season-to-date EPA/play as games accumulate.
  2. Predictions: expected points for each side → total and spread.
  3. Probabilities: normal distribution around the point estimate.

Team name resolution
────────────────────
The odds feed and CFBD do not agree on team names.  The Odds API returns
"Syracuse Orange" (school + mascot); CFBD keys everything on "Syracuse".
`TeamRatings.resolve()` maps feed names onto canonical CFBD names using the
/teams directory (school, school+mascot, abbreviation, alternate names).

This matters more than it sounds: an unresolved name used to fall through to
a zero-filled default, so *every* game collapsed to 21-21 and every market
looked mispriced.  Unresolved and unrated teams are now hard errors that the
scanner skips, rather than silent league-average predictions.

Cold-start (weeks 0-5)
──────────────────────
S&P+ offensive rating is adjusted points scored per game; defensive rating is
adjusted points allowed per game.  Both are already in points, so the prior
prediction is a direct combination:

  prior_pts_home = sp_off_home + sp_def_away - league_avg_ppg

Season EPA replaces it linearly, fully by game COLD_START_GAMES:

  cold_weight = max(0, 1 - games_played / COLD_START_GAMES)

In-season (points space, EPA is already points)
───────────────────────────────────────────────
EPA/play is expected *points* added per play, so points-per-game follows from
pace directly — no arbitrary scaling multiplier:

  pts = league_avg_ppg + (off_edge + def_edge) × plays_per_game

where off_edge/def_edge are the team's EPA/play relative to the league mean
(means computed from the loaded data, not hardcoded, so the model
self-calibrates each season).

Uncertainty
───────────
Model error for football point predictions is large and well documented:
spread RMSE ≈ 13 pts (CFB), total RMSE ≈ 14 pts.  Using an unrealistically
tight sigma turns a 3-point disagreement with the market into a fake 20pp
edge, so these values are deliberately wide.

Weather adjustment (totals, outdoor only)
─────────────────────────────────────────
Wind ≥ 15 mph reduces expected scoring ~2-3 pts; temp < 40°F another ~1-2.
"""

import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np

from cfbd_client import (
    get_season_ppa, get_sp_ratings, get_team_stats, get_teams,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

COLD_START_GAMES    = 6      # by game 6, season EPA fully replaces the S&P+ prior
HOME_FIELD_POINTS   = 2.5    # historical home-field advantage, in points
LEAGUE_AVG_PLAYS    = 72     # FBS average plays per game
NFL_AVG_PLAYS       = 64     # NFL average plays per game

# Fallbacks only — actual league averages are computed from loaded data.
DEFAULT_CFB_PPG     = 27.5
DEFAULT_NFL_PPG     = 22.5

# Model error (std dev, points). These are intentionally wide: published
# football prediction models land around 13 pts RMSE on spreads. A tighter
# sigma manufactures edges that are not real.
SIGMA_SPREAD_CFB    = 13.0
SIGMA_TOTAL_CFB     = 14.0
SIGMA_SPREAD_NFL    = 13.5
SIGMA_TOTAL_NFL     = 13.0

# Market shrinkage: the fraction of a model-vs-market disagreement we actually
# believe. The closing line is the single best public predictor of football
# outcomes, and this model is not sharper than Pinnacle. When the model says 55
# and the market says 52, most of that 3-point gap is model error, not market
# error — so we only act on a fraction of it.
#
# Without this the model treats every small disagreement as a real edge and
# fires on effectively every game. Raise these only if measured closing-line
# value justifies it.
MARKET_SHRINK_COLD  = 0.20   # weeks 0-5: prior-driven, least trustworthy
MARKET_SHRINK_WARM  = 0.35   # once teams have a real sample of games

# Wind and cold adjustments to expected total
WIND_THRESHOLD_MPH  = 15
WIND_PTS_ADJ        = -2.5   # pts per game (applied to the total, not one team)
COLD_THRESHOLD_F    = 40
COLD_PTS_ADJ        = -1.5


class UnknownTeamError(KeyError):
    """Team name from the odds feed could not be matched to a CFBD team."""


class UnratedTeamError(KeyError):
    """Team is known but has no ratings (typically FCS — no S&P+ coverage)."""


def _norm(name: str) -> str:
    """Normalize a team name for matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


# ── Rating data structures ──────────────────────────────────────────────────

class TeamRatings:
    """
    EPA- and S&P+-based ratings for all teams in a season, plus the name
    alias table used to resolve odds-feed names.

    Per-team fields:
      off_epa  : offensive EPA per play
      def_epa  : defensive EPA per play allowed (lower = better defense)
      games    : games played (drives the cold-start blend weight)
      pace     : plays per game
      sp_off   : S&P+ adjusted points scored per game (prior)
      sp_def   : S&P+ adjusted points allowed per game (prior)
      rated    : True once real data (S&P+ or season EPA) has been loaded
    """

    def __init__(self):
        self._data: dict[str, dict] = {}
        self._alias: dict[str, str] = {}
        # League averages, computed in build_ratings()
        self.avg_ppg     = DEFAULT_CFB_PPG
        self.avg_off_epa = 0.0
        self.avg_def_epa = 0.0

    # ── ratings ──
    def get(self, team: str) -> dict:
        return self._data.get(team, {
            "off_epa": 0.0, "def_epa": 0.0,
            "games": 0, "pace": LEAGUE_AVG_PLAYS,
            "sp_off": 0.0, "sp_def": 0.0, "rated": False,
            "conference": None,
        })

    def set(self, team: str, **kwargs) -> None:
        entry = self._data.get(team, self.get(team))
        entry.update(kwargs)
        self._data[team] = entry

    def teams(self) -> list[str]:
        return list(self._data.keys())

    def is_rated(self, team: str) -> bool:
        return bool(self._data.get(team, {}).get("rated", False))

    def rated_teams(self) -> list[str]:
        return [t for t, r in self._data.items() if r.get("rated")]

    def __len__(self) -> int:
        return len(self._data)

    # ── name resolution ──
    def add_alias(self, alias: str, canonical: str) -> None:
        n = _norm(alias)
        # First writer wins: canonical school names are registered before
        # mascot/abbreviation variants, so exact names never get shadowed.
        if n and n not in self._alias:
            self._alias[n] = canonical

    def resolve(self, name: str) -> str | None:
        """
        Map an odds-feed team name onto a canonical CFBD team name.
        Returns None if no match is found.
        """
        if not name:
            return None
        if name in self._data:
            return name
        n = _norm(name)
        if n in self._alias:
            return self._alias[n]
        # Fall back to stripping trailing mascot words:
        # "New Hampshire Wildcats" → "New Hampshire"
        parts = name.split()
        for cut in range(1, min(3, len(parts))):
            cand = _norm(" ".join(parts[:-cut]))
            if cand in self._alias:
                return self._alias[cand]
        return None


def _blend_weight(games_played: int) -> float:
    """Weight on the S&P+ prior; fades to 0 by COLD_START_GAMES."""
    return max(0.0, 1.0 - games_played / COLD_START_GAMES)


def _shrink_to_market(model_value: float, market_value: float,
                      games_played: int) -> float:
    """
    Pull the model's prediction toward the market line, keeping only the
    fraction of the disagreement we actually believe (see MARKET_SHRINK_*).
    """
    k = (MARKET_SHRINK_COLD if games_played < COLD_START_GAMES
         else MARKET_SHRINK_WARM)
    return market_value + k * (model_value - market_value)


# ── Build ratings ───────────────────────────────────────────────────────────

def build_ratings(year: int, through_week: int | None = None,
                  nfl: bool = False) -> TeamRatings:
    """
    Build team efficiency ratings for a season, optionally through a specific
    week (so ratings reflect what was known going into that week).

    NFL is not covered by CFBD; callers should skip nfl=True until an NFL data
    source is wired up. Returns an empty TeamRatings in that case.
    """
    ratings = TeamRatings()

    if nfl:
        logger.warning("build_ratings(nfl=True): CFBD has no NFL coverage — "
                       "returning empty ratings.")
        return ratings

    # ── Team directory → name alias table ──
    logger.info("Loading CFBD team directory for name resolution...")
    for row in get_teams(year):
        school = (row.get("school") or "").strip()
        if not school:
            continue
        ratings.add_alias(school, school)
        ratings.set(school, conference=row.get("conference"))
        mascot = (row.get("mascot") or "").strip()
        if mascot:
            ratings.add_alias(f"{school} {mascot}", school)
        for key in ("abbreviation", "alt_name1", "alt_name2", "alt_name3"):
            alt = (row.get(key) or "").strip()
            if alt:
                ratings.add_alias(alt, school)
        for alt in (row.get("alternateNames") or []):
            if alt:
                ratings.add_alias(str(alt).strip(), school)

    logger.info(f"  {len(ratings._alias)} name aliases registered")

    # ── S&P+ prior (cold-start seed, already in points) ──
    logger.info(f"Loading S&P+ ratings for {year}...")
    sp_offs: list[float] = []
    for row in get_sp_ratings(year):
        team = (row.get("team") or "").strip()
        if not team:
            continue
        sp_off = (row.get("offense") or {}).get("rating")
        sp_def = (row.get("defense") or {}).get("rating")
        if sp_off is None and sp_def is None:
            continue
        sp_off = float(sp_off or 0.0)
        sp_def = float(sp_def or 0.0)
        ratings.add_alias(team, team)
        ratings.set(team, sp_off=sp_off, sp_def=sp_def, rated=True)
        if sp_off > 0:
            sp_offs.append(sp_off)

    # League average points per game — self-calibrating from S&P+ offense
    # ratings (which are adjusted points scored vs an average defense).
    if sp_offs:
        ratings.avg_ppg = float(np.mean(sp_offs))
    logger.info(f"  S&P+ loaded for {len(sp_offs)} teams "
                f"(league avg {ratings.avg_ppg:.1f} ppg)")

    # ── Season EPA ──
    logger.info(f"Loading season PPA for {year}...")
    off_epas: list[float] = []
    def_epas: list[float] = []
    for row in get_season_ppa(year):
        team = (row.get("team") or "").strip()
        if not team:
            continue
        off_epa = (row.get("offense") or {}).get("ppa")
        def_epa = (row.get("defense") or {}).get("ppa")
        if off_epa is None and def_epa is None:
            continue
        off_epa = float(off_epa or 0.0)
        def_epa = float(def_epa or 0.0)
        ratings.add_alias(team, team)
        ratings.set(team, off_epa=off_epa, def_epa=def_epa, rated=True)
        off_epas.append(off_epa)
        def_epas.append(def_epa)

    if off_epas:
        ratings.avg_off_epa = float(np.mean(off_epas))
        ratings.avg_def_epa = float(np.mean(def_epas))
    logger.info(f"  season EPA loaded for {len(off_epas)} teams")

    # ── Pace + games played (from box scores) ──
    logger.info(f"Loading team box scores through week {through_week or 'all'}...")
    team_stats = get_team_stats(year, week=through_week)
    plays_by_team: dict[str, list[int]] = defaultdict(list)
    for row in team_stats:
        team  = (row.get("team") or row.get("school") or "").strip()
        stats = row.get("stats", row)
        attempts = ((stats.get("rushingAttempts") or 0)
                    + (stats.get("passingAttempts") or 0))
        if team and attempts > 0:
            plays_by_team[team].append(attempts)

    for team, plays_list in plays_by_team.items():
        canonical = ratings.resolve(team) or team
        ratings.set(canonical,
                    games=len(plays_list),
                    pace=int(np.mean(plays_list)))

    logger.info(f"Ratings built: {len(ratings.rated_teams())} rated teams "
                f"of {len(ratings)} known")
    return ratings


# ── Game prediction ─────────────────────────────────────────────────────────

def predict_game(home: str, away: str,
                 ratings: TeamRatings,
                 wind_mph: float = 0.0,
                 temp_f: float = 65.0,
                 indoor: bool = False,
                 nfl: bool = False) -> dict:
    """
    Predict expected total and spread for a game.

    Raises UnknownTeamError if a team name cannot be resolved, or
    UnratedTeamError if a resolved team has no ratings data (e.g. FCS
    opponents, which S&P+ does not cover). Callers should skip those games —
    predicting them from a zero-filled default is what produced thousands of
    bogus recommendations.

    Returns:
      expected_home_pts, expected_away_pts, expected_total,
      expected_spread (positive = home favoured), weather_adj,
      home_games, away_games
    """
    home_c = ratings.resolve(home)
    away_c = ratings.resolve(away)
    if home_c is None or away_c is None:
        unmatched = [n for n, c in ((home, home_c), (away, away_c)) if c is None]
        raise UnknownTeamError(f"unresolved team name(s): {', '.join(unmatched)}")
    if not ratings.is_rated(home_c) or not ratings.is_rated(away_c):
        unrated = [c for c in (home_c, away_c) if not ratings.is_rated(c)]
        raise UnratedTeamError(f"no ratings for: {', '.join(unrated)}")

    rh = ratings.get(home_c)
    ra = ratings.get(away_c)

    avg_plays = NFL_AVG_PLAYS if nfl else LEAGUE_AVG_PLAYS
    avg_ppg   = ratings.avg_ppg

    home_pace = rh["pace"] if rh["games"] >= 2 else avg_plays
    away_pace = ra["pace"] if ra["games"] >= 2 else avg_plays
    pace      = (home_pace + away_pace) / 2

    def _side_points(off: dict, deff: dict) -> float:
        """Expected points for the team whose ratings are `off`, facing `deff`."""
        # Prior (S&P+, already points): own adjusted scoring vs opponent's
        # adjusted points allowed, de-double-counting the league average.
        prior_pts = off["sp_off"] + deff["sp_def"] - avg_ppg

        # Season (EPA/play is expected points per play, so pace converts
        # directly to points — no arbitrary multiplier).
        off_edge = off["off_epa"]  - ratings.avg_off_epa
        def_edge = deff["def_epa"] - ratings.avg_def_epa
        season_pts = avg_ppg + (off_edge + def_edge) * pace

        # Blend on the *less* established of the two teams
        w = _blend_weight(min(off["games"], deff["games"]))
        return w * prior_pts + (1.0 - w) * season_pts

    base_home_pts = _side_points(rh, ra)
    base_away_pts = _side_points(ra, rh)

    # Home field advantage (split across both sides)
    if not indoor:
        base_home_pts += HOME_FIELD_POINTS / 2
        base_away_pts -= HOME_FIELD_POINTS / 2

    # Weather adjustment to the total (outdoor games only)
    weather_adj = 0.0
    if not indoor:
        if wind_mph >= WIND_THRESHOLD_MPH:
            weather_adj += WIND_PTS_ADJ * min(1.0, wind_mph / 25.0)
        if temp_f < COLD_THRESHOLD_F:
            weather_adj += COLD_PTS_ADJ * min(1.0, (COLD_THRESHOLD_F - temp_f) / 30.0)

    expected_home = max(0.0, base_home_pts + weather_adj / 2)
    expected_away = max(0.0, base_away_pts + weather_adj / 2)

    return {
        "expected_home_pts": round(expected_home, 2),
        "expected_away_pts": round(expected_away, 2),
        "expected_total"   : round(expected_home + expected_away, 2),
        "expected_spread"  : round(expected_home - expected_away, 2),
        "weather_adj"      : round(weather_adj, 2),
        "home_games"       : rh["games"],
        "away_games"       : ra["games"],
    }


# ── EV calculation ──────────────────────────────────────────────────────────

def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return odds / 100 + 1.0
    return 100 / abs(odds) + 1.0


def compute_ev_spread(model_cover_prob: float, odds: int = -110) -> float:
    """EV per $1 risked. model_cover_prob: P(selected side wins the bet)."""
    dec  = american_to_decimal(odds)
    win  = model_cover_prob
    return win * (dec - 1) - (1.0 - win)


def compute_ev_total(model_over_prob: float, over_odds: int = -110) -> float:
    """EV for the over side of a total. Use 1-model_over_prob for the under."""
    return compute_ev_spread(model_over_prob, over_odds)


def compute_kelly_sports(model_prob: float, odds: int = -110,
                         max_kelly: float = 0.05) -> float:
    """
    Half-Kelly fraction, capped. Lower cap than Kalshi because:
      1. Sports bets lack the bounded-loss property of binary contracts
      2. Model calibration uncertainty is higher, especially early season
      3. Book limits are a real constraint at higher fractions
    """
    dec = american_to_decimal(odds)
    b   = dec - 1.0
    f   = (model_prob * (b + 1) - 1) / b
    return float(np.clip(f * 0.5, 0.0, max_kelly))


def total_model_prob(prediction: dict, market_total: float,
                     sigma: float | None = None, nfl: bool = False) -> dict:
    """P(over) / P(under) from the point prediction, via a normal distribution."""
    from scipy.stats import norm
    if sigma is None:
        sigma = SIGMA_TOTAL_NFL if nfl else SIGMA_TOTAL_CFB
    raw_mu = prediction["expected_total"]
    games  = min(prediction.get("home_games", 0), prediction.get("away_games", 0))
    mu     = _shrink_to_market(raw_mu, market_total, games)
    prob_over = float(norm.sf(market_total, loc=mu, scale=sigma))
    return {
        "prob_over"      : round(prob_over, 4),
        "prob_under"     : round(1.0 - prob_over, 4),
        "model_total"    : round(raw_mu, 2),
        "shrunk_total"   : round(mu, 2),
        "market_total"   : market_total,
        # Reported as the raw disagreement — that's the interpretable number,
        # even though only a fraction of it is acted on.
        "model_vs_market": round(raw_mu - market_total, 2),
    }


def spread_model_prob(prediction: dict, market_spread: float,
                      sigma: float | None = None, nfl: bool = False,
                      anchor: float | None = None) -> dict:
    """
    P(home covers) / P(away covers).

    market_spread is the home team's handicap (negative = home favoured).
    Home covers when the actual home margin exceeds -market_spread.

    anchor: the market's own implied home margin, used as the shrinkage target.
      Defaults to -market_spread, which is correct for spread bets. Moneyline
      callers MUST pass the margin implied by the spread market — shrinking a
      moneyline toward a required margin of 0 would drag every favourite
      toward a coin flip and invent underdog value that isn't there.
    """
    from scipy.stats import norm
    if sigma is None:
        sigma = SIGMA_SPREAD_NFL if nfl else SIGMA_SPREAD_CFB
    required = -market_spread          # margin the home team must exceed
    raw_mu   = prediction["expected_spread"]
    games    = min(prediction.get("home_games", 0), prediction.get("away_games", 0))
    mu       = _shrink_to_market(raw_mu, required if anchor is None else anchor,
                                 games)
    prob_home_cover = float(norm.sf(required, loc=mu, scale=sigma))
    return {
        "prob_home_cover": round(prob_home_cover, 4),
        "prob_away_cover": round(1.0 - prob_home_cover, 4),
        "model_spread"   : round(raw_mu, 2),
        "shrunk_spread"  : round(mu, 2),
        "market_spread"  : market_spread,
        "model_vs_market": round(raw_mu - required, 2),
    }


def moneyline_model_prob(prediction: dict, market_margin: float,
                         sigma: float | None = None, nfl: bool = False) -> dict:
    """
    P(home wins) / P(away wins).

    market_margin: the home margin the market implies (positive = home
    favoured), taken from the spread market or inverted from the no-vig
    moneyline. It is the shrinkage anchor.

    Use this rather than spread_model_prob(prediction, 0.0): a win is
    "covering a spread of 0", but the shrinkage target is the market's implied
    margin, not zero. Anchoring to zero pulls every favourite toward a coin
    flip and invents enormous underdog edges.
    """
    out = spread_model_prob(prediction, 0.0, sigma=sigma, nfl=nfl,
                            anchor=market_margin)
    return {
        "prob_home_win": out["prob_home_cover"],
        "prob_away_win": out["prob_away_cover"],
        "model_spread" : out["model_spread"],
        "market_margin": market_margin,
        "model_vs_market": round(out["model_spread"] - market_margin, 2),
    }
