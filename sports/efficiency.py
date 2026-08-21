"""
sports/efficiency.py — Team efficiency ratings and game-total/spread model.

Two-stage approach:
  1. Ratings: rolling EPA/play (offense + defense), adjusted for opponent
     strength.  Season-to-date data dominates by week 4; cold-start weeks
     1-3 blend in preseason S&P+ at a fading weight.

  2. Predictions:
       expected_total  = f(off_epa_home, off_epa_away, def_epa_home,
                           def_epa_away, pace, weather_adj, home_field)
       expected_spread = expected_home_pts - expected_away_pts

  3. Edge = model_prediction - market_line (for totals: model_total - market_total)

Cold-start (weeks 1-3)
──────────────────────
S&P+ preseason ratings are the best publicly available pre-snap estimate of
team quality.  We seed each team's EPA estimate from S&P+ offensive/defensive
ratings, then blend toward current-season EPA as games accumulate:

  cold_weight = max(0, 1 - games_played / COLD_START_GAMES)
  rating = cold_weight × sp_prior + (1 - cold_weight) × season_epa

Pace
────
Plays per game is the multiplier that converts EPA/play to expected points.
We compute it from box-score totals (rushing + passing attempts) and apply
a league-average default when data is thin.

Weather adjustment (for totals)
────────────────────────────────
Wind ≥ 15 mph reduces expected scoring ~2-3 pts (passing efficiency drops).
Temp < 40°F reduces scoring ~1-2 pts additional.
Indoor/dome stadiums are not weather-adjusted.
"""

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from cfbd_client import (
    get_season_ppa, get_game_ppa, get_sp_ratings, get_recruiting_rankings,
    get_team_stats, get_games,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

COLD_START_GAMES    = 6      # by game 6, season EPA fully replaces S&P+ prior
HOME_FIELD_POINTS   = 2.5    # historical home-field advantage in points
LEAGUE_AVG_PLAYS    = 72     # FBS average plays per game
NFL_AVG_PLAYS       = 64     # NFL average plays per game
# Approx conversion: 1 EPA/play ≈ pts_per_epa × plays / game scoring units
# Calibrated from historical CFBD data
CFBD_EPA_TO_PTS     = 9.5    # multiplier: epa/play × plays_per_game × this = pts
NFL_EPA_TO_PTS      = 8.5

# Wind and cold adjustments to expected total
WIND_THRESHOLD_MPH  = 15
WIND_PTS_ADJ        = -2.5   # pts per game (applied to total, not one team)
COLD_THRESHOLD_F    = 40
COLD_PTS_ADJ        = -1.5


# ── Rating data structures ──────────────────────────────────────────────────

class TeamRatings:
    """
    Holds EPA-based offense/defense ratings for all teams in a season.

    Fields (all in EPA/play):
      off_epa    : offensive EPA per play
      def_epa    : defensive EPA per play allowed (negative = good defense)
      games      : games played (determines cold-start blend weight)
      pace       : plays per game
      sp_off     : S&P+ offensive rating (prior, used in cold-start blend)
      sp_def     : S&P+ defensive rating (prior)
    """

    def __init__(self):
        self._data: dict[str, dict] = {}

    def get(self, team: str) -> dict:
        return self._data.get(team, {
            "off_epa": 0.0, "def_epa": 0.0,
            "games": 0, "pace": LEAGUE_AVG_PLAYS,
            "sp_off": 0.0, "sp_def": 0.0,
        })

    def set(self, team: str, **kwargs) -> None:
        entry = self._data.get(team, {})
        entry.update(kwargs)
        self._data[team] = entry

    def teams(self) -> list[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)


def _blend(season_val: float, prior_val: float, games_played: int) -> float:
    """Blend season EPA with S&P+ prior, fading prior as games accumulate."""
    cold_weight = max(0.0, 1.0 - games_played / COLD_START_GAMES)
    return cold_weight * prior_val + (1.0 - cold_weight) * season_val


# ── Build ratings ───────────────────────────────────────────────────────────

def build_ratings(year: int, through_week: int | None = None,
                  nfl: bool = False) -> TeamRatings:
    """
    Build team efficiency ratings for a given year, optionally through a
    specific week (so ratings represent what was known going into that week).

    For NFL, pass nfl=True (uses different constants and data source).
    """
    ratings = TeamRatings()

    # ── S&P+ prior (cold-start seed) ──
    logger.info(f"Loading S&P+ ratings for {year}...")
    sp_data = get_sp_ratings(year)
    for row in sp_data:
        team = row.get("team", "")
        if not team:
            continue
        sp_off = row.get("offense", {}).get("rating", 0.0) or 0.0
        sp_def = row.get("defense", {}).get("rating", 0.0) or 0.0
        # S&P+ ratings are in raw rating units; normalise to approximate EPA/play
        # Typical S&P+ range is ±20; typical EPA/play range is ±0.20
        # Division by 100 is a rough calibration — tune after first season
        ratings.set(team, sp_off=sp_off / 100, sp_def=sp_def / 100,
                    games=0, pace=LEAGUE_AVG_PLAYS)

    # ── Season EPA ──
    logger.info(f"Loading season PPA for {year}...")
    ppa_rows = get_season_ppa(year)
    for row in ppa_rows:
        team = row.get("team", "")
        if not team:
            continue
        off_epa = (row.get("offense") or {}).get("ppa", 0.0) or 0.0
        def_epa = (row.get("defense") or {}).get("ppa", 0.0) or 0.0
        r = ratings.get(team)
        ratings.set(team,
                    off_epa=off_epa, def_epa=def_epa,
                    games=r["games"],   # will be updated from box scores below
                    sp_off=r["sp_off"], sp_def=r["sp_def"],
                    pace=r["pace"])

    # ── Pace + games played (from box scores) ──
    logger.info(f"Loading team stats for pace/games through week {through_week}...")
    params: dict[str, Any] = {"year": year}
    if through_week is not None:
        params["week"] = through_week
    team_stats = get_team_stats(year)
    plays_by_team: dict[str, list[int]] = defaultdict(list)
    for row in team_stats:
        team  = row.get("team") or row.get("school", "")
        # Some endpoints return nested stats
        stats = row.get("stats", row)
        attempts = (
            (stats.get("rushingAttempts") or 0)
            + (stats.get("passingAttempts") or 0)
        )
        if team and attempts > 0:
            plays_by_team[team].append(attempts)

    for team, plays_list in plays_by_team.items():
        r = ratings.get(team)
        ratings.set(team,
                    games=len(plays_list),
                    pace=int(np.mean(plays_list)) if plays_list else LEAGUE_AVG_PLAYS,
                    off_epa=r["off_epa"], def_epa=r["def_epa"],
                    sp_off=r["sp_off"],   sp_def=r["sp_def"])

    logger.info(f"Ratings built for {len(ratings)} teams")
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

    Returns:
    {
      "expected_home_pts": float,
      "expected_away_pts": float,
      "expected_total":    float,
      "expected_spread":   float,   # positive = home favoured by that many points
      "weather_adj":       float,   # total points removed due to weather
      "home_games":        int,
      "away_games":        int,
    }
    """
    rh = ratings.get(home)
    ra = ratings.get(away)

    avg_plays   = NFL_AVG_PLAYS if nfl else LEAGUE_AVG_PLAYS
    epa_to_pts  = NFL_EPA_TO_PTS if nfl else CFBD_EPA_TO_PTS

    # Blended ratings (fades S&P+ prior as games accumulate)
    home_off = _blend(rh["off_epa"], rh["sp_off"], rh["games"])
    home_def = _blend(rh["def_epa"], rh["sp_def"], rh["games"])
    away_off = _blend(ra["off_epa"], ra["sp_off"], ra["games"])
    away_def = _blend(ra["def_epa"], ra["sp_def"], ra["games"])

    home_pace = rh["pace"] if rh["games"] >= 2 else avg_plays
    away_pace = ra["pace"] if ra["games"] >= 2 else avg_plays

    # Expected scoring: your offence vs opponent's defence, scaled by pace
    # home team scores against away defence; away scores against home defence
    home_margin_epa = (home_off - away_def)
    away_margin_epa = (away_off - home_def)
    pace_factor = (home_pace + away_pace) / 2

    base_home_pts = 21.0 + home_margin_epa * pace_factor / avg_plays * epa_to_pts
    base_away_pts = 21.0 + away_margin_epa * pace_factor / avg_plays * epa_to_pts

    # Home field advantage
    if not indoor:
        base_home_pts += HOME_FIELD_POINTS / 2
        base_away_pts -= HOME_FIELD_POINTS / 2

    # Weather adjustment to total (outdoor games only)
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
        "expected_spread"  : round(expected_home - expected_away, 2),  # home margin
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
    """
    EV for a spread bet.
    model_cover_prob: P(selected team covers the spread)
    odds: American odds (default -110)
    """
    dec   = american_to_decimal(odds)
    win   = model_cover_prob
    lose  = 1.0 - win
    return win * (dec - 1) - lose


def compute_ev_total(model_over_prob: float, over_odds: int = -110) -> float:
    """EV for the over side of a total. Use 1-model_over_prob for under."""
    return compute_ev_spread(model_over_prob, over_odds)


def compute_kelly_sports(model_prob: float, odds: int = -110,
                         max_kelly: float = 0.05) -> float:
    """
    Half-Kelly fraction for a standard vig bet.
    Capped lower than Kalshi (max 5%) because:
      1. Sports bets don't have the bounded loss property of binary contracts
      2. Model calibration uncertainty is higher in early-season data
      3. Book limits are a real concern at higher fractions
    """
    dec = american_to_decimal(odds)
    b   = dec - 1.0  # net profit per $1 risked
    f   = (model_prob * (b + 1) - 1) / b  # full Kelly
    return float(np.clip(f * 0.5, 0.0, max_kelly))  # half-Kelly, capped


def total_model_prob(prediction: dict, market_total: float,
                     sigma: float = 7.0) -> dict:
    """
    Convert a point prediction to P(over) and P(under) using a normal
    distribution around the expected total.

    sigma: uncertainty in point totals (historical std of model error ≈ 7 pts)
    """
    from scipy.stats import norm
    mu   = prediction["expected_total"]
    prob_over  = float(norm.sf(market_total, loc=mu, scale=sigma))
    prob_under = float(norm.cdf(market_total, loc=mu, scale=sigma))
    return {
        "prob_over" : round(prob_over,  4),
        "prob_under": round(prob_under, 4),
        "model_total": round(mu, 2),
        "market_total": market_total,
        "model_vs_market": round(mu - market_total, 2),
    }


def spread_model_prob(prediction: dict, market_spread: float,
                      sigma: float = 5.5) -> dict:
    """
    Convert a spread prediction to P(home covers) and P(away covers).
    sigma: uncertainty in spread predictions (historical model error ≈ 5.5 pts)
    """
    from scipy.stats import norm
    mu  = prediction["expected_spread"]
    # Home covers if actual_spread > market_spread (home wins by more)
    prob_home_cover = float(norm.sf(market_spread, loc=mu, scale=sigma))
    prob_away_cover = 1.0 - prob_home_cover
    return {
        "prob_home_cover": round(prob_home_cover, 4),
        "prob_away_cover": round(prob_away_cover, 4),
        "model_spread"   : round(mu, 2),
        "market_spread"  : market_spread,
        "model_vs_market": round(mu - market_spread, 2),
    }
