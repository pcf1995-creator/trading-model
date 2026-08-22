"""
sports/cfbd_client.py — College Football Data API client.

Source: collegefootballdata.com  (free API key required)
Sign up at: https://collegefootballdata.com/key

Set env var:  CFBD_API_KEY=<your key>

Endpoints used:
  /games           — schedule + scores
  /games/teams     — per-team box scores (yards, turnovers, etc.)
  /stats/season/advanced — season EPA/play, success rate, explosiveness
  /ppa/games       — per-game EPA breakdown (offense + defense)
  /recruiting/teams — composite recruiting rank (talent proxy for cold-start)
  /ratings/sp      — S&P+ ratings (used to seed early-season estimates)
"""

import logging
import os
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)

CFBD_BASE = "https://api.collegefootballdata.com"
_API_KEY  = os.getenv("CFBD_API_KEY", "")


def _get(endpoint: str, params: dict | None = None) -> list[dict]:
    if not _API_KEY:
        logger.warning("CFBD_API_KEY not set — returning empty data")
        return []
    headers = {"Authorization": f"Bearer {_API_KEY}"}
    url = CFBD_BASE + endpoint
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"CFBD {endpoint} failed: {e}")
        return []


# ── Schedule / results ──────────────────────────────────────────────────────

def get_games(year: int, week: int | None = None,
              season_type: str = "regular") -> list[dict]:
    """
    Fetch games for a given year (and optionally week).
    season_type: 'regular', 'postseason', 'both'
    Returns list of game dicts with keys:
      id, home_team, away_team, home_points, away_points,
      start_date, neutral_site, conference_game, ...
    """
    params: dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        params["week"] = week
    return _get("/games", params)


def get_upcoming_games(week: int | None = None) -> list[dict]:
    """Games for the current calendar year, current or specified week."""
    return get_games(date.today().year, week=week)


# ── Per-game EPA (the primary signal) ──────────────────────────────────────

def get_game_ppa(year: int, week: int | None = None,
                 team: str | None = None) -> list[dict]:
    """
    Per-game predicted points added (EPA) broken down by offense/defense/phase.
    Each row: { game_id, team, conference, opponent, offense: {ppa, ...},
                defense: {ppa, ...} }
    """
    params: dict[str, Any] = {"year": year, "excludeGarbageTime": True}
    if week is not None:
        params["week"] = week
    if team is not None:
        params["team"] = team
    return _get("/ppa/games", params)


def get_season_ppa(year: int, team: str | None = None) -> list[dict]:
    """
    Season-level EPA/play for all teams (or one team).
    Keys: team, conference, offense (epa, successRate, explosiveness, ...),
          defense (same).
    """
    params: dict[str, Any] = {"year": year, "excludeGarbageTime": True}
    if team is not None:
        params["team"] = team
    return _get("/ppa/teams", params)


# ── Team box scores (for pace = plays per game) ─────────────────────────────

def get_team_stats(year: int, week: int | None = None,
                   team: str | None = None) -> list[dict]:
    """
    Per-game team box stats.  Used to compute plays/game (pace proxy).
    Keys: game_id, team, opponent, rushingAttempts, passingAttempts,
          totalYards, turnovers, ...
    """
    params: dict[str, Any] = {"year": year}
    if week is not None:
        params["week"] = week
    if team is not None:
        params["team"] = team
    return _get("/games/teams", params)


# ── Recruiting / talent (cold-start seed) ──────────────────────────────────

def get_recruiting_rankings(year: int) -> list[dict]:
    """
    Team recruiting composite ratings for a given signing class year.
    Keys: year, team, rank, points  (247Sports composite)
    """
    return _get("/recruiting/teams", {"year": year})


# ── S&P+ ratings (early-season prior) ──────────────────────────────────────

def get_sp_ratings(year: int, team: str | None = None) -> list[dict]:
    """
    Bill Connelly's S&P+ composite ratings.
    Keys: year, team, conference, rating, ranking,
          offense: {rating, ...}, defense: {rating, ...}
    Preseason ratings available for the upcoming year — use as cold-start prior.
    """
    params: dict[str, Any] = {"year": year}
    if team is not None:
        params["team"] = team
    return _get("/ratings/sp", params)


# ── NFL data note ───────────────────────────────────────────────────────────
# CFBD does not cover the NFL. NFL EPA data comes from nflverse R package
# exposed via nfl_client.py (fetches pre-exported CSVs from GitHub).
