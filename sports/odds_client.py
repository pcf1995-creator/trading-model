"""
sports/odds_client.py — The Odds API client.

Source: the-odds-api.com  (free tier: 500 requests/month, enough for daily scans)
Sign up at: https://the-odds-api.com

Set env var:  ODDS_API_KEY=<your key>

Sports covered:
  americanfootball_ncaaf   — College football
  americanfootball_nfl     — NFL

Markets covered:
  h2h       — Moneyline (head-to-head)
  spreads   — Point spread
  totals    — Over/under

Books used for line-shopping (US-accessible, in priority order):
  pinnacle     — Sharpest book; treat as the "closing line benchmark"
  draftkings
  fanduel
  betmgm
  caesars
  pointsbetus

The free tier returns "best available" across all books; you see the
remaining request quota in response headers (x-requests-remaining).
"""

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

ODDS_BASE   = "https://api.the-odds-api.com/v4"
_API_KEY    = os.getenv("ODDS_API_KEY", "")

SPORTS = {
    "ncaaf": "americanfootball_ncaaf",
    "nfl":   "americanfootball_nfl",
}

# Books to pull — Pinnacle is the line-quality benchmark; DK/FD are where
# most users actually place bets (highest limits, deepest markets).
BOOKMAKERS = [
    "pinnacle", "draftkings", "fanduel", "betmgm", "caesars", "pointsbetus",
]

# Standard American juice: -110 on both sides
STANDARD_JUICE = -110


def american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return -american_odds / (-american_odds + 100)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if american_odds > 0:
        return american_odds / 100 + 1
    return 100 / -american_odds + 1


def no_vig_prob(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove the bookmaker's vig (overround) to get true implied probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def _get(endpoint: str, params: dict | None = None) -> dict | list:
    if not _API_KEY:
        logger.warning("ODDS_API_KEY not set — returning empty odds data")
        return []
    url  = ODDS_BASE + endpoint
    p    = {"apiKey": _API_KEY, **(params or {})}
    try:
        resp = requests.get(url, params=p, timeout=20)
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.debug(f"Odds API remaining requests: {remaining}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Odds API {endpoint} failed: {e}")
        return []


# ── Core fetch ──────────────────────────────────────────────────────────────

def get_odds(sport: str = "ncaaf",
             markets: list[str] | None = None,
             bookmakers: list[str] | None = None) -> list[dict]:
    """
    Fetch odds for all upcoming games in a sport.

    Returns a list of game dicts, each with:
      id, sport_key, home_team, away_team, commence_time,
      bookmakers: [{ key, title, markets: [{ key, outcomes }] }]

    market keys: 'h2h', 'spreads', 'totals'
    """
    markets    = markets    or ["h2h", "spreads", "totals"]
    bookmakers = bookmakers or BOOKMAKERS
    sport_key  = SPORTS.get(sport, sport)

    return _get(
        f"/sports/{sport_key}/odds",
        params={
            "regions"    : "us",
            "markets"    : ",".join(markets),
            "bookmakers" : ",".join(bookmakers),
            "oddsFormat" : "american",
        },
    )


# ── Parsing helpers ─────────────────────────────────────────────────────────

def best_line(game: dict, market: str, side: str) -> dict | None:
    """
    Find the best (most favorable) line for `side` in `market` across all books.

    market: 'h2h', 'spreads', 'totals'
    side:   team name for h2h/spreads; 'Over'/'Under' for totals

    Returns: { book, price (American odds), point (for spreads/totals) }
    """
    best: dict | None = None
    for bm in game.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                if outcome.get("name") != side:
                    continue
                price = outcome.get("price")
                if price is None:
                    continue
                if best is None or price > best["price"]:
                    best = {
                        "book" : bm["key"],
                        "price": price,
                        "point": outcome.get("point"),
                    }
    return best


def parse_game_lines(game: dict) -> dict:
    """
    Extract the best available lines for h2h, spreads, and totals from a
    raw Odds API game object.

    Returns:
    {
      "id": ..., "home": ..., "away": ..., "commence_time": ...,
      "moneyline": { "home": {book, price}, "away": {book, price} },
      "spread":    { "home": {book, price, point}, "away": ... },
      "total":     { "over": {book, price, point}, "under": ... },
    }
    """
    home = game["home_team"]
    away = game["away_team"]
    return {
        "id"           : game["id"],
        "home"         : home,
        "away"         : away,
        "commence_time": game.get("commence_time"),
        "moneyline": {
            "home": best_line(game, "h2h",     home),
            "away": best_line(game, "h2h",     away),
        },
        "spread": {
            "home": best_line(game, "spreads", home),
            "away": best_line(game, "spreads", away),
        },
        "total": {
            "over" : best_line(game, "totals", "Over"),
            "under": best_line(game, "totals", "Under"),
        },
    }


def get_parsed_odds(sport: str = "ncaaf") -> list[dict]:
    """Convenience: fetch + parse all upcoming games for a sport."""
    raw = get_odds(sport)
    return [parse_game_lines(g) for g in raw]
