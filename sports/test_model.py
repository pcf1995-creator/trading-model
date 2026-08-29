"""
sports/test_model.py — regression tests for the sports EV model.

Run:  python sports/test_model.py

Guards the failure mode that produced ~1000 bogus recommendations: odds-feed
team names ("Syracuse Orange") not matching CFBD names ("Syracuse"), which
silently fell back to zero-filled ratings so every game was predicted 21-21
and every market looked mispriced.

Uses synthetic CFBD/odds fixtures — no API keys or network required.
"""
import sys, logging
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import cfbd_client

TEAMS = [
    {"school": "Syracuse", "mascot": "Orange", "conference": "ACC"},
    {"school": "Iowa", "mascot": "Hawkeyes", "conference": "Big Ten"},
    {"school": "Vanderbilt", "mascot": "Commodores", "conference": "SEC"},
    {"school": "Northern Illinois", "mascot": "Huskies", "conference": "Mid-American"},
    {"school": "New Hampshire", "mascot": "Wildcats", "conference": "CAA"},
]
SP = [
    {"team": "Syracuse",          "offense": {"rating": 30.1}, "defense": {"rating": 24.8}},
    {"team": "Iowa",              "offense": {"rating": 24.0}, "defense": {"rating": 15.2}},
    {"team": "Vanderbilt",        "offense": {"rating": 26.5}, "defense": {"rating": 26.0}},
    {"team": "Northern Illinois", "offense": {"rating": 22.3}, "defense": {"rating": 28.4}},
]
cfbd_client.get_teams      = lambda year=None: TEAMS
cfbd_client.get_sp_ratings = lambda year, team=None: SP
cfbd_client.get_season_ppa = lambda year, team=None: []
cfbd_client.get_team_stats = lambda year, week=None, team=None: []

import efficiency
for fn in ("get_teams", "get_sp_ratings", "get_season_ppa", "get_team_stats"):
    setattr(efficiency, fn, getattr(cfbd_client, fn))

def _ml(price, book="draftkings"):        return {"book": book, "price": price, "point": None}
def _pt(price, point, book="draftkings"): return {"book": book, "price": price, "point": point}

GAMES = [
    # FCS body-bag game — must be skipped entirely
    {"id": "g1", "home": "Syracuse Orange", "away": "New Hampshire Wildcats",
     "commence_time": "2026-09-05T16:00:00Z",
     "moneyline": {"home": _ml(-20000), "away": _ml(5000)},
     "spread":    {"home": _pt(-110, -38.5), "away": _pt(-110, 38.5)},
     "total":     {"over": _pt(-110, 58.5), "under": _pt(-110, 58.5)}},
    # Market agrees with model — should produce nothing
    {"id": "g2", "home": "Iowa Hawkeyes", "away": "Northern Illinois Huskies",
     "commence_time": "2026-09-05T20:15:00Z",
     "moneyline": {"home": _ml(-750), "away": _ml(525)},
     "spread":    {"home": _pt(-110, -17.5), "away": _pt(-110, 17.5)},
     "total":     {"over": _pt(-110, 38.5), "under": _pt(-110, 38.5)}},
    # Market total way off model (model ~56, market 44) — should fire
    {"id": "g3", "home": "Vanderbilt Commodores", "away": "Syracuse Orange",
     "commence_time": "2026-09-05T23:00:00Z",
     "moneyline": {"home": _ml(115), "away": _ml(-135)},
     "spread":    {"home": _pt(-110, 2.5), "away": _pt(-110, -2.5)},
     "total":     {"over": _pt(-110, 44.5), "under": _pt(-110, 44.5)}},
    # Unknown team — must be skipped
    {"id": "g4", "home": "Iowa Hawkeyes", "away": "Nonexistent Tech Aardvarks",
     "commence_time": "2026-09-06T00:00:00Z",
     "moneyline": {"home": _ml(-400), "away": _ml(320)},
     "spread":    {"home": _pt(-110, -10.5), "away": _pt(-110, 10.5)},
     "total":     {"over": _pt(-110, 50.5), "under": _pt(-110, 50.5)}},
]

import odds_client
odds_client.get_parsed_odds = lambda sport="ncaaf": list(GAMES)

import scanner
scanner.get_parsed_odds = odds_client.get_parsed_odds
logging.disable(logging.INFO)

recs = scanner.run_scan(sport="ncaaf", year=2026, bankroll=500)

print("\nSCAN DIAGNOSTICS:", scanner.LAST_SCAN_STATS)
print(f"\nRecommendations: {len(recs)}  (was 1018 with the old code)\n")
for r in recs:
    print(f"  {r['market']:<10} {r['side']:<28} {r['away'][:18]:<20} @ {r['home'][:18]:<20} "
          f"odds {r['odds']:+5}  mdl {r['model_prob']*100:5.1f}%  "
          f"mkt {r['market_prob']*100:5.1f}%  edge {r['edge']*100:+5.1f}pp  "
          f"EV {r['ev']:+.3f}  kelly {r['kelly_pct']:.2f}%")

assert scanner.LAST_SCAN_STATS["skipped_unrated"] == 1, "FCS game not skipped"
assert scanner.LAST_SCAN_STATS["skipped_unknown"] == 1, "unknown team not skipped"
assert scanner.LAST_SCAN_STATS["scored"] == 2, "wrong number of games scored"
assert all(abs(r["odds"]) <= scanner.MAX_ML_ODDS
           for r in recs if r["market"] == "moneyline"), "longshot ML leaked through"
assert not any(r["home"] == "Syracuse Orange" and r["away"] == "New Hampshire Wildcats"
               for r in recs), "FCS game produced a bet"
print("\nAll assertions passed.")
