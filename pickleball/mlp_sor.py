"""
MLP 2026 Strength of Record (SOR) Calculator

Methodology:
- For each team, look at each event they participated in.
- "Strength of wins"  = avg DUPR of teams you placed above at that event.
- "Strength of losses" = avg DUPR of teams that placed above you at that event.
- Event SOR = strength_of_wins - strength_of_losses.
- Team SOR = mean of event SORs across all events played.

DUPR sources:
  Confirmed values sourced from DUPR.com, The Dink Pickleball, and The Kitchen Pickleball.
  Values marked (est) are estimates based on draft position, known rankings, and comparable players.
  MLP uses doubles DUPR. Scale: ~5.0 (recreational) to ~7.5+ (world-class pro).

Event data is current through MLP Austin (Event 4, June 11-14 2026).
Tied placements mean no win/loss credit vs. each other (used for unknown sub-placings).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Team rosters + DUPR ratings
# ---------------------------------------------------------------------------
# Each entry: (player_name, dupr, is_estimate)
# is_estimate=True means the value was not directly confirmed from a public source.

TEAMS: dict[str, list[tuple[str, float, bool]]] = {
    "New Jersey 5s": [
        ("Will Howells",        6.64, False),
        ("Noe Khlif",           6.49, False),
        ("Martin Emmrich",      6.45, True),
        ("Anna Leigh Waters",   6.56, False),  # from 2025 rating; may have shifted
        ("Jorja Johnson",       6.50, True),   # 2nd overall draft pick / 2025 MLP MVP
        ("Lina Padegimaite",    5.65, True),
    ],
    "St. Louis Shock": [
        ("Hayden Patriquin",    6.90, False),
        ("Gabriel Tardio",      6.88, False),
        ("John Lucian Goins",   6.25, True),
        ("Anna Bright",         6.38, False),
        ("Kate Fahey",          5.83, False),
        ("Elsie Hendershot",    5.85, True),
    ],
    "Los Angeles Mad Drops": [
        ("Ben Johns",           7.12, False),
        ("Max Freeman",         6.50, True),
        ("Gabriel Joseph",      6.30, True),
        ("Catherine Parenteau", 6.16, False),
        ("Jade Kawamoto",       5.99, False),
        ("Samantha Parker",     5.55, True),
    ],
    "Columbus Sliders": [
        ("Andrei Daescu",       6.95, False),
        ("CJ Klinger",          6.70, False),
        ("Alix Truong",         6.20, True),
        ("Parris Todd",         6.00, False),
        # Danni-Elle Townsend (pick #3) was traded to Dallas Flash.
        # Columbus acquired a player in return (identity TBD).
        ("Acquired Player",     5.85, True),
        ("Player 6",            5.80, True),
    ],
    "Brooklyn Pickleball Team": [
        # Dylan Frazier (pick #7) was traded to Texas Ranchers, then on to Miami.
        ("Riley Newman",        6.81, False),
        ("Christian Alshon",    6.50, True),
        ("Chris Haworth",       6.45, True),
        ("Jackie Kawamoto",     5.80, True),
        ("Rachel Rohrabacher",  5.75, True),
        ("Hannah Blatt",        5.65, True),
    ],
    "Texas Ranchers": [
        # Acquired Nico Acevedo from Miami in exchange for Dylan Frazier.
        ("Eric Oncins",         6.50, True),
        ("Nico Acevedo",        6.30, True),
        ("Matthew Barlow",      6.20, True),
        ("Lea Jansen",          6.10, True),   # 4th overall pick
        ("Layne Sleeth",        5.80, True),
        ("Marcela Hones",       5.45, True),
    ],
    "SoCal Hard Eights": [
        ("Will MacKinnon",      6.40, True),   # pick #17
        ("Armaan Bhatia",       6.30, True),   # pick #14
        ("Player 3",            6.20, True),
        ("Meghan Dizon",        6.00, True),   # pick #11
        ("Cailyn Campbell",     5.80, True),   # pick #13
        ("Player 6",            5.60, True),
    ],
    "Dallas Flash": [
        # Acquired Danni-Elle Townsend from Columbus.
        ("JW Johnson",          7.17, False),
        ("Augie Ge",            6.50, True),
        ("Ivan Jakovljevic",    6.40, True),
        ("Danni-Elle Townsend", 5.90, True),   # pick #3 overall, moved here via trade
        ("Brooke Buckner",      5.60, True),
        ("Albie Huang",         5.50, True),
    ],
    "Utah Black Diamonds": [
        ("Connor Garnett",      6.60, True),
        ("Tama Shimabukuro",    6.00, True),   # pick #9
        ("Player 3",            6.20, True),
        ("Allyce Jones",        5.80, True),
        ("Etta Tuionetoa",      5.60, True),
        ("Player 6",            5.50, True),
    ],
    "Florida Smash": [
        ("Cason Campbell",      6.40, True),
        ("Travis Rettenmaier",  6.50, True),
        ("Player 3",            6.10, True),
        ("Zoey Weil",           5.60, True),
        ("Martina Frantova",    5.70, True),
        ("Genie Bouchard",      5.20, True),   # pick #23, tennis-to-pickleball crossover
    ],
    "Carolina Hogs": [
        ("James Delgado",       6.40, True),
        ("DJ Young",            6.50, True),
        ("Brandon French",      6.10, True),
        ("Angie Walker",        5.60, True),
        ("Allison Phillips",    5.80, True),
        ("Ava Ignatowich",      5.90, True),
    ],
    "Atlanta Bouncers": [
        ("Jaume Martinez Vich", 6.75, True),
        ("Jay Devilliers",      6.70, True),
        ("Player 3",            6.30, True),
        ("Jessie Irvine",       5.80, True),
        ("Kaitlyn Christian",   5.70, True),
        ("Daria Walczak",       5.60, True),   # pick #20
    ],
    "Chicago Slice": [
        ("Hunter Johnson",      6.80, True),
        ("Zane Navratil",       6.65, True),
        ("Tom Protzek",         6.20, True),
        ("Mari Humberg",        5.70, True),
        ("Jamie Wei",           5.50, True),
        ("Jalina Ingram",       5.40, True),
    ],
    "Las Vegas Night Owls": [
        ("Blaine Hovenier",     6.20, True),
        ("Roscoe Bellamy",      6.30, True),
        ("Player 3",            6.10, True),
        ("Zoey Wang",           5.50, True),
        ("Player 5",            5.60, True),
        ("Player 6",            5.45, True),
    ],
    "Phoenix Flames": [
        ("Jonathan Truong",     6.20, True),   # pick #10
        ("Cam Chaffin",         6.40, True),   # pick #16
        ("Wyatt Stone",         6.30, True),
        ("Judit Castillo",      5.70, True),   # pick #21
        ("Alexa Schull",        5.60, True),
        ("Player 6",            5.50, True),
    ],
    "Orlando Squeeze": [
        ("Federico Staksrud",   6.70, True),
        ("Jack Sock",           6.20, True),   # pick #8, tennis crossover
        ("Yates Johnson",       6.00, True),
        ("Lacy Schneemann",     5.80, True),
        ("Milan Rane",          5.70, True),
        ("Alex Walker",         5.90, True),
    ],
    "Miami Pickleball Club": [
        # Acquired Dylan Frazier from Texas Ranchers.
        ("Dylan Frazier",       6.92, True),
        ("Player 2",            6.20, True),
        ("Player 3",            6.00, True),
        ("Player 4",            5.80, True),
        ("Player 5",            5.70, True),
        ("Player 6",            5.50, True),
    ],
    "Bay Area Breakers": [
        # International roster with experienced veterans and young players.
        ("Len Yang",            6.10, True),   # pick #18
        ("Player 2",            6.20, True),
        ("Player 3",            6.30, True),
        ("Player 4",            6.00, True),
        ("Mya [Player]",        5.80, True),
        ("Player 6",            5.60, True),
    ],
    "California Black Bears": [
        ("Kiora Kunimoto",      5.80, True),   # pick #15
        ("Player 2",            6.30, True),
        ("Player 3",            6.10, True),
        ("Player 4",            5.70, True),
        ("Player 5",            5.60, True),
        ("Player 6",            5.50, True),
    ],
    "Palm Beach Royals": [
        ("Tyson McGuffin",      6.60, True),   # pick #12, established veteran
        ("Player 2",            6.30, True),
        ("Sofia Sewing",        5.70, True),
        ("Pisnik [Player]",     5.80, True),
        ("Player 5",            5.80, True),
        ("Player 6",            5.50, True),
    ],
}


# ---------------------------------------------------------------------------
# Event results
# ---------------------------------------------------------------------------
# Each event lists placements as a list-of-lists.
# Each inner list is a "tier" (1st, 2nd, ...).
# Multiple teams in one tier = tied (no SOR credit against each other).
# Confirmed placements are at the top; tied buckets at the bottom = uncertain ordering.

EVENT_RESULTS: list[dict] = [
    {
        "name": "MLP Dallas",
        "date": "2026-05-25",
        # Group A (5): Columbus Sliders, Dallas Flash, NJ 5s, Orlando Squeeze, Phoenix Flames
        # Group B (6): Bay Area Breakers, Carolina Hogs, LA Mad Drops, St. Louis Shock,
        #              Texas Ranchers, Utah Black Diamonds
        "placements": [
            ["Los Angeles Mad Drops"],    # 1st  — confirmed
            ["Columbus Sliders"],          # 2nd  — confirmed
            ["New Jersey 5s"],             # 3rd  — estimated (likely Group A #2)
            ["St. Louis Shock"],           # 4th  — estimated (lost at Dallas per reporting)
            # Positions 5–11 unknown; grouped as tied to avoid spurious SOR data.
            [
                "Bay Area Breakers",
                "Carolina Hogs",
                "Dallas Flash",
                "Orlando Squeeze",
                "Phoenix Flames",
                "Texas Ranchers",
                "Utah Black Diamonds",
            ],
        ],
    },
    {
        "name": "MLP Columbus",
        "date": "2026-05-31",
        # Group A (6): Atlanta Bouncers, CA Black Bears, Chicago Slice, Columbus Sliders,
        #              Miami Pickleball Club, NJ 5s
        # Group B (5): Carolina Hogs, Florida Smash, Las Vegas Night Owls, Palm Beach Royals,
        #              St. Louis Shock
        "placements": [
            ["New Jersey 5s"],            # 1st  — confirmed
            ["St. Louis Shock"],          # 2nd  — confirmed
            ["Columbus Sliders"],          # 3rd  — confirmed
            ["Palm Beach Royals"],         # 4th  — confirmed
            ["Las Vegas Night Owls"],      # 5th  — confirmed (beat Atlanta 3-1)
            ["Atlanta Bouncers"],          # 6th  — confirmed
            # 7th–11th unknown order
            [
                "California Black Bears",
                "Chicago Slice",
                "Miami Pickleball Club",
                "Carolina Hogs",
                "Florida Smash",
            ],
        ],
    },
    {
        "name": "MLP St. Louis",
        "date": "2026-06-07",
        # Group A (5): Atlanta Bouncers, Bay Area Breakers, Orlando Squeeze,
        #              Palm Beach Royals, St. Louis Shock
        # Group B (6): Brooklyn Pickleball Team, Las Vegas Night Owls, LA Mad Drops,
        #              Phoenix Flames, SoCal Hard Eights, Utah Black Diamonds
        "placements": [
            ["St. Louis Shock"],           # 1st  — confirmed
            ["Los Angeles Mad Drops"],     # 2nd  — confirmed
            ["Brooklyn Pickleball Team"],  # 3rd  — confirmed
            ["Orlando Squeeze"],           # 4th  — confirmed
            ["Palm Beach Royals"],         # 5th  — confirmed
            ["Las Vegas Night Owls"],      # 6th  — confirmed
            ["Utah Black Diamonds"],       # 7th  — confirmed
            # 8th is one of Bay Area / Atlanta; 9th-10th are SoCal + the other of Bay Area/Atlanta
            ["Bay Area Breakers", "Atlanta Bouncers"],   # tied 8th-9th (uncertain order)
            ["SoCal Hard Eights"],         # 10th  — confirmed (1 standings pt)
            ["Phoenix Flames"],            # 11th  — confirmed (0 standings pts)
        ],
    },
    {
        "name": "MLP Austin",
        "date": "2026-06-14",
        # Group A (5): CA Black Bears, Carolina Hogs, Columbus Sliders,
        #              Miami Pickleball Club, SoCal Hard Eights
        # Group B (6): Atlanta Bouncers, Bay Area Breakers, Dallas Flash,
        #              Florida Smash, NJ 5s, Texas Ranchers
        "placements": [
            ["New Jersey 5s"],            # 1st  — confirmed
            ["Columbus Sliders"],          # 2nd  — confirmed
            ["Texas Ranchers"],            # 3rd  — confirmed
            ["SoCal Hard Eights"],         # 4th  — confirmed
            ["Miami Pickleball Club"],     # 5th  — confirmed (beat Bay Area on Sunday)
            ["Bay Area Breakers"],         # 6th  — confirmed
            # 7th–10th: Atlanta, Carolina Hogs, CA Black Bears, Florida Smash (unknown order)
            [
                "Atlanta Bouncers",
                "Carolina Hogs",
                "California Black Bears",
                "Florida Smash",
            ],
            ["Dallas Flash"],              # 11th  — confirmed (0 standings pts)
        ],
    },
]


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def avg_dupr(team: str) -> float:
    """Return mean DUPR across all 6 roster spots."""
    return statistics.mean(dupr for _, dupr, _ in TEAMS[team])


def all_estimated(team: str) -> bool:
    """True if every DUPR on the roster is an estimate."""
    return all(est for _, _, est in TEAMS[team])


def any_confirmed(team: str) -> bool:
    return any(not est for _, _, est in TEAMS[team])


@dataclass
class EventSOR:
    event: str
    placement_tier: int          # 1 = 1st place tier
    beats: list[str]             # teams placed below this team
    loses_to: list[str]          # teams placed above this team
    strength_of_wins: float
    strength_of_losses: float
    contribution: float          # strength_of_wins - strength_of_losses


def compute_sor() -> pd.DataFrame:
    rows = []

    for team in TEAMS:
        event_sors: list[float] = []
        event_details: list[str] = []
        events_played = 0

        for event in EVENT_RESULTS:
            placements = event["placements"]

            # Find which tier this team is in (0-indexed).
            team_tier: Optional[int] = None
            for tier_idx, tier_teams in enumerate(placements):
                if team in tier_teams:
                    team_tier = tier_idx
                    break

            if team_tier is None:
                continue  # team didn't play this event

            events_played += 1

            # Teams at strictly higher placements (lower tier index) beat this team.
            loses_to = [t for idx in range(team_tier) for t in placements[idx]]

            # Teams at strictly lower placements (higher tier index) were beaten.
            beats = [t for idx in range(team_tier + 1, len(placements)) for t in placements[idx]]

            all_event_teams = [t for tier in placements for t in tier]
            max_field_dupr = max(avg_dupr(t) for t in all_event_teams)
            n_opponents = len(all_event_teams) - 1  # every team at event except self

            # Win vs O: +O.dupr (beating a stronger team is better)
            # Loss vs O: +(O.dupr - max_field_dupr)  ≤ 0
            #   → losing to the best team in the field costs ~0; losing to a weak team
            #     costs more. This is the key fix: strong beaters ≠ bigger penalty.
            win_contributions = [avg_dupr(t) for t in beats]
            loss_contributions = [avg_dupr(t) - max_field_dupr for t in loses_to]

            total = sum(win_contributions) + sum(loss_contributions)
            contribution = total / n_opponents if n_opponents > 0 else 0.0
            event_sors.append(contribution)

            n_wins = len(beats)
            n_losses = len(loses_to)
            event_details.append(
                f"{event['name']}: tier={team_tier+1}, "
                f"W={n_wins} L={n_losses}, "
                f"max_field={max_field_dupr:.3f}, "
                f"contrib={contribution:+.3f}"
            )

        team_sor = statistics.mean(event_sors) if event_sors else float("nan")
        team_avg_dupr = avg_dupr(team)
        has_confirmed = any_confirmed(team)

        rows.append(
            {
                "Team": team,
                "Avg DUPR": round(team_avg_dupr, 3),
                "Events Played": events_played,
                "SOR": round(team_sor, 3),
                "Has Confirmed DUPR": has_confirmed,
                "_event_details": event_details,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("SOR", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank
    return df


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_results(df: pd.DataFrame, verbose: bool = False) -> None:
    print("\n" + "=" * 70)
    print("  MLP 2026 STRENGTH OF RECORD — through Austin (Event 4 of 9)")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Team':<30} {'Avg DUPR':>9} {'Events':>7} {'SOR':>8}  {'Data'}")
    print("-" * 70)

    for rank, row in df.iterrows():
        data_flag = "" if row["Has Confirmed DUPR"] else "(all est.)"
        print(
            f"{rank:<5} {row['Team']:<30} {row['Avg DUPR']:>9.3f} "
            f"{row['Events Played']:>7} {row['SOR']:>+8.3f}  {data_flag}"
        )

    if verbose:
        print("\n" + "=" * 70)
        print("  PER-EVENT BREAKDOWN")
        print("=" * 70)
        for rank, row in df.iterrows():
            print(f"\n{rank}. {row['Team']}")
            for detail in row["_event_details"]:
                print(f"   {detail}")

    print(
        "\nSOR formula: per-event contribution = (Σ win_opponent_DUPR + Σ(loss_opponent_DUPR − max_field_DUPR)) / (n_teams_at_event − 1)"
    )
    print("Losses to the strongest team in the field cost ~0; losses to weak teams cost more.")
    print("Positive SOR = record stronger than field quality alone would predict.")
    print("⚠  DUPR values marked (est) were estimated; treat exact numbers as directional.")

    print("\nDUPR data notes:")
    print("  Confirmed: Ben Johns 7.12, JW Johnson 7.17, Andrei Daescu 6.95,")
    print("             Hayden Patriquin 6.90, Gabriel Tardio 6.88, Riley Newman 6.81,")
    print("             CJ Klinger 6.70, Will Howells 6.64, Anna Leigh Waters 6.56,")
    print("             Noe Khlif 6.49, Anna Bright 6.38, Catherine Parenteau 6.16,")
    print("             Parris Todd 6.00, Jade Kawamoto 5.99, Kate Fahey 5.83.")
    print("  Event placement notes:")
    print("    Dallas: positions 3-4 are estimated (NJ 5s 3rd, St. Louis 4th).")
    print("    Columbus: positions 7-11 are grouped as tied.")
    print("    St. Louis: positions 8-9 (Bay Area / Atlanta) are tied; 10th = SoCal.")
    print("    Austin: positions 7-10 are grouped as tied.")


def print_roster_table() -> None:
    print("\n" + "=" * 70)
    print("  TEAM ROSTERS & DUPR (confirmed / estimated)")
    print("=" * 70)
    for team, players in TEAMS.items():
        team_avg = avg_dupr(team)
        print(f"\n{team} (avg DUPR: {team_avg:.3f})")
        for name, dupr, est in players:
            flag = " *" if est else "  "
            print(f"  {flag} {name:<28} {dupr:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    rosters = "--rosters" in sys.argv or "-r" in sys.argv

    df = compute_sor()

    if rosters:
        print_roster_table()

    print_results(df, verbose=verbose)
