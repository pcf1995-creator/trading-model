# MLP 2026 Strength of Record — Through Event 4 (Austin)
*As of June 14, 2026 · Internal analysis · Not official MLP data*

---

## What Is Strength of Record?

Strength of Record (SOR) measures how *impressive* a team's results have been relative to the quality of the opponents involved. It answers a question that raw standings don't: **did you beat good teams, and did you only lose to good teams?**

**Key insight:** Two teams can have the same event placement but very different SOR. Finishing 2nd by losing only to the highest-rated team in the field is more impressive than finishing 2nd by barely surviving a weak field.

---

## How It's Calculated

Each team is assigned an average roster quality score using **DUPR** (Dynamic Universal Pickleball Rating), the official MLP player rating system. Doubles DUPR runs roughly 5.0 (recreational) to 7.5+ (world-class pro). The six-player roster average is used as each team's quality proxy.

For each event a team plays, we compute a **per-event SOR contribution:**

> **SOR contribution = (sum of beaten teams' avg DUPR + sum of loss penalties) ÷ (field size − 1)**

Where the loss penalty for each team that beat you is:
> **(that team's avg DUPR) − (highest avg DUPR in the event field)**

This means:
- **Beating a strong team** contributes more than beating a weak team.
- **Losing to the best team in the field** costs nearly zero.
- **Losing to a weak team** costs significantly more.

A team's final SOR is the average of its per-event contributions across all events played.

**Placement tiers:** MLP does not publish full head-to-head results for every match. Where exact sub-placings are unknown, teams are grouped into a "tied tier" and contribute zero SOR credit to each other. This is conservative — it avoids fabricating wins or losses.

---

## Current Standings

| Rank | Team | Avg DUPR | Events Played | SOR |
|------|------|:--------:|:-------------:|:---:|
| 1 | Los Angeles Mad Drops | 6.270 | 2 | **+5.77** |
| 2 | New Jersey 5s | 6.382 | 3 | **+5.64** |
| 3 | Columbus Sliders | 6.250 | 3 | **+5.23** |
| 4 | St. Louis Shock | 6.348 | 3 | **+5.22** |
| 5 | Brooklyn Pickleball Team † | 6.160 | 1 | **+4.79** |
| 6 | Palm Beach Royals ‡ | 5.950 | 2 | **+3.85** |
| 7 | Las Vegas Night Owls ‡ | 5.858 | 2 | **+3.23** |
| 8 | Texas Ranchers ‡ | 6.058 | 2 | **+2.39** |
| 9 | SoCal Hard Eights ‡ | 6.050 | 2 | **+2.26** |
| 10 | Orlando Squeeze ‡ | 6.050 | 2 | **+2.07** |
| 11 | Miami Pickleball Club ‡ | 6.020 | 2 | **+1.70** |
| 12 | Atlanta Bouncers ‡ | 6.142 | 3 | **+1.45** |
| 13 | Bay Area Breakers ‡ | 6.000 | 3 | **+1.30** |
| 14 | Utah Black Diamonds ‡ | 5.950 | 2 | **+1.12** |
| 15 | Florida Smash ‡ | 5.917 | 2 | **+0.16** |
| 16 | California Black Bears ‡ | 5.833 | 2 | **+0.16** |
| 17 | Carolina Hogs ‡ | 6.050 | 3 | **+0.10** |
| 18 | Chicago Slice ‡ | 6.042 | 1 | **−0.14** |
| 19 | Phoenix Flames ‡ | 5.950 | 2 | **−0.15** |
| 20 | Dallas Flash | 6.178 | 2 | **−0.17** |

**† Small sample** (1 event played — treat with caution)
**‡ All DUPR values estimated** — see below

---

## Where Estimates Are Made

### Player DUPR Ratings
DUPR.com and affiliated sites block automated data access. We have **confirmed DUPR values for 15 players:**

Ben Johns (7.12), JW Johnson (7.17), Andrei Daescu (6.95), Hayden Patriquin (6.90), Gabriel Tardio (6.88), Riley Newman (6.81), CJ Klinger (6.70), Will Howells (6.64), Anna Leigh Waters (6.56), Noe Khlif (6.49), Anna Bright (6.38), Catherine Parenteau (6.16), Parris Todd (6.00), Jade Kawamoto (5.99), Kate Fahey (5.83).

All other player DUPRs (~57 players across 20 teams) are **estimated** based on draft position, known pro rankings, and comparable players. Teams with at least one confirmed DUPR are: LA Mad Drops, NJ 5s, St. Louis Shock, Columbus Sliders, Brooklyn, and Dallas Flash. The remaining 14 teams are marked ‡ above and their relative rankings should be treated as directional only.

### Event Placements
MLP publishes final standings but not always the detailed sub-placings within the lower half of the field. The following positions are grouped as "tied" in our model:

| Event | Tied Positions | Teams Affected |
|-------|---------------|----------------|
| Dallas | 5th–11th | Bay Area, Carolina Hogs, Dallas Flash, Orlando, Phoenix, Texas Ranchers, Utah |
| Columbus | 7th–11th | CA Black Bears, Chicago Slice, Miami, Carolina Hogs, Florida Smash |
| St. Louis | 8th–9th | Bay Area Breakers & Atlanta Bouncers (order unknown) |
| Austin | 7th–10th | Atlanta Bouncers, Carolina Hogs, CA Black Bears, Florida Smash |

---

## What We'd Need to Make This Fully Accurate

1. **Confirmed DUPR for all roster players** — the single biggest driver of accuracy. With ~57 estimated DUPRs, the DUPR quality proxy can be off by 0.2–0.5 points for many teams, which cascades into SOR errors. Direct DUPR.com access (or manual lookup) for all ~120 players on 20 rosters would resolve this.

2. **Exact placements for all tied-tier positions** — particularly the Dallas 5–11 bucket (7 teams lumped together). Getting the actual 5th–11th place order for Dallas would unlock meaningful SOR signal for roughly half the league.

3. **Pool play results within each event** — currently we use only final Super Sunday standings. If pool play records were available (who beat whom during the group stage), we could weight head-to-head results directly rather than inferring from final placement tiers.

4. **Trade/roster confirmation for several teams** — Columbus's acquired player (return for Danni-Elle Townsend), Miami's full roster beyond Dylan Frazier, and several teams listed with placeholder "Player 2/3/etc." entries.

---

*Analysis produced using `pickleball/mlp_sor.py`. Events: Dallas (5/25), Columbus (5/31), St. Louis (6/7), Austin (6/14). Five events remain in the 2026 regular season.*
