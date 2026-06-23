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
| 1 | Los Angeles Mad Drops | 6.180 | 2 | **+5.74** |
| 2 | New Jersey 5s | 6.343 | 3 | **+5.59** |
| 3 | St. Louis Shock | 6.212 | 3 | **+5.19** |
| 4 | Columbus Sliders | 6.177 | 3 | **+5.18** |
| 5 | Brooklyn Pickleball Team † | 6.274 | 1 | **+4.78** |
| 6 | Palm Beach Royals | 6.030 | 2 | **+3.84** |
| 7 | Las Vegas Night Owls ‡ | 5.813 | 2 | **+3.23** |
| 8 | Texas Ranchers | 5.929 | 2 | **+2.35** |
| 9 | SoCal Hard Eights | 5.849 | 2 | **+2.28** |
| 10 | Orlando Squeeze | 6.150 | 2 | **+2.06** |
| 11 | Miami Pickleball Club ‡ | 6.014 | 2 | **+1.67** |
| 12 | Atlanta Bouncers ‡ | 6.149 | 3 | **+1.44** |
| 13 | Bay Area Breakers ‡ | 6.000 | 3 | **+1.28** |
| 14 | Utah Black Diamonds ‡ | 5.975 | 2 | **+1.13** |
| 15 | Florida Smash ‡ | 5.917 | 2 | **+0.15** |
| 16 | California Black Bears ‡ | 5.833 | 2 | **+0.15** |
| 17 | Carolina Hogs | 5.829 | 3 | **+0.09** |
| 18 | Phoenix Flames ‡ | 5.950 | 2 | **−0.13** |
| 19 | Chicago Slice ‡ | 5.999 | 1 | **−0.13** |
| 20 | Dallas Flash | 6.090 | 2 | **−0.19** |

**† Small sample** (1 event played — treat with caution)
**‡ All DUPR values estimated** — see below

---

## Where Estimates Are Made

### Player DUPR Ratings
DUPR.com blocks direct access, so values were sourced from PickleWave profiles, Electrum Pickleball athlete pages, and continental rankings. We now have **confirmed DUPR values for ~55+ players**, including full rosters for NJ 5s, St. Louis Shock, LA Mad Drops, Brooklyn, and Texas Ranchers.

**Teams with fully confirmed rosters** (no estimates): NJ 5s, St. Louis Shock, LA Mad Drops, Brooklyn Pickleball Team, Texas Ranchers.

**Teams with mostly confirmed rosters** (1–2 players estimated): Columbus Sliders, Dallas Flash, SoCal Hard Eights, Carolina Hogs, Atlanta Bouncers, Chicago Slice, Las Vegas Night Owls, Orlando Squeeze, Miami, Palm Beach Royals, Utah Black Diamonds.

**Teams with all-estimated rosters** (marked ‡): Bay Area Breakers, California Black Bears, Florida Smash, Phoenix Flames.

Two values to note: Elsie Hendershot's DUPR (5.124) is confirmed from Sep 2025 and may have since risen; JW Johnson's value was updated to 7.021 (May 2026 source) from a previously cited 7.17.

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
