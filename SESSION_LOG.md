# Session Log — Trading Model

This file is the cross-session memory for Claude Code. Read it at the start of every session. Update it at the end of every session before committing and pushing.

---

## How to Use This File

**At session start:** Read this file fully. Treat "Open Issues" as your starting context — don't ask the user to re-explain problems already listed here.

**During a session:** When you hit an error or a fix attempt fails, append it under the relevant issue (or open a new one). Be specific: include the error message, the file and line number, what was tried, and why it failed.

**At session end:** Before committing, update the status of any issue you touched. Move resolved issues to the "Resolved" section with a one-line summary of the fix. Push to GitHub.

---

## Open Issues

### [ISSUE-2] Performance tab — P&L wrong for NO positions; YES P&L shows flat
- **First seen:** 2026-04-30
- **File(s):** `app.py:3044-3046`, `app.py:3099-3103`
- **Symptom:** Table shows -$23.75 for an ETH NO position that had a partial sell of ~$20 before expiry. YES side shows $0 P&L despite having winning bets. P&L numbers don't match Kalshi order history.
- **Root cause:** Kalshi fills only include `yes_price` — there is no `no_price` field. Backfill was calling `_price_dollars(fill, "no_price")` which returned 0 when the field was absent, causing all NO sell proceeds to be $0 (every closed NO position looked like it expired worthless). The `_no_price_dollars()` helper that computes `1 - yes_price` existed but wasn't used in the backfill path.
- **Fix applied:** commit 9c0695d — use `_no_price_dollars()` for NO-side price extraction in both the fill validity filter and the per-fill price calculation.
- **Next step:** User must re-run "📥 Backfill Mar 1+" to refresh the Supabase data with corrected P&Ls. After that, verify YES P&L is no longer flat.
- **Status:** Code fixed, awaiting user to re-run backfill to confirm

<!--
TEMPLATE — copy and fill in for each new issue:

### [ISSUE-N] Short description
- **First seen:** YYYY-MM-DD
- **File(s):** path/to/file.py:line
- **Symptom:** What the user observes
- **Error:** Exact error message or traceback
- **Attempts so far:**
  - YYYY-MM-DD: What was tried → why it didn't work
- **Current theory:** Best guess at root cause
- **Status:** Open / Blocked / In-progress
-->

---

## Resolved Issues

### [ISSUE-1] Performance tab — Kalshi backfill not showing actual settled bets
- **Resolved:** 2026-04-30
- **Fix 1 (price filter):** Backfill called `_price_dollars(f, "yes_price_dollars")` as prefix, causing the function to try `"yes_price_dollars_dollars"` first (doesn't exist), only accidentally finding the right field as a fallback. Changed to `"yes_price"` / `"no_price"` (consistent with the rest of the codebase), so the function correctly tries `"yes_price_dollars"` first.
- **Fix 2 (partial close + expiry):** `_is_auto_settled` required `sold == 0`, silently dropping positions that were partially sold then expired. Now computes `remaining = bought - sold` and captures any expired position with `remaining > 0`, adding both manual sell proceeds and expiry settlement value to the total.
- **Fix 3 (NO price fallback):** Kalshi fills don't include `no_price`; backfill was calling `_price_dollars(fill, "no_price")` → returned 0 → all NO sell proceeds were $0. Changed to use `_no_price_dollars()` which derives `1 - yes_price`. Commit: 9c0695d.
- **Note:** DEBUG output is still in place in `app.py` — remove `st.write("**DEBUG:**...")` lines once confirmed working.

<!--
TEMPLATE:

### [ISSUE-N] Short description
- **Resolved:** YYYY-MM-DD
- **Fix:** One sentence describing what the actual problem was and how it was fixed.
- **Commit:** abc1234
-->
