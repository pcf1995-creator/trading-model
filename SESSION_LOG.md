# Session Log — Trading Model

This file is the cross-session memory for Claude Code. Read it at the start of every session. Update it at the end of every session before committing and pushing.

---

## How to Use This File

**At session start:** Read this file fully. Treat "Open Issues" as your starting context — don't ask the user to re-explain problems already listed here.

**During a session:** When you hit an error or a fix attempt fails, append it under the relevant issue (or open a new one). Be specific: include the error message, the file and line number, what was tried, and why it failed.

**At session end:** Before committing, update the status of any issue you touched. Move resolved issues to the "Resolved" section with a one-line summary of the fix. Push to GitHub.

---

## Open Issues

### [ISSUE-1] Performance tab — Kalshi backfill not showing actual settled bets
- **First seen:** ~2026-04-28
- **File(s):** `app.py:3044`, `app.py:3099`, `app.py:3119`, `app.py:3122`
- **Symptom:** After clicking "📥 Backfill Mar 1+", the Performance tab still shows no closed positions (or wrong data). ~15 commits across 36+ hours have not fully resolved it.
- **What the code does:** Fetches fills from Kalshi `/portfolio/fills` API since March 1, 2026 → groups by ticker+side → identifies "closed" positions → upserts to `kalshi_trades` Supabase table → `load_kalshi_trades()` reads them for the Performance tab.
- **Attempts so far:**
  - Fixed price field to use `yes_price_dollars` vs `no_price_dollars` per side (commit 2846636)
  - Fixed tracking YES and NO sides separately (commits f52fba0, 8ddc476)
  - Fixed Unix timestamp → ISO for `settled_at` (commit 518715f)
  - Added auto-settled expired position handling (commit 22d6423)
  - Fixed timezone-aware expiry comparison (commit 4003860)
  - Added extensive DEBUG output (commits 7bce4b7, b3ea479, ee86db4)
  - Fixed `clear_kalshi_trades` delete query (commits ac7b1a0, 47ea243)
- **Suspected remaining bugs (code analysis, not yet tested):**
  1. **Price field prefix mismatch** (`app.py:3044-3045`, `app.py:3099`): Backfill calls `_price_dollars(f, "yes_price_dollars")` but the function tries suffixes `_dollars`, `_fixed`, `""` — so it looks for `"yes_price_dollars_dollars"` first (doesn't exist), then `"yes_price_dollars"` last. The rest of the code calls `_price_dollars(f, "yes_price")` which correctly tries `"yes_price_dollars"` as the first attempt. If the Kalshi API returns only `"yes_price"` (in cents) and not `"yes_price_dollars"`, the backfill filter at line 3044 would return 0.0 for all fills, filtering them all out.
  2. **Partial-close + expiry gap** (`app.py:3119-3122`): `_is_closed` requires `bought == sold` exactly. `_is_auto_settled` requires `sold == 0`. A position that was partially sold then expired (bought=5, sold=3) satisfies neither condition and is silently dropped.
- **Status:** Fixed in commit 89c8d86 — see Resolved section

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
- **Commit:** 89c8d86
- **Note:** DEBUG output is still in place in `app.py` — remove `st.write("**DEBUG:**...")` lines once confirmed working.

<!--
TEMPLATE:

### [ISSUE-N] Short description
- **Resolved:** YYYY-MM-DD
- **Fix:** One sentence describing what the actual problem was and how it was fixed.
- **Commit:** abc1234
-->
