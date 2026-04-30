# Claude Code Guidelines

## Session Log (Read This First)

**At the start of every session, read `SESSION_LOG.md` in this repo.** It contains open bugs, failed fix attempts, and in-progress context from previous sessions. Treat it as ground truth — do not ask the user to re-explain issues already logged there.

**At the end of every session** (before the final commit/push), update `SESSION_LOG.md`:
- Add any new errors encountered, with exact messages and file:line references
- Record what was tried and why it failed, so the next session doesn't repeat it
- Move fixed issues to the Resolved section with the commit hash
- Commit and push: `git add SESSION_LOG.md && git commit -m "chore: update session log" && git push origin <branch>`

## Deployment Model — Two Separate Systems

This project has two distinct runtime environments. Do not conflate them.

### 1. Render — cron job only (`scan_cron.py`)
Render runs `scan_cron.py` as a web process (see `Procfile: web: python scan_cron.py`). It auto-deploys from GitHub on every push to `main`.

- Changes to `scan_cron.py`, `predict.py`, `features.py`, `db.py`, `kalshi_api.py`, or anything the cron imports → **push to `main` and wait ~1-2 min for Render to redeploy**.
- Render does NOT run `app.py`.

### 2. Local machine — Streamlit dashboard (`app.py`)
The Streamlit dashboard runs locally. Changes to `app.py` take effect when the user pulls and restarts Streamlit — Render is not involved.

- Changes to `app.py` → **push to GitHub, then the user pulls locally and restarts Streamlit**.
- Do NOT tell the user to "wait for Render to redeploy" for `app.py` changes.

### Commits & Push — Always Required
Regardless of which system is affected, always commit and push:
1. Make the change
2. `git commit -m "..."`
3. `git push origin main`
4. For cron changes: wait for Render (~1-2 min). For dashboard changes: user pulls locally.

## Architecture: Cloud-First State

State and configuration must live in the cloud, not local files — the cron runs on Render and cannot read files that only exist on the user's machine.

- **Shared config / flags** → Supabase
- **Secrets** → Render environment variables
- **NOT local JSON files** (e.g. `feature_flags.json`) — Render has its own copy, changes made locally never reach the cron

**Why this matters:** User toggles a flag locally → saves to local JSON → Render cron reads its own stale copy → nothing happens → hours wasted debugging.
