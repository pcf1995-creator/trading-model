# Claude Code Guidelines

## Session Log (Read This First)

**At the start of every session, read `SESSION_LOG.md` in this repo.** It contains open bugs, failed fix attempts, and in-progress context from previous sessions. Treat it as ground truth — do not ask the user to re-explain issues already logged there.

**At the end of every session** (before the final commit/push), update `SESSION_LOG.md`:
- Add any new errors encountered, with exact messages and file:line references
- Record what was tried and why it failed, so the next session doesn't repeat it
- Move fixed issues to the Resolved section with the commit hash
- Commit and push: `git add SESSION_LOG.md && git commit -m "chore: update session log" && git push origin <branch>`

## Commits & Push to GitHub

**CRITICAL: Always commit AND push code changes to GitHub.** Render auto-deploys from GitHub — commits stay local until pushed.

1. Make the change
2. Test it (refresh browser, check output)
3. **Commit** with clear message: `git commit -m "..."`
4. **Push to GitHub**: `git push origin main`
5. Wait for Render to redeploy (~1-2 min)
6. Only then consider the task complete

This is essential for:
- App modifications (Streamlit dashboard changes)
- Any changes to Python files that are actively running
- Bug fixes and feature updates
- Feature flags and cloud-first state changes

If you commit without pushing, Render won't see the changes and the user won't see them on the deployed app.

## Architecture: Cloud-First, Not Local

**Always design for cloud deployment (Render), not local development.** This project is automated trading that runs 24/7 on Render. Default to:
- **Cloud state (Supabase)** for any shared configuration, flags, or state
- **Environment variables (Render)** for secrets and configuration
- **NOT local JSON files** (feature_flags.json, etc) — local files only exist on your machine, Render doesn't have them

**Why:** Local JSON files create sync issues:
- User toggles flag locally → saves to local feature_flags.json
- Render has its own copy (empty or stale)
- Cron runs on Render, reads stale flags, nothing happens
- Hours wasted debugging

**How to apply:** When adding a feature that needs state/config:
1. Store it in Supabase (preferred), not local files
2. Or store it as Render environment variables
3. Not in local JSON unless it's dev-only and explicitly ignored
