# Claude Code Guidelines

## Commits

**Always commit code changes before considering work complete.** This is especially important for:
- App modifications (Streamlit dashboard changes must be committed for server to pick them up)
- Any changes to Python files that are actively running
- Bug fixes and feature updates

When making code changes, commit immediately after verifying they work, rather than leaving them uncommitted.

## Code Changes

When editing `app.py` or other active files:
1. Make the change
2. Test it (refresh browser, check output)
3. **Commit the change** with a clear message
4. Only then consider the task complete

If you make changes and don't commit them, the changes may not be picked up by the running server or may be lost.

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
