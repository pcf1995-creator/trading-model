# Claude Code Guidelines

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

## Kalshi Scan & Stop-Loss Architecture

### Scan Cron (scan_cron.py)
- Deployed as a **Render Web Service** (main branch, `Procfile: web: python kalshi/scan_cron.py`)
- Triggered by **cron-job.org** hitting the `/scan` endpoint
- **Run interval: 30–60 minutes** (NOT every 5 minutes — caused memory/503 errors on Render free tier)
- Runs **weekly-only scan** (`kalshi_crypto_weekly.py --auto-save-db`) — only scores >24h contracts
- After scan: runs `place_scheduled_orders()` for auto-placement of weekly trades only
- After placement: runs `check_positions()` from monitor.py for stop-loss enforcement

### Stop-Loss Monitor (monitor.py)
- `STOP_LOSS_PCT = 0.50` — triggers when price drops 50% from entry
- `STOP_LOSS_EXEMPT_MINUTES = 15` — vol model contracts within 15 min of expiry are skipped
- **No longer runs from local Mac crontab** — those were removed. Runs exclusively via scan_cron.py on Render
- The old local Mac cron jobs (3min/10min/30min) have been permanently deleted

### Vol Model (contracts < 1 hour to expiry)
- Uses Black-Scholes log-normal formula with Binance 1m realized vol
- Stop-loss is SKIPPED for vol model contracts within 15 minutes of expiry (they settle naturally)
- Vol bucket budget: $100, max single bet $25 (25% Kelly cap)

### What NOT to do
- Do NOT add local Mac cron jobs for any trading automation — use Render + scan_cron.py
- Do NOT run the scan every 5 minutes — memory constraints on Render free tier require 30–60 min intervals
- Do NOT score intraday (<24h) contracts in the automated scan — weekly only to stay within memory limits

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
