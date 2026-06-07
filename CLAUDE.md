# Claude Code Guidelines

## ⚠️ CRITICAL: No Hidden Infrastructure

**Claude must NEVER set up external services, cron jobs, Render services, scheduled tasks, or any automated processes without the user explicitly seeing and approving them in their own account.**

Previous sessions created Render services the user could not find or control, which resulted in live orders being placed invisibly. This must never happen again.

**Rules:**
- Do NOT create Render services, workers, or cron jobs
- Do NOT set up cron-job.org, GitHub Actions, or any other scheduler
- Do NOT configure any external service that fires automatically
- Push code to GitHub only — the user decides how and where to deploy it
- If deployment setup is needed, walk the user through doing it themselves in their own account

---

## Deployment — Current Status

**Where the app runs:** _(to be confirmed by user — do not assume Render)_

**How code gets deployed:** Push to GitHub main branch. The user then deploys manually or via whatever hosting they control and can see.

**If you need to add infrastructure** (e.g. a new cron job): describe what's needed, provide the exact commands/config, and let the user set it up in their account. Never do it silently.

---

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

### Two Separate Cron Jobs (IMPORTANT — do not merge back into one)

The scan and stop-loss check are intentionally split into two endpoints with different cadences:

| Job | Endpoint | Interval | What it does |
|---|---|---|---|
| Weekly scan | `GET /scan` | Every 60 min | Heavy: runs `kalshi_crypto_weekly.py`, places orders, runs one stop-loss pass |
| Stop-loss only | `GET /check-stops` | Every 15 min | Lightweight: only `check_positions()`, no subprocess, trivial memory |

**Why split?** The weekly scorer (`kalshi_crypto_weekly.py`) fetches all Kalshi markets and runs Black-Scholes — enough memory to cause 503s on Render's free tier at high frequency. The stop-loss check only fetches prices for 2-5 held positions and is safe to run every 15 min. **Never merge them back into one job.**

### Scan Cron (scan_cron.py)
- Deployed as a **Render Web Service** (main branch, `Procfile: web: python kalshi/scan_cron.py`)
- Triggered by **cron-job.org** hitting the endpoints above
- Runs **weekly-only scan** (`kalshi_crypto_weekly.py --auto-save-db`) — only scores >24h contracts
- After scan: runs `place_scheduled_orders()` for auto-placement of weekly trades only
- After placement: runs one `check_positions()` pass for stop-loss enforcement

### Stop-Loss Rules (monitor.py) — Per-Bucket, Computed Dynamically

Stop thresholds and exempt windows are determined at **check time** from `hours_left`, not stored at entry time.

| Bucket | Trigger | Exempt window | Rationale |
|---|---|---|---|
| Weekly (>24 hr) | Price drops 50% from entry | Last 60 min | Short dips before weekly settlement are common |
| Intraday (1–24 hr) | Price drops 70% from entry (price ≤ 30% of entry) | Last 15 min | Intraday is more volatile; keep stop active longer |
| Vol model (<1 hr) | Never — always exempt | Entire duration | Contracts settle naturally; bad fill risk exceeds protection |

**DO NOT** collapse these into a single `STOP_LOSS_EXEMPT_MINUTES` constant. A universal 60-min window would also exempt intraday contracts with 45 min left when they're collapsing — exactly the scenario the stop-loss is designed to catch.

**Auto-placement feature flags** are in Supabase (`feature_flags` table). If nothing is being placed after a scan, check that `auto_place_weekly` is `True` in the Streamlit dashboard → Auto-Placement section. Default is `False`.

### Vol Model (contracts < 1 hour to expiry)
- Uses Black-Scholes log-normal formula with Binance 1m realized vol
- Stop-loss is always skipped — vol contracts are <1 hr and settle naturally
- Vol bucket budget: $100, max single bet $25 (25% Kelly cap)

### What NOT to do
- Do NOT add local Mac cron jobs for any trading automation — use Render + scan_cron.py
- Do NOT run the weekly scan every 5–15 minutes — memory constraints on Render free tier require 60 min intervals
- Do NOT score intraday (<24h) contracts in the automated scan — weekly only to stay within memory limits
- Do NOT merge `/scan` and `/check-stops` back into one job — they have different memory profiles
- Do NOT use a single exempt window constant for all buckets — weekly and intraday need different values

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
