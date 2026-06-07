# Claude Code Guidelines

## TWO SEPARATE APPS — Do Not Mix

This repo contains **two completely independent applications**. Always be clear which one you're working on.

| App | What it is | Where it runs | Entry point |
|---|---|---|---|
| **Stocks app** | S&P 500 L/S conviction model, paper/real trade tracking, overnight drift scanner | **Streamlit Cloud** (share.streamlit.io) | `stocks/app.py` |
| **Kalshi app** | Kalshi prediction market trading, cron-based scanning and order placement | Separate infrastructure (do not touch without explicit instruction) | `kalshi/scan_cron.py` |

---

## Stocks App — Deployment

- **Platform: Streamlit Cloud** — NOT Render, NOT local
- Auto-deploys from GitHub `main` branch (~1-2 min after push)
- To access: log into share.streamlit.io with GitHub account `pcf1995-creator`
- Live URL: _(fill in from Streamlit Cloud dashboard)_
- **When code is pushed to main → Streamlit Cloud redeploys automatically**
  - Do NOT say "Render will redeploy" — it is Streamlit Cloud, not Render

---

## Commits & Push to GitHub

**CRITICAL: Always commit AND push code changes to GitHub.** Streamlit Cloud auto-deploys from GitHub — commits stay local until pushed.

1. Make the change
2. **Commit** with clear message: `git commit -m "..."`
3. **Push to GitHub**: `git push origin main`
4. Wait for Streamlit Cloud to redeploy (~1-2 min)
5. Only then consider the task complete

---

## ⚠️ No Hidden Infrastructure

**Claude must NEVER set up external services, cron jobs, scheduled tasks, or automated processes without the user explicitly doing it themselves in their own account.**

A previous session created services the user could not find or control, resulting in live orders placed invisibly. This must never happen again.

- Do NOT create Render services, workers, or cron jobs
- Do NOT set up cron-job.org, GitHub Actions, or any other scheduler
- Push code to GitHub only — the user controls all deployment and automation
- If infrastructure is needed, describe exactly what's needed and let the user set it up

---

## Kalshi App — Separate, Do Not Touch Unless Asked

The Kalshi app is a completely separate trading system for Kalshi prediction markets. It has its own scan/stop-loss architecture documented in `SCAN_CRON_SETUP.md`. Do not modify or deploy anything related to Kalshi unless explicitly asked.

### Stop-Loss Rules (monitor.py) — Per-Bucket, Computed Dynamically

| Bucket | Trigger | Exempt window |
|---|---|---|
| Weekly (>24 hr) | Price drops 50% from entry | Last 60 min |
| Intraday (1–24 hr) | Price drops 70% from entry | Last 15 min |
| Vol model (<1 hr) | Never exempt | Entire duration |

**DO NOT** collapse these into a single constant.

### What NOT to do (Kalshi)
- Do NOT add local Mac cron jobs for any trading automation
- Do NOT run the weekly scan every 5–15 minutes (memory constraints)
- Do NOT score intraday (<24h) contracts in the automated scan
- Do NOT merge `/scan` and `/check-stops` back into one job

---

## Architecture: Cloud-First, Not Local

**Always design for cloud deployment, not local development.**

- **Cloud state (Supabase)** for any shared configuration, flags, or state
- **NOT local JSON files** — local files only exist on the dev machine, not in the cloud
- Local JSON files create silent sync failures that are hard to debug
