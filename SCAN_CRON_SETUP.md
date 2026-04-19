# Setting Up 5-Minute Kalshi Scan Cron

## Overview

This guide sets up automated scans every 5 minutes via a free external cron service + Flask endpoint.

**Architecture**:
1. Flask app (`scan_cron.py`) with POST `/scan` endpoint
2. Deploy to free cloud platform (Render, Railway, or similar)
3. cron-job.org (free) POSTs to the endpoint every 5 minutes
4. Each POST triggers `kalshi_crypto.py --scan --auto-save-db --skip-bucket intraday_short`
5. Results saved to Supabase

---

## Step 1: Deploy Flask App to Render (Free)

### Create Render account
1. Go to https://render.com
2. Sign up (free tier available)

### Create new Web Service
1. Click "New Web Service"
2. Connect your GitHub repo (or upload the project)
3. Configure:
   - **Name**: `kalshi-scan-cron`
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn scan_cron:app`
   - **Free tier** (optional plan)
4. Deploy

### Get your endpoint URL
After deployment, you'll have a URL like: `https://kalshi-scan-cron.onrender.com`

Test it:
```bash
curl -X POST https://kalshi-scan-cron.onrender.com/scan
curl https://kalshi-scan-cron.onrender.com/health
```

---

## Step 2: Set Up cron-job.org (Free)

1. Go to https://cron-job.org
2. Sign up (free account)
3. Create new cron job:
   - **Title**: `Kalshi 5-min scan`
   - **URL**: `https://kalshi-scan-cron.onrender.com/scan`
   - **Execution**: Every 5 minutes
   - **Request method**: POST
4. Save

---

## Step 3: Monitor Scans

Check scan logs on Render:
- Go to your Web Service
- View "Logs" to see scan results

Or query Supabase directly to verify paper trades are being saved:
```python
import db
trades = db.load_paper_trades()
print(f"Total trades: {len(trades)}")
print(f"Latest: {trades[-1] if trades else 'None'}")
```

---

## Alternative: Deploy to Railway

1. Go to https://railway.app
2. Create new project
3. Deploy from GitHub or upload project
4. Set start command: `gunicorn scan_cron:app`
5. Get URL and use in cron-job.org

---

## Troubleshooting

**"Scan failed" errors**:
- Check Render logs for error details
- Verify `KALSHI_KEY_ID` and `KALSHI_KEY_PATH` are set as environment variables
- Ensure `db.py` and `kalshi_crypto.py` can import Supabase credentials

**"Timeout" errors**:
- Scan takes >5 minutes
- Increase timeout in `scan_cron.py` line 45 or reduce number of contracts to score

**No trades saving to DB**:
- Check that `--auto-save-db` flag is being passed in `scan_cron.py` line 30
- Verify Supabase connection in `db.py`
- Check `kalshi-scan-cron.log` on Render for details

---

## Local Testing

Test locally before deploying:
```bash
python kalshi_crypto.py --auto-save-db --skip-bucket intraday_short
```

Or test the Flask endpoint:
```bash
python scan_cron.py
curl -X POST http://localhost:5000/scan
```
