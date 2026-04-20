"""
scan_cron.py — Flask endpoint for automated 5-minute Kalshi scans

Deploy to: Render, Railway, or another free cloud platform
Trigger via: cron-job.org or easycron (POST /scan every 5 minutes)

Usage:
  python scan_cron.py                    # run locally on :5000
  gunicorn scan_cron:app                 # production (for Render/Railway)
"""

import os
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify
import subprocess
import sys
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCAN_LOG_FILE = "kalshi-scan-cron.log"


def _run_scan_background():
    """Run scan in background thread so HTTP response returns quickly."""
    try:
        logger.info("Background scan starting...")
        cmd = [
            sys.executable,
            "kalshi_crypto.py",
            "--scan",
            "--auto-save-db",
            "--skip-bucket",
            "intraday_short",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(__file__),
        )

        # Log output
        with open(SCAN_LOG_FILE, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Scan at {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Exit code: {result.returncode}\n")
            if result.stdout:
                f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")

        # Check if auto-placement is enabled and execute if so
        if result.returncode == 0:
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                import db
                from kalshi_api import KalshiClient
                from kalshi_crypto import place_scheduled_orders

                flags = db.load_feature_flags()
                if flags.get("auto_place_weekly"):
                    client = KalshiClient()
                    if not client.dry_run:
                        client.login()

                    # Get pending trades before placing
                    trades_before = db.load_paper_trades()
                    pending_ids = [t["id"] for t in trades_before
                                  if t.get("status") == "open" and t.get("bucket") == "weekly"]

                    stats = place_scheduled_orders(client)

                    # Mark newly-placed trades with auto_placed flag
                    if stats.get("weekly_placed", 0) > 0:
                        trades_after = db.load_paper_trades()
                        for t in trades_after:
                            if t["id"] not in pending_ids and t.get("bucket") == "weekly":
                                client.table("paper_trades").update({"auto_placed": True}).eq("id", t["id"]).execute()

                    with open(SCAN_LOG_FILE, "a") as f:
                        f.write(f"Auto-placement: {stats}\n")
                    logger.info(f"Auto-placed trades: {stats}")
            except Exception as e:
                logger.warning(f"Auto-placement failed: {e}")
                with open(SCAN_LOG_FILE, "a") as f:
                    f.write(f"Auto-placement error: {e}\n")

        logger.info("Background scan completed")

    except subprocess.TimeoutExpired:
        logger.error("Scan timed out after 5 minutes")
        with open(SCAN_LOG_FILE, "a") as f:
            f.write("Scan timed out after 5 minutes\n")
    except Exception as e:
        logger.error(f"Error in background scan: {e}")
        with open(SCAN_LOG_FILE, "a") as f:
            f.write(f"Error: {e}\n")


@app.route("/scan", methods=["POST"])
def trigger_scan():
    """Trigger a Kalshi scan asynchronously and return immediately."""
    try:
        logger.info("Cron request received, spawning background scan...")
        thread = threading.Thread(target=_run_scan_background, daemon=True)
        thread.start()
        return jsonify({"status": "accepted", "message": "Scan queued"}), 202
    except Exception as e:
        logger.error(f"Error queuing scan: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    logger.info("Starting Flask scan server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
