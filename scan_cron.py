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
            "kalshi_crypto_weekly.py",
            "--auto-save-db",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=os.path.dirname(__file__),
        )

        # Log output
        logger.info(f"Scan subprocess exit code: {result.returncode}")
        if result.returncode != 0:
            logger.error(f"Scan failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr[:500]}")

        with open(SCAN_LOG_FILE, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Scan at {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Exit code: {result.returncode}\n")
            if result.stdout:
                f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")

        # Auto-placement via cron enabled (race condition fixes in place)
        logger.info("Auto-placement enabled. Calling place_scheduled_orders()...")
        try:
            from kalshi_crypto import place_scheduled_orders
            from kalshi_api import KalshiClient
            kalshi_client = KalshiClient()
            stats = place_scheduled_orders(kalshi_client)
            logger.info(f"Auto-placement results: {stats}")
        except Exception as e:
            logger.error(f"Auto-placement failed: {e}")

        logger.info("Background scan completed")

    except subprocess.TimeoutExpired:
        logger.error("Scan timed out after 15 minutes")
        with open(SCAN_LOG_FILE, "a") as f:
            f.write("Scan timed out after 15 minutes\n")
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
