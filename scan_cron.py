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
# Log to both file and stdout so Render can capture it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("kalshi-scan-cron.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

SCAN_LOG_FILE = "kalshi-scan-cron.log"
LAST_SCAN_STATUS = {"timestamp": None, "status": None, "error": None}


def _run_scan_background():
    """Run scan in background thread so HTTP response returns quickly."""
    try:
        logger.info("Background scan starting...")
        logger.info(f"Using Python: {sys.executable}")
        logger.info(f"Working directory: {os.path.dirname(__file__)}")

        # Check if another scan is already running (prevent overlaps)
        import glob
        old_scans = glob.glob("kalshi-scan-cron.log.*")
        recent_scan = any(True for f in old_scans)  # Simple overlap check
        if recent_scan:
            logger.warning("Overlapping scan detected; skipping this execution")
            return

        # Run with unbuffered output to see logs in real-time
        cmd = [
            sys.executable,
            "-u",  # unbuffered
            "kalshi_crypto_weekly.py",
            "--auto-save-db",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        # Run subprocess with timeout - if it hangs, it will be killed
        logger.info(f"Starting subprocess: {' '.join(cmd)}")
        logger.info(f"Timeout: 600 seconds (10 min)")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=os.path.dirname(__file__),
            )
            logger.info(f"Subprocess completed with exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            logger.error("SUBPROCESS TIMEOUT: Scan did not complete within 10 minutes")
            raise

        # Log output (dump to stdout for Render visibility)
        logger.info(f"Scan subprocess exit code: {result.returncode}")
        logger.info(f"STDOUT length: {len(result.stdout) if result.stdout else 0} chars")
        logger.info(f"STDERR length: {len(result.stderr) if result.stderr else 0} chars")

        # Track status for health check
        LAST_SCAN_STATUS["timestamp"] = datetime.now(timezone.utc).isoformat()

        if result.returncode != 0:
            logger.error(f"Scan failed with exit code {result.returncode}")
            LAST_SCAN_STATUS["status"] = "failed"
            LAST_SCAN_STATUS["error"] = f"Exit code {result.returncode}"
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
                LAST_SCAN_STATUS["error"] = result.stderr[:500]
            if result.stdout:
                logger.error(f"STDOUT:\n{result.stdout}")
        else:
            LAST_SCAN_STATUS["status"] = "success"
            LAST_SCAN_STATUS["error"] = None

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
        LAST_SCAN_STATUS["timestamp"] = datetime.now(timezone.utc).isoformat()
        LAST_SCAN_STATUS["status"] = "timeout"
        LAST_SCAN_STATUS["error"] = "Scan timed out after 15 minutes"
        with open(SCAN_LOG_FILE, "a") as f:
            f.write("Scan timed out after 15 minutes\n")
    except Exception as e:
        logger.error(f"Error in background scan: {e}")
        LAST_SCAN_STATUS["timestamp"] = datetime.now(timezone.utc).isoformat()
        LAST_SCAN_STATUS["status"] = "error"
        LAST_SCAN_STATUS["error"] = str(e)
        with open(SCAN_LOG_FILE, "a") as f:
            f.write(f"Error: {e}\n")
    finally:
        # Force garbage collection to free memory
        import gc
        gc.collect()
        logger.info("Memory cleanup completed")


@app.route("/scan", methods=["GET", "POST"])
def trigger_scan():
    """Trigger a Kalshi scan asynchronously and return immediately."""
    try:
        LAST_SCAN_STATUS["timestamp"] = datetime.now(timezone.utc).isoformat()
        LAST_SCAN_STATUS["status"] = "running"
        LAST_SCAN_STATUS["error"] = None

        logger.info("="*70)
        logger.info("Cron request received, spawning background scan...")
        logger.info(f"Status: {LAST_SCAN_STATUS}")

        thread = threading.Thread(target=_run_scan_background, daemon=True)
        thread.start()

        logger.info(f"Thread started: {thread.name}, alive: {thread.is_alive()}")
        logger.info("="*70)

        return jsonify({"status": "accepted", "message": "Scan queued"}), 202
    except Exception as e:
        logger.error(f"Error queuing scan: {e}")
        LAST_SCAN_STATUS["status"] = "error"
        LAST_SCAN_STATUS["error"] = str(e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


@app.route("/status", methods=["GET"])
def scan_status():
    """Return status of the last scan attempt."""
    return jsonify(LAST_SCAN_STATUS), 200


@app.route("/test", methods=["GET", "POST"])
def test_scan():
    """Quick test endpoint to verify the app is working."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_scan": LAST_SCAN_STATUS
    }), 200


if __name__ == "__main__":
    logger.info("Starting Flask scan server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
