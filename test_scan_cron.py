#!/usr/bin/env python3
"""
Test script to simulate what the cron job does and identify failures.
Run this to see the exact error the cron job hits.
"""

import sys
import subprocess
import traceback
from datetime import datetime, timezone

print(f"\n{'='*70}")
print(f"Testing Kalshi Scan Cron — {datetime.now(timezone.utc).isoformat()}")
print(f"{'='*70}\n")

# ── Test 1: Run kalshi_crypto_weekly.py ──────────────────────────────────────
print("[TEST 1] Running kalshi_crypto_weekly.py --auto-save-db\n")
try:
    result = subprocess.run(
        [sys.executable, "kalshi_crypto_weekly.py", "--auto-save-db"],
        capture_output=True,
        text=True,
        timeout=900,
    )

    print(f"Exit code: {result.returncode}")
    print(f"\nSTDOUT:\n{result.stdout}")
    print(f"\nSTDERR:\n{result.stderr}")

    if result.returncode != 0:
        print(f"\n❌ SCAN FAILED with exit code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✓ Scan completed successfully")

except subprocess.TimeoutExpired:
    print(f"❌ SCAN TIMED OUT after 15 minutes")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Test 2: Auto-placement ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[TEST 2] Testing auto-placement\n")
try:
    from kalshi_crypto import place_scheduled_orders
    from kalshi_api import KalshiClient

    print("Initializing Kalshi client...")
    kalshi_client = KalshiClient()
    print(f"  ✓ Client created (dry_run={kalshi_client.dry_run})")

    print("Calling place_scheduled_orders()...")
    stats = place_scheduled_orders(kalshi_client)
    print(f"  ✓ Auto-placement completed: {stats}")

except Exception as e:
    print(f"❌ AUTO-PLACEMENT FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*70}")
print("✓ All tests passed! Cron job should work.")
print(f"{'='*70}\n")
