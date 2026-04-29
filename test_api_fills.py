#!/usr/bin/env python3
"""Debug script to test Kalshi API fills endpoint with different timestamp formats."""

import sys
from datetime import datetime, timezone

sys.path.insert(0, '/Users/pcf19/trading-model/trading-model')

from kalshi_api import KalshiClient

print("="*70)
print("KALSHI API FILLS DEBUG TEST")
print("="*70)

client = KalshiClient()
print(f"Client dry_run: {client.dry_run}")

if client.dry_run:
    print("ERROR: Running in dry-run mode - no credentials!")
    sys.exit(1)

# Test parameters
march_1_sec = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())
march_1_ms = march_1_sec * 1000

print(f"\nTimestamp conversions:")
print(f"  Seconds: {march_1_sec}")
print(f"  Milliseconds: {march_1_ms}")
print(f"  Date: {datetime.fromtimestamp(march_1_sec, tz=timezone.utc)}")

# Test 1: No min_ts (recent fills only)
print("\n" + "="*70)
print("TEST 1: Fetch recent fills (no min_ts)")
print("="*70)
try:
    recent_fills = client.get_fills(limit=100)
    print(f"✓ Returned {len(recent_fills)} fills")
    if recent_fills:
        print(f"\nFirst fill:")
        for k, v in list(recent_fills[0].items())[:5]:
            print(f"  {k}: {v}")
        print(f"  ts: {recent_fills[0].get('ts')}")
        print(f"  created_time: {recent_fills[0].get('created_time')}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: With min_ts in seconds
print("\n" + "="*70)
print(f"TEST 2: Fetch with min_ts={march_1_sec} (seconds)")
print("="*70)
try:
    sec_fills = client.get_fills(limit=100, min_ts=march_1_sec)
    print(f"✓ Returned {len(sec_fills)} fills")
    if sec_fills:
        print(f"First fill ts: {sec_fills[0].get('ts')}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: With min_ts in milliseconds
print("\n" + "="*70)
print(f"TEST 3: Fetch with min_ts={march_1_ms} (milliseconds)")
print("="*70)
try:
    ms_fills = client.get_fills(limit=100, min_ts=march_1_ms)
    print(f"✓ Returned {len(ms_fills)} fills")
    if ms_fills:
        print(f"First fill ts: {ms_fills[0].get('ts')}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Which test returned the most fills?")
