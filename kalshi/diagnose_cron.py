#!/usr/bin/env python3
"""
Diagnostic script to identify why the Kalshi scan cron is failing with 503.
Tests imports, dependencies, and runs a minimal scan to catch errors early.
"""

import sys
import os
import traceback
from datetime import datetime

print(f"\n{'='*70}")
print(f"Kalshi Scan Cron Diagnostic — {datetime.now().isoformat()}")
print(f"{'='*70}\n")

# ── Test 1: Flask import ──────────────────────────────────────────────────────
print("[1/6] Testing Flask import...")
try:
    from flask import Flask, jsonify
    print("  ✓ Flask imported OK")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# ── Test 2: Core dependencies ────────────────────────────────────────────────
print("\n[2/6] Testing core dependencies...")
deps = ["numpy", "pandas", "scipy", "yfinance", "sklearn"]
for dep in deps:
    try:
        __import__(dep)
        print(f"  ✓ {dep}")
    except ImportError as e:
        print(f"  ✗ {dep}: {e}")
        sys.exit(1)

# ── Test 3: Kalshi API imports ───────────────────────────────────────────────
print("\n[3/6] Testing Kalshi API imports...")
try:
    from kalshi_api import KalshiClient
    print("  ✓ KalshiClient imported OK")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Test 4: Kalshi crypto module ─────────────────────────────────────────────
print("\n[4/6] Testing kalshi_crypto imports...")
try:
    from kalshi_crypto import (
        CRYPTO_ASSETS,
        load_crypto_models,
        load_calibration,
    )
    print(f"  ✓ kalshi_crypto imported OK")
    print(f"  ✓ CRYPTO_ASSETS: {len(CRYPTO_ASSETS)} assets")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ✗ ERROR during import: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Test 5: Model loading ────────────────────────────────────────────────────
print("\n[5/6] Testing model loading...")
try:
    print("  Loading crypto models...")
    models = load_crypto_models()
    print(f"  ✓ Models loaded: {len(models)} assets")

    print("  Loading calibration...")
    calibration = load_calibration()
    print(f"  ✓ Calibration loaded: {len(calibration)} entries")
except FileNotFoundError as e:
    print(f"  ✗ MISSING FILE: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Test 6: Kalshi API connectivity ──────────────────────────────────────────
print("\n[6/6] Testing Kalshi API connectivity...")
try:
    client = KalshiClient()
    print("  ✓ KalshiClient initialized")

    # Try a simple API call
    markets = client.list_markets(status="open", limit=1)
    print(f"  ✓ API call successful (sample: {len(markets)} markets returned)")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Test 7: Database connectivity ────────────────────────────────────────────
print("\n[7/6] Testing database connectivity...")
try:
    from db import load_paper_trades
    trades = load_paper_trades()
    print(f"  ✓ Database connected: {len(trades)} paper trades")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("✓ All diagnostics passed! Cron app should be able to start.")
print("  Issue may be:")
print("  1. Memory/resource limit hit during actual scan execution")
print("  2. Timeout during data fetching")
print("  3. Kalshi API rate limiting")
print(f"{'='*70}\n")
