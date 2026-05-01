#!/usr/bin/env python3
"""Test each import to find what hangs."""
import sys

print("TEST: Starting import tests", flush=True)
sys.stdout.flush()

print("TEST: Importing argparse", flush=True)
sys.stdout.flush()
import argparse

print("TEST: Importing logging", flush=True)
sys.stdout.flush()
import logging

print("TEST: Importing math", flush=True)
sys.stdout.flush()
import math

print("TEST: Importing datetime", flush=True)
sys.stdout.flush()
from datetime import date, timedelta

print("TEST: Importing pathlib", flush=True)
sys.stdout.flush()
from pathlib import Path

print("TEST: Importing numpy", flush=True)
sys.stdout.flush()
import numpy as np

print("TEST: Importing pandas", flush=True)
sys.stdout.flush()
import pandas as pd

print("TEST: Importing yfinance", flush=True)
sys.stdout.flush()
import yfinance as yf

print("TEST: Importing scipy", flush=True)
sys.stdout.flush()
from scipy import stats

print("TEST: Importing kalshi_api", flush=True)
sys.stdout.flush()
from kalshi_api import KalshiClient

print("TEST: Importing kalshi_crypto (this might hang)", flush=True)
sys.stdout.flush()
from kalshi_crypto import (
    CRYPTO_ASSETS,
    load_crypto_models,
)

print("TEST: All imports successful!", flush=True)
sys.stdout.flush()
