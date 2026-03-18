"""
kalshi_api.py — Kalshi REST API v2 client

Credentials are read from environment variables:
  KALSHI_EMAIL      — account email
  KALSHI_PASSWORD   — account password
  KALSHI_API_KEY    — API key (alternative to email/password)

Set in shell:
  export KALSHI_EMAIL="you@example.com"
  export KALSHI_PASSWORD="yourpassword"

Or in a .env file and load with python-dotenv.

When no credentials are found the client runs in dry-run mode:
  - read operations return mock data
  - place_order() logs without hitting the API
"""

import json
import logging
import os
import time
from datetime import date, datetime

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KALSHI_CONFIG = {
    "base_url" : "https://trading-api.kalshi.com/trade-api/v2",
    "demo_url" : "https://demo-api.kalshi.co/trade-api/v2",
    "email"    : os.environ.get("KALSHI_EMAIL"),
    "password" : os.environ.get("KALSHI_PASSWORD"),
    "api_key"  : os.environ.get("KALSHI_API_KEY"),
    "demo"     : os.environ.get("KALSHI_DEMO", "false").lower() == "true",
}
# ──────────────────────────────────────────────────────────────────────────────

# Mock markets for dry-run mode (realistic BTC + ETH contracts)
_MOCK_MARKETS = [
    {
        "ticker"    : "KXBTCD-25MAR2026-T83000",
        "status"    : "open",
        "yes_ask"   : 58,
        "yes_bid"   : 55,
        "last_price": 56,
        "volume"    : 1200,
        "close_time": "2026-03-25T23:59:00Z",
    },
    {
        "ticker"    : "KXBTCD-25MAR2026-T87000",
        "status"    : "open",
        "yes_ask"   : 32,
        "yes_bid"   : 29,
        "last_price": 30,
        "volume"    : 850,
        "close_time": "2026-03-25T23:59:00Z",
    },
    {
        "ticker"    : "KXBTCD-25MAR2026-T90000",
        "status"    : "open",
        "yes_ask"   : 18,
        "yes_bid"   : 15,
        "last_price": 16,
        "volume"    : 430,
        "close_time": "2026-03-25T23:59:00Z",
    },
    {
        "ticker"    : "KXETHD-25MAR2026-T2000",
        "status"    : "open",
        "yes_ask"   : 62,
        "yes_bid"   : 59,
        "last_price": 60,
        "volume"    : 670,
        "close_time": "2026-03-25T23:59:00Z",
    },
    {
        "ticker"    : "KXETHD-25MAR2026-T2200",
        "status"    : "open",
        "yes_ask"   : 38,
        "yes_bid"   : 35,
        "last_price": 36,
        "volume"    : 310,
        "close_time": "2026-03-25T23:59:00Z",
    },
]


class KalshiAPIError(Exception):
    pass


class KalshiClient:
    def __init__(self, config: dict | None = None):
        cfg          = config or KALSHI_CONFIG
        self.base_url = cfg["demo_url"] if cfg.get("demo") else cfg["base_url"]
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.token    = None

        has_password = cfg.get("email") and cfg.get("password")
        has_api_key  = bool(cfg.get("api_key"))

        if has_api_key:
            self.session.headers.update({"Authorization": f"Token {cfg['api_key']}"})
            self.dry_run = False
        elif has_password:
            self._email    = cfg["email"]
            self._password = cfg["password"]
            self.dry_run   = False
        else:
            logger.warning(
                "No Kalshi credentials found — running in dry-run mode. "
                "Set KALSHI_EMAIL + KALSHI_PASSWORD or KALSHI_API_KEY."
            )
            self.dry_run = True

    # ── Core HTTP ─────────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, **kwargs) -> dict:
        url     = f"{self.base_url}{path}"
        retries = 3
        for attempt in range(retries):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                if not resp.ok:
                    raise KalshiAPIError(
                        f"{method} {path} → {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()
            except requests.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise KalshiAPIError(str(e)) from e
        return {}

    # ── Auth ──────────────────────────────────────────────────────────────────
    def login(self) -> str:
        if self.dry_run:
            return ""
        if self.token:
            return self.token
        if not hasattr(self, "_email"):
            return ""   # using API key — no login needed
        data = self._request("POST", "/login",
                             json={"email": self._email, "password": self._password})
        self.token = data.get("token", "")
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        logger.info("Kalshi login successful")
        return self.token

    # ── Markets ───────────────────────────────────────────────────────────────
    def get_markets(self, series_ticker: str | None = None,
                    status: str = "open", limit: int = 200) -> list[dict]:
        if self.dry_run:
            if series_ticker:
                return [m for m in _MOCK_MARKETS if series_ticker in m["ticker"]]
            return _MOCK_MARKETS

        markets = []
        cursor  = None
        while True:
            params: dict = {"limit": min(limit, 200), "status": status}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            data    = self._request("GET", "/markets", params=params)
            batch   = data.get("markets", [])
            markets.extend(batch)
            cursor  = data.get("cursor")
            if not cursor or not batch:
                break
        return markets

    def get_market(self, ticker: str) -> dict:
        if self.dry_run:
            for m in _MOCK_MARKETS:
                if m["ticker"] == ticker:
                    return m
            return {}
        return self._request("GET", f"/markets/{ticker}").get("market", {})

    def get_market_history(self, ticker: str, limit: int = 100) -> list[dict]:
        if self.dry_run:
            return []
        return self._request(
            "GET", f"/markets/{ticker}/history", params={"limit": limit}
        ).get("history", [])

    # ── Portfolio ─────────────────────────────────────────────────────────────
    def get_balance(self) -> dict:
        if self.dry_run:
            return {"balance": 100_000}   # $1,000 in cents
        return self._request("GET", "/portfolio/balance")

    def place_order(self, ticker: str, side: str, count: int,
                    limit_price: int) -> dict:
        """
        side        : 'yes' or 'no'
        count       : number of contracts (each pays $1)
        limit_price : price in cents (1–99)
        """
        payload = {
            "ticker"   : ticker,
            "action"   : "buy",
            "side"     : side,
            "count"    : count,
            "type"     : "limit",
            "yes_price": limit_price if side == "yes" else 100 - limit_price,
        }
        if self.dry_run:
            logger.info(f"DRY RUN — would place order: {payload}")
            return {"status": "dry_run", **payload}
        return self._request("POST", "/portfolio/orders", json=payload)
