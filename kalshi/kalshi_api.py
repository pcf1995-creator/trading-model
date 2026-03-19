"""
kalshi_api.py — Kalshi REST API v2 client

Authentication uses API keys (RSA-PSS signing). Get yours at:
  kalshi.com → Settings → API Keys → Create Key
  This gives you a Key ID and a private key (.pem file).

Set environment variables:
  export KALSHI_KEY_ID="/path/to/your/private_key.pem"
  export KALSHI_KEY_ID="your-key-id-here"

Or in a .env file (never commit this):
  KALSHI_KEY_ID=your-key-id-here
  KALSHI_KEY_PATH=/path/to/private_key.pem

When no credentials are found the client runs in dry-run mode:
  - read operations return mock data
  - place_order() logs without hitting the API

Docs: https://docs.kalshi.com
"""

import base64
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KALSHI_CONFIG = {
    "base_url" : "https://api.elections.kalshi.com/trade-api/v2",
    "demo_url" : "https://demo-api.kalshi.co/trade-api/v2",
    "key_id"   : os.environ.get("KALSHI_KEY_ID"),         # your API key ID
    "key_path" : os.environ.get("KALSHI_KEY_PATH"),       # path to .pem file
    "demo"     : os.environ.get("KALSHI_DEMO", "false").lower() == "true",
}
# ──────────────────────────────────────────────────────────────────────────────

# Mock markets for dry-run mode (realistic BTC + ETH contracts)
_MOCK_MARKETS = [
    {
        "ticker"        : "KXBTCD-25MAR2026-T83000",
        "status"        : "open",
        "yes_ask_fp"    : 5800,    # fixed-point cents × 100
        "yes_bid_fp"    : 5500,
        "last_price_fp" : 5600,
        "volume"        : 1200,
        "close_time"    : "2026-03-25T23:59:00Z",
    },
    {
        "ticker"        : "KXBTCD-25MAR2026-T87000",
        "status"        : "open",
        "yes_ask_fp"    : 3200,
        "yes_bid_fp"    : 2900,
        "last_price_fp" : 3000,
        "volume"        : 850,
        "close_time"    : "2026-03-25T23:59:00Z",
    },
    {
        "ticker"        : "KXBTCD-25MAR2026-T90000",
        "status"        : "open",
        "yes_ask_fp"    : 1800,
        "yes_bid_fp"    : 1500,
        "last_price_fp" : 1600,
        "volume"        : 430,
        "close_time"    : "2026-03-25T23:59:00Z",
    },
    {
        "ticker"        : "KXETHD-25MAR2026-T2000",
        "status"        : "open",
        "yes_ask_fp"    : 6200,
        "yes_bid_fp"    : 5900,
        "last_price_fp" : 6000,
        "volume"        : 670,
        "close_time"    : "2026-03-25T23:59:00Z",
    },
    {
        "ticker"        : "KXETHD-25MAR2026-T2200",
        "status"        : "open",
        "yes_ask_fp"    : 3800,
        "yes_bid_fp"    : 3500,
        "last_price_fp" : 3600,
        "volume"        : 310,
        "close_time"    : "2026-03-25T23:59:00Z",
    },
]


def _fp_to_cents(fp_value: int | None) -> int | None:
    """Convert Kalshi fixed-point price (×100) to cents (0-100)."""
    if fp_value is None:
        return None
    return round(fp_value / 100)


class KalshiAPIError(Exception):
    pass


class KalshiClient:
    def __init__(self, config: dict | None = None):
        cfg           = config or KALSHI_CONFIG
        self.base_url = cfg["demo_url"] if cfg.get("demo") else cfg["base_url"]
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._key_id  = cfg.get("key_id")
        self._key_path = cfg.get("key_path")
        self._private_key = None

        if self._key_id and self._key_path:
            self._load_private_key()
            self.dry_run = False
        else:
            logger.warning(
                "No Kalshi credentials found — running in dry-run mode.\n"
                "Set KALSHI_KEY_ID and KALSHI_KEY_PATH environment variables.\n"
                "Get API keys at: kalshi.com → Settings → API Keys"
            )
            self.dry_run = True

    def _load_private_key(self):
        """Load RSA private key from .pem file."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            pem_path = Path(self._key_path)
            if not pem_path.exists():
                raise KalshiAPIError(f"Private key file not found: {pem_path}")
            with open(pem_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        except ImportError:
            raise KalshiAPIError(
                "cryptography package required for API key auth. "
                "Run: pip install cryptography"
            )

    def _sign_request(self, method: str, path: str) -> dict:
        """Generate Kalshi RSA-PSS auth headers."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        # Sign: timestamp + method + path (no query params)
        message   = (timestamp + method.upper() + path).encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY"      : self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    # ── Core HTTP ─────────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, **kwargs) -> dict:
        url     = f"{self.base_url}{path}"
        headers = self._sign_request(method, path) if self._private_key else {}
        retries = 3
        for attempt in range(retries):
            try:
                resp = self.session.request(method, url,
                                            headers=headers, **kwargs)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                if not resp.ok:
                    raise KalshiAPIError(
                        f"{method} {path} → {resp.status_code}: {resp.text[:300]}"
                    )
                return resp.json()
            except requests.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise KalshiAPIError(str(e)) from e
        return {}

    # ── Auth ──────────────────────────────────────────────────────────────────
    def login(self) -> None:
        """No-op for API key auth — signing is per-request."""
        pass

    # ── Markets ───────────────────────────────────────────────────────────────
    def get_markets(self, series_ticker: str | None = None,
                    status: str = "open", limit: int = 200) -> list[dict]:
        if self.dry_run:
            markets = _MOCK_MARKETS
            if series_ticker:
                markets = [m for m in markets if series_ticker in m["ticker"]]
            for m in markets:
                m.setdefault("yes_ask",    _fp_to_cents(m.get("yes_ask_fp")))
                m.setdefault("yes_bid",    _fp_to_cents(m.get("yes_bid_fp")))
                m.setdefault("last_price", _fp_to_cents(m.get("last_price_fp")))
            return markets

        markets = []
        cursor  = None
        while True:
            params: dict = {"limit": min(limit, 1000), "status": status}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            data   = self._request("GET", "/markets", params=params)
            batch  = data.get("markets", [])
            # Normalise fixed-point prices to cents for downstream code
            for m in batch:
                m["yes_ask"]    = _fp_to_cents(m.get("yes_ask_fp"))
                m["yes_bid"]    = _fp_to_cents(m.get("yes_bid_fp"))
                m["last_price"] = _fp_to_cents(m.get("last_price_fp"))
            markets.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
        return markets

    def get_market(self, ticker: str) -> dict:
        if self.dry_run:
            for m in _MOCK_MARKETS:
                if m["ticker"] == ticker:
                    return m
            return {}
        m = self._request("GET", f"/markets/{ticker}").get("market", {})
        m["yes_ask"]    = _fp_to_cents(m.get("yes_ask_fp"))
        m["yes_bid"]    = _fp_to_cents(m.get("yes_bid_fp"))
        m["last_price"] = _fp_to_cents(m.get("last_price_fp"))
        return m

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
                    limit_price_cents: int) -> dict:
        """
        side              : 'yes' or 'no'
        count             : number of contracts (each pays $1)
        limit_price_cents : price in cents (1–99)
        """
        # API now expects fixed-point (cents × 100)
        price_fp = limit_price_cents * 100
        payload  = {
            "ticker"      : ticker,
            "action"      : "buy",
            "side"        : side,
            "count"       : count,
            "type"        : "limit",
            "yes_price_fp": price_fp if side == "yes" else (100 - limit_price_cents) * 100,
        }
        if self.dry_run:
            logger.info(f"DRY RUN — would place order: {payload}")
            return {"status": "dry_run", **payload}
        return self._request("POST", "/portfolio/orders", json=payload)
