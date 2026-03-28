"""
kalshi_api.py — Kalshi REST API v2 client

Authentication uses API keys (RSA-PSS signing). Get yours at:
  kalshi.com → Settings → API Keys → Create Key
  This gives you a Key ID and a private key (.pem file).

Set environment variables:
  export KALSHI_KEY_ID="your-key-id-here"
  export KALSHI_KEY_PATH="/path/to/your/private_key.pem"

Or in a .env file (never commit this):
  KALSHI_KEY_ID=your-key-id-here
  KALSHI_KEY_PATH=/path/to/private_key.pem

When no credentials are found the client runs in dry-run mode:
  - read operations return mock data
  - place_order() logs without hitting the API

Docs: https://docs.kalshi.com
"""

import base64
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


def _dollars_to_cents(dollars: str | float | None) -> int | None:
    """Convert Kalshi dollar string (e.g. '0.5800') to cents (58)."""
    if dollars is None:
        return None
    try:
        return round(float(dollars) * 100)
    except (ValueError, TypeError):
        return None


def _normalize_prices(m: dict) -> None:
    """Populate yes_ask / yes_bid / no_ask / no_bid / last_price in cents."""
    if m.get("yes_ask") is None:
        if m.get("yes_ask_fp") is not None:
            m["yes_ask"] = _fp_to_cents(m["yes_ask_fp"])
        elif m.get("yes_ask_dollars") is not None:
            m["yes_ask"] = _dollars_to_cents(m["yes_ask_dollars"])
    if m.get("yes_bid") is None:
        if m.get("yes_bid_fp") is not None:
            m["yes_bid"] = _fp_to_cents(m["yes_bid_fp"])
        elif m.get("yes_bid_dollars") is not None:
            m["yes_bid"] = _dollars_to_cents(m["yes_bid_dollars"])
    if m.get("no_ask") is None:
        if m.get("no_ask_dollars") is not None:
            m["no_ask"] = _dollars_to_cents(m["no_ask_dollars"])
    if m.get("no_bid") is None:
        if m.get("no_bid_dollars") is not None:
            m["no_bid"] = _dollars_to_cents(m["no_bid_dollars"])
    if m.get("last_price") is None:
        if m.get("last_price_fp") is not None:
            m["last_price"] = _fp_to_cents(m["last_price_fp"])
        elif m.get("last_price_dollars") is not None:
            m["last_price"] = _dollars_to_cents(m["last_price_dollars"])


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

        self._key_content = cfg.get("key_content")  # raw PEM string (Streamlit secrets)

        if self._key_id and (self._key_path or self._key_content):
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
        """Load RSA private key from .pem file or raw PEM string."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            if self._key_content:
                pem_bytes = self._key_content.encode() if isinstance(self._key_content, str) else self._key_content
            else:
                pem_path = Path(self._key_path)
                if not pem_path.exists():
                    raise KalshiAPIError(f"Private key file not found: {pem_path}")
                with open(pem_path, "rb") as f:
                    pem_bytes = f.read()
            self._private_key = serialization.load_pem_private_key(
                pem_bytes, password=None, backend=default_backend()
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
        # Sign: timestamp + method + FULL path (includes /trade-api/v2 prefix)
        full_path = "/trade-api/v2" + path
        message   = (timestamp + method.upper() + full_path).encode("utf-8")
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
            for m in batch:
                _normalize_prices(m)
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
        _normalize_prices(m)
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

    def get_positions(self) -> list[dict]:
        """Return all open positions from the portfolio."""
        if self.dry_run:
            return []
        return self._request("GET", "/portfolio/positions").get("market_positions", [])

    def get_fills(self, limit: int = 200) -> list[dict]:
        """Return recent trade fills (executed orders)."""
        if self.dry_run:
            return []
        fills  = []
        cursor = None
        while True:
            params = {"limit": 1000}   # max per page
            if cursor:
                params["cursor"] = cursor
            data   = self._request("GET", "/portfolio/fills", params=params)
            batch  = data.get("fills", [])
            fills.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
            if limit and len(fills) >= limit:
                break
        return fills

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
