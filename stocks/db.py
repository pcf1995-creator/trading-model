"""
stocks/db.py — Stocks-side persistence layer.

Owns Supabase tables:
  - stock_paper_trades   (per-ticker probability-model paper trades)
  - stock_real_trades    (real-money trades, manual close)
  - lt_positions         (Absolute Return L/S; conviction also lives here w/ cv_ id prefix)
  - stock-models bucket  (Supabase Storage for *.joblib + features_*.csv)

Falls back to local JSON if Supabase is unavailable.
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable so `shared` resolves regardless of how this
# module is loaded (Streamlit Cloud, `python stocks/upload_models.py`, etc.)
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.db_common import _get_client, _load_json, _save_json

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
_STOCK_PAPER_TRADES_JSON   = ROOT / "stock_paper_trades.json"
_STOCK_REAL_TRADES_JSON    = ROOT / "stock_real_trades.json"
_LT_POSITIONS_JSON         = ROOT / "lt_positions.json"
_CONVICTION_POSITIONS_JSON = ROOT / "conviction_positions.json"


# ── Stock paper trades ─────────────────────────────────────────────────────────
def load_stock_paper_trades() -> list[dict]:
    client = _get_client()
    if client:
        try:
            resp = (client.table("stock_paper_trades")
                    .select("*")
                    .order("placed_at", desc=False)
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.warning(f"load_stock_paper_trades failed: {e}")
    return _load_json(_STOCK_PAPER_TRADES_JSON)


def add_stock_paper_trade(trade: dict) -> None:
    client = _get_client()
    if client:
        client.table("stock_paper_trades").insert(trade).execute()
        return
    trades = _load_json(_STOCK_PAPER_TRADES_JSON)
    trades.append(trade)
    _save_json(_STOCK_PAPER_TRADES_JSON, trades)


def close_stock_paper_trade(trade_id: str, exit_price: float, exit_date: str,
                             exit_reason: str, pnl_dollars: float, pnl_pct: float) -> None:
    updates = {
        "status": "closed",
        "exit_price": exit_price,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
        "pnl_dollars": pnl_dollars,
        "pnl_pct": pnl_pct,
    }
    client = _get_client()
    if client:
        try:
            client.table("stock_paper_trades").update(updates).eq("id", trade_id).execute()
            return
        except Exception as e:
            logger.warning(f"close_stock_paper_trade failed: {e}")
    trades = _load_json(_STOCK_PAPER_TRADES_JSON)
    for t in trades:
        if t.get("id") == trade_id:
            t.update(updates)
    _save_json(_STOCK_PAPER_TRADES_JSON, trades)


# ── Stock REAL trades (actual money) ───────────────────────────────────────────
# Parallel log to stock_paper_trades. Never auto-settled — closes are manual so
# the row reflects the real fill price. Populated only by explicit user action.
def load_stock_real_trades() -> list[dict]:
    client = _get_client()
    if client:
        try:
            resp = (client.table("stock_real_trades")
                    .select("*")
                    .order("placed_at", desc=False)
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.warning(f"load_stock_real_trades failed: {e}")
    return _load_json(_STOCK_REAL_TRADES_JSON)


def add_stock_real_trade(trade: dict, source: str = "scan",
                          paper_trade_id: str | None = None) -> None:
    """Insert a real-trade row. `source` is 'scan', 'paper_promotion', or 'manual'."""
    row = {**trade, "source": source}
    if paper_trade_id is not None:
        row["paper_trade_id"] = paper_trade_id
    client = _get_client()
    if client:
        client.table("stock_real_trades").insert(row).execute()
        return
    trades = _load_json(_STOCK_REAL_TRADES_JSON)
    trades.append(row)
    _save_json(_STOCK_REAL_TRADES_JSON, trades)




def update_stock_real_trade(trade_id: str, updates: dict) -> None:
    """Update editable fields of an open real trade (entry_price, entry_date, shares, side).

    Automatically recomputes `dollars` when entry_price or shares change.
    """
    allowed = {"ticker", "side", "entry_price", "entry_date", "shares",
               "exit_price", "exit_date", "exit_reason", "pnl_pct", "pnl_dollars"}
    payload = {k: v for k, v in updates.items() if k in allowed}
    if not payload:
        return

    # Recompute invested dollars if price or shares changed
    client = _get_client()
    rows = []
    if client:
        try:
            resp = client.table("stock_real_trades").select("*").eq("id", trade_id).execute()
            rows = resp.data or []
        except Exception as e:
            logger.warning(f"update_stock_real_trade fetch failed: {e}")
    if not rows:
        rows = [t for t in _load_json(_STOCK_REAL_TRADES_JSON) if t.get("id") == trade_id]
    if rows:
        r = rows[0]
        ep = float(payload.get("entry_price", r.get("entry_price") or 0))
        sh = float(payload.get("shares",      r.get("shares")      or 0))
        payload["dollars"] = round(ep * sh, 2)

    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    if client:
        try:
            client.table("stock_real_trades").update(payload).eq("id", trade_id).execute()
            return
        except Exception as e:
            logger.warning(f"update_stock_real_trade failed: {e}")
    trades = _load_json(_STOCK_REAL_TRADES_JSON)
    for t in trades:
        if t.get("id") == trade_id:
            t.update(payload)
    _save_json(_STOCK_REAL_TRADES_JSON, trades)


def close_stock_real_trade(trade_id: str, exit_price: float, exit_date: str,
                            exit_reason: str, notes: str | None = None) -> None:
    """Manually close a real trade. Computes P&L from the row's entry_price/shares/side."""
    client = _get_client()
    rows = []
    if client:
        try:
            resp = client.table("stock_real_trades").select("*").eq("id", trade_id).execute()
            rows = resp.data or []
        except Exception as e:
            logger.warning(f"close_stock_real_trade fetch failed: {e}")
    if not rows:
        rows = [t for t in _load_json(_STOCK_REAL_TRADES_JSON) if t.get("id") == trade_id]
    if not rows:
        raise ValueError(f"real trade {trade_id} not found")
    r = rows[0]

    entry_price = float(r["entry_price"])
    shares = float(r.get("shares") or 0)
    side = r.get("side", "long")
    if side == "short":
        pnl_pct = (entry_price - exit_price) / entry_price if entry_price else 0.0
        pnl_dollars = round((entry_price - exit_price) * shares, 2)
    else:
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
        pnl_dollars = round((exit_price - entry_price) * shares, 2)

    updates = {
        "status": "closed",
        "exit_price": exit_price,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
        "pnl_dollars": pnl_dollars,
        "pnl_pct": round(pnl_pct * 100, 2),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if notes:
        updates["notes"] = notes

    if client:
        try:
            client.table("stock_real_trades").update(updates).eq("id", trade_id).execute()
            return
        except Exception as e:
            logger.warning(f"close_stock_real_trade failed: {e}")
    trades = _load_json(_STOCK_REAL_TRADES_JSON)
    for t in trades:
        if t.get("id") == trade_id:
            t.update(updates)
    _save_json(_STOCK_REAL_TRADES_JSON, trades)


def promote_paper_to_real(paper_trade_id: str) -> str:
    """Copy an existing paper trade into stock_real_trades, leaving the paper row untouched.
    Returns the new real-trade id."""
    import uuid
    paper_rows = [t for t in load_stock_paper_trades() if t.get("id") == paper_trade_id]
    if not paper_rows:
        raise ValueError(f"paper trade {paper_trade_id} not found")
    p = paper_rows[0]

    new_id = str(uuid.uuid4())
    new_row = {
        "id"         : new_id,
        "ticker"     : p["ticker"],
        "side"       : p.get("side", "long"),
        "entry_price": float(p["entry_price"]),
        "entry_date" : p.get("entry_date"),
        "shares"     : float(p.get("shares") or 0),
        "dollars"    : float(p.get("dollars") or 0),
        "model_prob" : p.get("model_prob"),
        "status"     : "open",
        "placed_at"  : datetime.now(timezone.utc).isoformat(),
    }
    add_stock_real_trade(new_row, source="paper_promotion", paper_trade_id=paper_trade_id)
    return new_id


# ── Long-term L/S positions ────────────────────────────────────────────────────
def load_lt_positions() -> list[dict]:
    client = _get_client()
    if client:
        try:
            resp = (client.table("lt_positions")
                    .select("*")
                    .filter("id", "not.like", "cv_%")
                    .order("entry_date", desc=False)
                    .execute())
            rows = resp.data or []
            result = []
            for row in rows:
                extra = row.pop("data", None) or {}
                merged = {**extra, **row}
                result.append(merged)
            return result
        except Exception as e:
            logger.warning(f"load_lt_positions failed: {e}")
    return _load_json(_LT_POSITIONS_JSON)


def save_lt_positions(positions: list[dict]) -> None:
    client = _get_client()
    if client:
        try:
            now = datetime.now(timezone.utc).isoformat()
            _KNOWN_COLS = {
                "id", "ticker", "direction", "status", "entry_price", "entry_date",
                "shares", "cost", "exit_price", "exit_date", "exit_reason",
                "pnl_dollars", "pnl_pct", "score_at_entry", "updated_at",
            }
            rows = []
            for pos in positions:
                pos_id = pos.get("id") or f"{pos['ticker']}_{pos.get('entry_date', 'unknown')}"
                known = {k: pos.get(k) for k in _KNOWN_COLS if k in pos}
                known["id"] = pos_id
                known["score_at_entry"] = pos.get("composite_score")
                known["updated_at"] = now
                extra = {k: v for k, v in pos.items() if k not in _KNOWN_COLS and k != "composite_score"}
                if extra:
                    known["data"] = extra
                rows.append(known)
            if rows:
                client.table("lt_positions").upsert(rows).execute()
            return
        except Exception as e:
            logger.warning(f"save_lt_positions failed: {e}")
    _save_json(_LT_POSITIONS_JSON, positions)


# ── Conviction L/S positions ───────────────────────────────────────────────────
# Stored in the same lt_positions Supabase table, distinguished by "cv_" ID prefix.
_CV_KNOWN_COLS = {
    "id", "ticker", "direction", "status", "entry_price", "entry_date",
    "shares", "cost", "exit_price", "exit_date", "exit_reason",
    "pnl_dollars", "pnl_pct", "score_at_entry", "updated_at",
}


def load_conviction_positions() -> list[dict]:
    client = _get_client()
    if client:
        try:
            resp = (client.table("lt_positions")
                    .select("*")
                    .filter("id", "like", "cv_%")
                    .order("entry_date", desc=False)
                    .execute())
            rows = resp.data or []
            result = []
            for row in rows:
                extra  = row.pop("data", None) or {}
                merged = {**extra, **row}
                result.append(merged)
            return result
        except Exception as e:
            logger.warning(f"load_conviction_positions failed: {e}")
    return _load_json(_CONVICTION_POSITIONS_JSON)


def save_conviction_positions(positions: list[dict]) -> None:
    client = _get_client()
    if client:
        try:
            now  = datetime.now(timezone.utc).isoformat()
            rows = []
            for pos in positions:
                raw_id = pos.get("id") or f"{pos['ticker']}_{pos.get('entry_date', 'unknown')}"
                pos_id = raw_id if raw_id.startswith("cv_") else f"cv_{raw_id}"
                known  = {k: pos.get(k) for k in _CV_KNOWN_COLS if k in pos}
                known["id"]             = pos_id
                known["score_at_entry"] = (
                    pos.get("score_at_entry")
                    if pos.get("score_at_entry") is not None
                    else pos.get("composite_score")
                )
                known["updated_at"]     = now
                extra = {k: v for k, v in pos.items()
                         if k not in _CV_KNOWN_COLS and k != "composite_score"}
                if extra:
                    known["data"] = extra
                rows.append(known)
            if rows:
                import math as _math
                # Strip NaN/Inf floats before upsert — JSON serialization rejects them
                # and the Supabase client will raise, silently falling back to local file.
                def _clean(v):
                    if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                        return None
                    return v
                rows = [{k: _clean(v) for k, v in r.items()} for r in rows]
                client.table("lt_positions").upsert(rows).execute()
            return
        except Exception as e:
            logger.warning(f"save_conviction_positions failed: {e}")
    _save_json(_CONVICTION_POSITIONS_JSON, positions)


# ── Stock model file storage (Supabase Storage) ─────────────────────────────
_MODEL_CACHE_DIR = Path("/tmp/trading_models")


def get_stock_file(filename: str, local_root: Path | None = None) -> Path | None:
    """Return path to a stock model file, downloading from Supabase Storage if needed.

    Priority: local_root/filename, then /tmp cache, then Supabase Storage 'stock-models'.
    """
    if local_root:
        local_path = local_root / filename
        if local_path.exists():
            return local_path

    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _MODEL_CACHE_DIR / filename
    if cached.exists():
        return cached

    client = _get_client()
    if not client:
        return None
    try:
        data = client.storage.from_("stock-models").download(filename)
        cached.write_bytes(data)
        logger.info(f"Downloaded {filename} from Supabase Storage")
        return cached
    except Exception as e:
        logger.warning(f"Could not download {filename} from Supabase Storage: {e}")
        return None


def upload_stock_file(filename: str, local_path: Path) -> bool:
    """Upload a stock model file to Supabase Storage bucket 'stock-models'."""
    client = _get_client()
    if not client:
        logger.error("No Supabase client — cannot upload")
        return False
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        try:
            client.storage.from_("stock-models").upload(filename, data)
        except Exception:
            client.storage.from_("stock-models").update(filename, data)
        logger.info(f"Uploaded {filename} to Supabase Storage")
        return True
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}")
        return False


# ── Overnight drift paper trades ──────────────────────────────────────────────
#
# Supabase table DDL (run once in the Supabase SQL editor):
#
#   CREATE TABLE overnight_paper_trades (
#       id           TEXT PRIMARY KEY,
#       ticker       TEXT NOT NULL,
#       entry_date   DATE NOT NULL,
#       entry_price  FLOAT NOT NULL,
#       shares       FLOAT,
#       dollars      FLOAT,
#       status       TEXT DEFAULT 'open',
#       exit_date    DATE,
#       exit_price   FLOAT,
#       pnl_pct      FLOAT,
#       pnl_dollars  FLOAT,
#       t_stat       FLOAT,
#       net_sharpe   FLOAT,
#       win_rate     FLOAT,
#       placed_at    TIMESTAMPTZ DEFAULT now(),
#       updated_at   TIMESTAMPTZ
#   );
#
_OVERNIGHT_PAPER_TRADES_JSON = ROOT / "overnight_paper_trades.json"


def load_overnight_paper_trades() -> list[dict]:
    client = _get_client()
    if client:
        try:
            resp = (client.table("overnight_paper_trades")
                    .select("*")
                    .order("placed_at", desc=False)
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.warning(f"load_overnight_paper_trades failed: {e}")
    return _load_json(_OVERNIGHT_PAPER_TRADES_JSON)


def add_overnight_paper_trade(trade: dict) -> None:
    client = _get_client()
    if client:
        try:
            client.table("overnight_paper_trades").insert(trade).execute()
            return
        except Exception as e:
            logger.warning(f"add_overnight_paper_trade failed: {e}")
    trades = _load_json(_OVERNIGHT_PAPER_TRADES_JSON)
    trades.append(trade)
    _save_json(_OVERNIGHT_PAPER_TRADES_JSON, trades)


def close_overnight_paper_trade(trade_id: str, exit_price: float, exit_date: str,
                                 pnl_pct: float, pnl_dollars: float) -> None:
    updates = {
        "status":      "closed",
        "exit_price":  exit_price,
        "exit_date":   exit_date,
        "pnl_pct":     pnl_pct,
        "pnl_dollars": pnl_dollars,
        "updated_at":  datetime.now(timezone.utc).isoformat(),
    }
    client = _get_client()
    if client:
        try:
            client.table("overnight_paper_trades").update(updates).eq("id", trade_id).execute()
            return
        except Exception as e:
            logger.warning(f"close_overnight_paper_trade failed: {e}")
    trades = _load_json(_OVERNIGHT_PAPER_TRADES_JSON)
    for t in trades:
        if t.get("id") == trade_id:
            t.update(updates)
    _save_json(_OVERNIGHT_PAPER_TRADES_JSON, trades)


def close_overnight_real_trade(trade_id: str, exit_price: float, exit_date: str) -> None:
    """Close a real overnight trade. Fetches entry_price/shares to compute P&L."""
    client = _get_client()
    rows = []
    if client:
        try:
            resp = client.table("overnight_paper_trades").select("*").eq("id", trade_id).execute()
            rows = resp.data or []
        except Exception as e:
            logger.warning(f"close_overnight_real_trade fetch failed: {e}")
    if not rows:
        rows = [t for t in _load_json(_OVERNIGHT_PAPER_TRADES_JSON) if t.get("id") == trade_id]
    if not rows:
        raise ValueError(f"overnight trade {trade_id} not found")
    r = rows[0]
    entry_price = float(r.get("entry_price") or 0)
    shares      = float(r.get("shares") or 0)
    pnl_pct     = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
    pnl_dollars = round((exit_price - entry_price) * shares, 2)
    close_overnight_paper_trade(
        trade_id    = trade_id,
        exit_price  = exit_price,
        exit_date   = exit_date,
        pnl_pct     = round(pnl_pct * 100, 4),
        pnl_dollars = pnl_dollars,
    )


# ── Crypto trend paper strategy ─────────────────────────────────────────────────
#
# Single-row JSON state store for the ETH/SOL/HYPE 200d trend-to-cash paper
# strategy (Crypto Trend tab). Supabase table DDL (run once):
#
#   CREATE TABLE IF NOT EXISTS crypto_strategy (
#       id          TEXT PRIMARY KEY,
#       state       JSONB NOT NULL DEFAULT '{}'::jsonb,
#       updated_at  TIMESTAMPTZ DEFAULT NOW()
#   );
#
_CRYPTO_STRATEGY_JSON = ROOT / "crypto_strategy.json"


def load_crypto_strategy() -> dict | None:
    client = _get_client()
    if client:
        try:
            resp = client.table("crypto_strategy").select("state").eq("id", "default").execute()
            rows = resp.data or []
            return (rows[0].get("state") or None) if rows else None
        except Exception as e:
            logger.warning(f"load_crypto_strategy failed: {e}")
    data = _load_json(_CRYPTO_STRATEGY_JSON)
    return data or None


def save_crypto_strategy(state: dict) -> None:
    row = {"id": "default", "state": state,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    client = _get_client()
    if client:
        try:
            client.table("crypto_strategy").upsert(row, on_conflict="id").execute()
            return
        except Exception as e:
            logger.warning(f"save_crypto_strategy failed: {e}")
    _save_json(_CRYPTO_STRATEGY_JSON, state)


# ── SMA trades ────────────────────────────────────────────────────────────────
#
# Supabase table DDL (run once):
#
#   CREATE TABLE IF NOT EXISTS sma_trades (
#       id           TEXT PRIMARY KEY,
#       ticker       TEXT NOT NULL,
#       entry_date   DATE NOT NULL,
#       entry_price  NUMERIC NOT NULL,
#       shares       NUMERIC,
#       dollars      NUMERIC,
#       status       TEXT DEFAULT 'open',
#       exit_date    DATE,
#       exit_price   NUMERIC,
#       pnl_pct      NUMERIC,
#       pnl_dollars  NUMERIC,
#       notes        TEXT,
#       placed_at    TIMESTAMPTZ DEFAULT NOW(),
#       updated_at   TIMESTAMPTZ
#   );
#
_SMA_TRADES_JSON = ROOT / "sma_trades.json"


def load_sma_trades() -> list[dict]:
    client = _get_client()
    if client:
        try:
            resp = (client.table("sma_trades")
                    .select("*")
                    .order("placed_at", desc=False)
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.warning(f"load_sma_trades failed: {e}")
    return _load_json(_SMA_TRADES_JSON)


def add_sma_trade(trade: dict) -> None:
    client = _get_client()
    if client:
        try:
            client.table("sma_trades").insert(trade).execute()
            return
        except Exception as e:
            logger.warning(f"add_sma_trade failed: {e}")
    trades = _load_json(_SMA_TRADES_JSON)
    trades.append(trade)
    _save_json(_SMA_TRADES_JSON, trades)


def close_sma_trade(trade_id: str, exit_price: float, exit_date: str) -> None:
    """Close an SMA trade, auto-computing P&L from the stored entry row."""
    client = _get_client()
    rows = []
    if client:
        try:
            resp = client.table("sma_trades").select("*").eq("id", trade_id).execute()
            rows = resp.data or []
        except Exception as e:
            logger.warning(f"close_sma_trade fetch failed: {e}")
    if not rows:
        rows = [t for t in _load_json(_SMA_TRADES_JSON) if t.get("id") == trade_id]
    if not rows:
        raise ValueError(f"SMA trade {trade_id} not found")
    r = rows[0]
    entry_price = float(r.get("entry_price") or 0)
    shares      = float(r.get("shares") or 0)
    pnl_pct     = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
    pnl_dollars = round((exit_price - entry_price) * shares, 2)
    updates = {
        "status":      "closed",
        "exit_price":  round(exit_price, 4),
        "exit_date":   exit_date,
        "pnl_pct":     round(pnl_pct, 4),
        "pnl_dollars": pnl_dollars,
        "updated_at":  datetime.now(timezone.utc).isoformat(),
    }
    client = _get_client()
    if client:
        try:
            client.table("sma_trades").update(updates).eq("id", trade_id).execute()
            return
        except Exception as e:
            logger.warning(f"close_sma_trade update failed: {e}")
    trades = _load_json(_SMA_TRADES_JSON)
    for t in trades:
        if t.get("id") == trade_id:
            t.update(updates)
    _save_json(_SMA_TRADES_JSON, trades)


# ── Fundamental cache ──────────────────────────────────────────────────────────
#
# Supabase table DDL (run once in the Supabase SQL editor):
#
#   CREATE TABLE fundamental_cache (
#       ticker           TEXT PRIMARY KEY,
#       fetch_date       DATE NOT NULL,
#       source           TEXT,
#       market_cap       FLOAT,
#       roe              FLOAT,
#       gross_margin     FLOAT,
#       revenue_growth   FLOAT,
#       earnings_growth  FLOAT,
#       profit_margin    FLOAT,
#       fcf_yield        FLOAT,
#       debt_equity_neg  FLOAT,
#       pb               FLOAT,
#       fwd_pe           FLOAT,
#       eps_qoq_growth   FLOAT,
#       beat_rate        FLOAT,
#       avg_eps_surprise FLOAT,
#       next_earnings_date DATE,
#       updated_at       TIMESTAMPTZ DEFAULT now()
#   );
#
FUNDAMENTAL_CACHE_TTL_DAYS = 7

_FUND_CACHE_COLS = {
    "market_cap", "roe", "gross_margin", "revenue_growth", "earnings_growth",
    "profit_margin", "fcf_yield", "debt_equity_neg", "pb", "fwd_pe",
    "eps_qoq_growth", "beat_rate", "avg_eps_surprise", "next_earnings_date",
    "source",
}


def get_fundamental_cache(ticker: str) -> dict | None:
    """Return cached fundamentals for ticker if within TTL, else None.

    Returns None on any error so callers always fall back to live fetch.
    """
    from datetime import date as _date
    import math
    client = _get_client()
    if not client:
        return None
    try:
        resp = (
            client.table("fundamental_cache")
            .select("*")
            .eq("ticker", ticker)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        row = rows[0]
        fetch_date_str = row.get("fetch_date")
        if not fetch_date_str:
            return None
        fetch_date = _date.fromisoformat(str(fetch_date_str))
        if (_date.today() - fetch_date).days > FUNDAMENTAL_CACHE_TTL_DAYS:
            return None
        return row
    except Exception as e:
        logger.warning(f"get_fundamental_cache({ticker}) failed: {e}")
        return None


def upsert_fundamental_cache(ticker: str, data: dict) -> None:
    """Write or overwrite the fundamental cache row for ticker.

    Silently skips NaN/inf/None values so the column stays NULL rather than
    storing a sentinel that would look like real data.
    """
    import math
    client = _get_client()
    if not client:
        return
    try:
        row: dict = {
            "ticker":     ticker,
            "fetch_date": datetime.now(timezone.utc).date().isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for k, v in data.items():
            if k not in _FUND_CACHE_COLS:
                continue
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            row[k] = v
        client.table("fundamental_cache").upsert(row, on_conflict="ticker").execute()
    except Exception as e:
        logger.warning(f"upsert_fundamental_cache({ticker}) failed: {e}")
