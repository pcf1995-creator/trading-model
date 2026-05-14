"""
stocks/conviction.py — Multi-factor conviction L/S scanner (30–90 day hold)

Four scoring pillars:
  Technical   25% — price vs SMAs, medium-term momentum, ADX, 52-week proximity
  Fundamental 35% — ROE, margins, growth, FCF yield, leverage, valuation
  Earnings    25% — EPS beat history, surprise magnitude, quarterly growth
  Macro       15% — sector rotation vs SPY, market regime, VIX, rate trend

Positions: up to MAX_LONG long + MAX_SHORT short.
A slot is only filled when composite score clears MIN_LONG_SCORE / MIN_SHORT_SCORE,
so fewer than 10 trades may be recommended when conviction is low.
Hard stop 15%. Monthly reassessment.
"""

import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Alpha Vantage API key — try Streamlit secrets, fall back to environment
import os
ALPHA_VANTAGE_KEY = ""
try:
    import streamlit as st
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "")
except Exception:
    pass
if not ALPHA_VANTAGE_KEY:
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as _db

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_LONG_BY_REGIME  = {"Expansion": 7, "Caution": 5, "Contraction": 3}
MAX_SHORT_BY_REGIME = {"Expansion": 3, "Caution": 5, "Contraction": 7}
HARD_STOP_PCT       = 0.20
SMALL_CAP_THRESH   = 5e9
EXIT_DECILE        = 0.35      # exit long below 35th pct; short above 65th pct
MIN_LONG_SCORE     = 0.25      # composite must beat this to open a long
MIN_SHORT_SCORE    = -0.25     # composite must beat this (below) to open a short
EARNINGS_FLAG_DAYS = 14        # flag if next earnings within this many days
EARNINGS_CONFIDENCE = 0.70     # multiply earnings z-score by this if flagged

# Paper portfolio limits
PAPER_MAX_POSITIONS = 15
PAPER_MAX_GROSS     = 5_000.0
EARNINGS_TRIM_GAIN  = 0.15     # flag for trim if unrealized gain exceeds this
EARNINGS_TRIM_DAYS  = 7        # flag if earnings within this many days AND gain > above

SPY_DRAWDOWN_THRESHOLD = 0.10  # 10% off 52w high triggers one regime signal

# Pillar weights — must sum to 1.0
W_TECHNICAL   = 0.25
W_FUNDAMENTAL = 0.35
W_EARNINGS    = 0.25
W_MACRO       = 0.15

# Macro flat adjustments applied uniformly across all tickers
REGIME_ADJ = {"Expansion": 0.20, "Caution": 0.00, "Contraction": -0.20}
VIX_ADJ    = {"low": 0.15, "mid": 0.00, "high": -0.15}
VIX_LOW    = 15
VIX_HIGH   = 25

# Ticker → sector ETF for cross-sectional relative-strength signal
SECTOR_MAP: dict[str, str] = {
    # Technology / Software / Semis
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK",
    "NVDA": "XLK", "AMD":  "XLK", "QCOM":  "XLK", "INTC": "XLK",
    "MU":   "XLK", "AMAT": "XLK", "AVGO":  "XLK", "ARM":  "XLK",
    "KLAC": "XLK", "ORCL": "XLK", "CRM":   "XLK", "ADBE": "XLK",
    "SNOW": "XLK", "DDOG": "XLK", "NET":   "XLK", "ZS":   "XLK",
    "PLTR": "XLK", "MSTR": "XLK", "XYZ":   "XLK", "SNAP": "XLK",
    "ROKU": "XLK",
    # Consumer Discretionary
    "TSLA": "XLY", "AMZN": "XLY", "NFLX": "XLY", "SHOP": "XLY",
    "UBER": "XLY", "LYFT": "XLY", "RBLX": "XLY", "RIVN": "XLY",
    "HD":   "XLY", "NKE":  "XLY", "SBUX": "XLY", "TGT":  "XLY",
    "DIS":  "XLY",
    # Financials
    "JPM":  "XLF", "GS":   "XLF", "BAC":  "XLF", "MS":   "XLF",
    "WFC":  "XLF", "V":    "XLF", "MA":   "XLF", "SCHW": "XLF",
    "BX":   "XLF", "HOOD": "XLF", "COIN": "XLF",
    # Healthcare
    "JNJ":  "XLV", "UNH":  "XLV", "LLY":  "XLV", "PFE":  "XLV",
    "ABBV": "XLV", "AMGN": "XLV", "MRK":  "XLV", "MRNA": "XLV",
    "GILD": "XLV",
    # Industrials
    "CAT":  "XLI", "BA":   "XLI", "HON":  "XLI", "GE":   "XLI",
    "LMT":  "XLI", "RTX":  "XLI", "DE":   "XLI", "UPS":  "XLI",
    # Energy
    "XOM":  "XLE", "CVX":  "XLE", "COP":  "XLE", "SLB":  "XLE",
    "MPC":  "XLE",
    # Consumer Staples
    "WMT":  "XLP", "COST": "XLP", "MCD":  "XLP",
    # ETFs / indices — map to themselves or closest proxy
    "SPY":  "SPY", "QQQ":  "XLK", "IWM":  "IWM",
    "GLD":  "GLD", "TLT":  "TLT", "SLV":  "GLD",
    "XLE":  "XLE", "XLF":  "XLF", "XLK":  "XLK",
    "XLV":  "XLV", "XLI":  "XLI", "ARKK": "XLK",
}
_DEFAULT_SECTOR = "SPY"


# ── Utilities ──────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def _get_close(batch: pd.DataFrame, sym: str, multi: bool) -> pd.Series:
    """Extract Close series for sym from a yf.download group_by='ticker' result."""
    try:
        if multi:
            return batch[sym]["Close"].dropna()
        return batch["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


def _fetch_live_price(ticker_obj) -> float | None:
    """Best-effort live/intraday price for entry sizing.

    Tries yfinance fast_info first (fastest, intraday quote during market hours).
    Falls back to a 1-minute bar pull, then None. Used as an overlay for the
    `current_price` field in ticker scoring — all model FEATURES still derive
    from prior-daily-close history. The intent is to surface where the stock
    is trading right now so sizing and "% from close" reflect reality, while
    leaving the conviction score untouched.
    """
    try:
        fi = ticker_obj.fast_info
        v = None
        if hasattr(fi, "last_price"):
            v = fi.last_price
        elif isinstance(fi, dict):
            v = fi.get("last_price") or fi.get("lastPrice")
        if v is not None and not pd.isna(v) and float(v) > 0:
            return float(v)
    except Exception:
        pass
    try:
        bars = ticker_obj.history(period="1d", interval="1m", auto_adjust=True)
        if not bars.empty:
            v = float(bars["Close"].dropna().iloc[-1])
            if v > 0:
                return v
    except Exception:
        pass
    return None


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down,  0.0)

    alpha = 1.0 / period
    atr   = tr.ewm(alpha=alpha, adjust=False).mean()
    pdi   = 100 * pd.Series(plus_dm,  index=close.index).ewm(alpha=alpha, adjust=False).mean() / atr
    mdi   = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean() / atr
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx   = dx.ewm(alpha=alpha, adjust=False).mean().dropna()
    return float(adx.iloc[-1]) if len(adx) else np.nan


# ── Macro / regime detection ───────────────────────────────────────────────────

def detect_macro_extended() -> tuple[str, float, dict]:
    """
    Returns (regime_label, macro_flat_bonus, signals_dict).

    macro_flat_bonus is the same scalar added to every ticker's macro z-score.
    It captures market-wide regime (SPY vs 200dma + drawdown), VIX level,
    and TLT rate trend. The ticker-specific sector rotation signal is handled
    separately in _fetch_sector_returns.
    """
    signals: dict = {
        "spy_below_200dma"  : False,
        "spy_drawdown_10pct": False,
        "vix_level"         : None,
        "vix_regime"        : "mid",
        "tlt_3m_return"     : None,
        "spy_3m_return"     : None,
    }
    regime = "Expansion"

    try:
        batch = yf.download(
            ["SPY", "TLT", "^VIX"],
            period="400d", auto_adjust=True, progress=False, group_by="ticker",
        )
        multi = isinstance(batch.columns, pd.MultiIndex)

        spy_close = _get_close(batch, "SPY",  multi)
        tlt_close = _get_close(batch, "TLT",  multi)
        vix_close = _get_close(batch, "^VIX", multi)

        if len(spy_close) >= 200:
            ma200    = spy_close.iloc[-200:].mean()
            cur_spy  = float(spy_close.iloc[-1])
            high_52w = float(spy_close.iloc[-252:].max()) if len(spy_close) >= 252 else float(spy_close.max())
            drawdown = (high_52w - cur_spy) / high_52w

            signals["spy_below_200dma"]   = cur_spy < ma200
            signals["spy_drawdown_10pct"] = drawdown >= SPY_DRAWDOWN_THRESHOLD
            signals["spy_price"]          = round(cur_spy, 2)
            signals["spy_ma200"]          = round(ma200, 2)
            signals["spy_drawdown_pct"]   = round(drawdown * 100, 1)

            if len(spy_close) >= 63:
                signals["spy_3m_return"] = float(spy_close.iloc[-1] / spy_close.iloc[-63] - 1)

        n_firing = sum([signals["spy_below_200dma"], signals["spy_drawdown_10pct"]])
        regime = "Contraction" if n_firing == 2 else ("Caution" if n_firing == 1 else "Expansion")

        if len(vix_close) >= 1:
            vix = float(vix_close.iloc[-1])
            signals["vix_level"]  = round(vix, 1)
            signals["vix_regime"] = "low" if vix < VIX_LOW else ("high" if vix >= VIX_HIGH else "mid")

        if len(tlt_close) >= 63:
            tlt_3m = float(tlt_close.iloc[-1] / tlt_close.iloc[-63] - 1)
            signals["tlt_3m_return"] = round(tlt_3m * 100, 2)

    except Exception:
        pass

    regime_adj = REGIME_ADJ[regime]
    vix_adj    = VIX_ADJ[signals["vix_regime"]]
    tlt_3m_raw = (signals.get("tlt_3m_return") or 0) / 100
    tlt_adj    = 0.10 if tlt_3m_raw > 0 else (-0.10 if tlt_3m_raw < 0 else 0.0)

    macro_flat_bonus = regime_adj + vix_adj + tlt_adj
    signals["regime"] = regime
    return regime, macro_flat_bonus, signals


def _fetch_sector_returns(tickers: list[str], spy_3m: float | None) -> dict[str, float]:
    """
    Batch-download all sector ETFs needed for the universe.
    Returns {etf_symbol: (etf_3m_return - spy_3m_return)}.
    """
    unique_etfs = list(set(SECTOR_MAP.get(t, _DEFAULT_SECTOR) for t in tickers))
    rel: dict[str, float] = {}
    if not unique_etfs:
        return rel

    try:
        raw   = yf.download(unique_etfs, period="200d", auto_adjust=True,
                            progress=False, group_by="ticker")
        multi = isinstance(raw.columns, pd.MultiIndex)
        for etf in unique_etfs:
            try:
                c = _get_close(raw, etf, multi and len(unique_etfs) > 1)
                if len(c) >= 63:
                    rel[etf] = float(c.iloc[-1] / c.iloc[-63] - 1) - (spy_3m or 0)
            except Exception:
                pass
    except Exception:
        pass

    return rel


# ── Per-ticker data ────────────────────────────────────────────────────────────

def _fetch_fundamentals_alpha_vantage(ticker: str) -> dict:
    """Fetch fundamentals via Alpha Vantage REST API.

    Makes up to three endpoint calls per ticker:
      1. OVERVIEW      — market cap, ROE, margins, growth rates, PE, PB,
                         gross margin (GrossProfitTTM / RevenueTTM)
      2. BALANCE_SHEET — debt/equity ratio (totalLiabilities / totalShareholderEquity)
      3. CASH_FLOW     — free cash flow yield (operatingCashflow − |capitalExpenditures|)

    Only called when yfinance is missing one or more critical fundamental fields.
    Results are written to the Supabase fundamental_cache by the calling function;
    this function is purely a data-fetch layer.
    """
    if not ALPHA_VANTAGE_KEY:
        return {}

    import requests as _requests

    _AV_BASE = "https://www.alphavantage.co/query"
    fund: dict = {}

    def _av_get(function: str) -> dict:
        try:
            time.sleep(0.2)
            r = _requests.get(
                _AV_BASE,
                params={"function": function, "symbol": ticker, "apikey": ALPHA_VANTAGE_KEY},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            # Rate-limit and info messages signal a failed call
            if "Information" in data or "Note" in data:
                return {}
            return data
        except Exception:
            return {}

    def _fval(d: dict, key: str) -> float:
        v = d.get(key)
        if v in (None, "None", "-", "", "0"):
            return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    # ── 1. OVERVIEW ────────────────────────────────────────────────────────────
    ov = _av_get("OVERVIEW")
    if ov:
        mcap = _fval(ov, "MarketCapitalization")
        if not np.isnan(mcap):
            fund["market_cap"] = mcap

        for dest, src in [
            ("roe",            "ReturnOnEquityTTM"),
            ("profit_margin",  "ProfitMargin"),
            ("fwd_pe",         "ForwardPE"),
            ("pb",             "PriceToBookRatio"),
            ("eps_qoq_growth", "QuarterlyEarningsGrowthYOY"),
            ("revenue_growth", "QuarterlyRevenueGrowthYOY"),
            ("earnings_growth","QuarterlyEarningsGrowthYOY"),
        ]:
            v = _fval(ov, src)
            if not np.isnan(v):
                fund[dest] = v

        # Gross margin from TTM figures (not available as a single field)
        gp  = _fval(ov, "GrossProfitTTM")
        rev = _fval(ov, "RevenueTTM")
        if not np.isnan(gp) and not np.isnan(rev) and rev > 0:
            fund["gross_margin"] = gp / rev

    # ── 2. BALANCE_SHEET ───────────────────────────────────────────────────────
    bs = _av_get("BALANCE_SHEET")
    annual_bs = (bs.get("annualReports") or [])
    if annual_bs:
        latest = annual_bs[0]
        equity = _fval(latest, "totalShareholderEquity")
        debt   = _fval(latest, "totalLiabilities")
        if not np.isnan(equity) and not np.isnan(debt) and equity != 0:
            fund["debt_equity_neg"] = -debt / equity

    # ── 3. CASH_FLOW ───────────────────────────────────────────────────────────
    cf = _av_get("CASH_FLOW")
    annual_cf = (cf.get("annualReports") or [])
    if annual_cf:
        latest = annual_cf[0]
        op_cf  = _fval(latest, "operatingCashflow")
        capex  = _fval(latest, "capitalExpenditures")
        if not np.isnan(op_cf) and not np.isnan(capex):
            fcf  = op_cf - abs(capex)
            mcap = fund.get("market_cap")
            if mcap and not np.isnan(mcap) and mcap > 0:
                fund["fcf_yield"] = fcf / mcap

    return fund


_CRITICAL_FUND_FIELDS = {"roe", "gross_margin", "revenue_growth", "profit_margin"}


def _fetch_and_cache_fundamentals(ticker: str, t) -> dict:
    """Return all non-technical per-ticker fields, using Supabase cache when fresh.

    Covers:
      - Fundamental pillar inputs (roe, margins, PE, PB, growth rates, FCF yield)
      - Earnings beat history (beat_rate, avg_eps_surprise)
      - Next earnings date as an ISO string (next_earnings_date)

    Strategy:
      1. Check Supabase fundamental_cache — return immediately if within TTL.
      2. Fetch yfinance t.info for all fundamental fields.
      3. If any critical field (roe, gross_margin, revenue_growth, profit_margin)
         is still NaN, call Alpha Vantage (OVERVIEW + BALANCE_SHEET + CASH_FLOW)
         and fill gaps — yfinance values are never overwritten.
      4. Fetch earnings beat history and next earnings date from yfinance.
      5. Write the complete result to the Supabase cache.
      6. Return the result dict.

    Any error in a sub-step is swallowed so the scan keeps running with partial data.
    """
    # 1. Cache hit — fast path
    cached = _db.get_fundamental_cache(ticker)
    if cached:
        return {
            k: v for k, v in cached.items()
            if k not in ("ticker", "fetch_date", "updated_at", "source")
        }

    fund: dict = {}

    # 2. yfinance t.info fundamentals
    try:
        info = t.info or {}

        mcap = info.get("marketCap")
        if mcap is not None:
            try:
                fund["market_cap"] = float(mcap)
            except (TypeError, ValueError):
                pass

        for dest, src in [
            ("roe",            "returnOnEquity"),
            ("gross_margin",   "grossMargins"),
            ("revenue_growth", "revenueGrowth"),
            ("earnings_growth","earningsGrowth"),
            ("profit_margin",  "profitMargins"),
        ]:
            try:
                v = info.get(src)
                if v is not None:
                    fv = float(v)
                    if not np.isnan(fv):
                        fund[dest] = fv
            except (TypeError, ValueError):
                pass

        try:
            fcf   = info.get("freeCashflow")
            mcap_ = fund.get("market_cap")
            if fcf is not None and mcap_ and mcap_ > 0:
                fund["fcf_yield"] = float(fcf) / mcap_
        except (TypeError, ValueError):
            pass

        try:
            de = info.get("debtToEquity")
            if de is not None:
                fund["debt_equity_neg"] = -float(de) / 100
        except (TypeError, ValueError):
            pass

        for dest, src, positive_only in [
            ("pb",            "priceToBook",             True),
            ("fwd_pe",        "forwardPE",               True),
            ("eps_qoq_growth","earningsQuarterlyGrowth", False),
        ]:
            try:
                v = info.get(src)
                if v is not None:
                    fv = float(v)
                    if not np.isnan(fv) and (not positive_only or fv > 0):
                        fund[dest] = fv
            except (TypeError, ValueError):
                pass

    except Exception:
        pass

    # 3. Alpha Vantage supplement — only where yfinance left critical fields empty
    critical_missing = any(
        np.isnan(fund.get(f, np.nan)) for f in _CRITICAL_FUND_FIELDS
    )
    source = "yfinance"
    if critical_missing:
        av = _fetch_fundamentals_alpha_vantage(ticker)
        if av:
            for k, v in av.items():
                yf_val = fund.get(k)
                # Only fill in if yfinance returned nothing usable
                if yf_val is None or (isinstance(yf_val, float) and np.isnan(yf_val)):
                    fund[k] = v
            # AV contributed data — classify as mixed regardless of how many
            # critical fields remain (some tickers genuinely have no public data)
            source = "mixed"

    # 4. Earnings beat history
    try:
        eh = None
        for attr in ("earnings_history", "earnings_dates"):
            try:
                eh = getattr(t, attr, None)
                if eh is not None and isinstance(eh, pd.DataFrame) and len(eh) >= 2:
                    break
                eh = None
            except Exception:
                pass

        if eh is not None and isinstance(eh, pd.DataFrame) and len(eh) >= 2:
            eh.columns = [c.lower().replace(" ", "_") for c in eh.columns]
            actual_col   = next((c for c in eh.columns if "actual"   in c), None)
            estimate_col = next((c for c in eh.columns if "estimate" in c), None)
            if actual_col and estimate_col:
                valid = eh.dropna(subset=[actual_col, estimate_col])
                if len(valid) > 0:
                    beats = (valid[actual_col] > valid[estimate_col]).sum()
                    fund["beat_rate"] = float(beats / len(valid))
                    est_abs  = valid[estimate_col].abs().replace(0, np.nan)
                    surprise = ((valid[actual_col] - valid[estimate_col]) / est_abs).clip(-2, 2)
                    fund["avg_eps_surprise"] = float(surprise.mean())
    except Exception:
        pass

    # 5. Next earnings date — stored as ISO string; earnings_days_out computed at runtime
    try:
        cal     = t.calendar
        next_ed = None
        if isinstance(cal, dict):
            ed      = cal.get("Earnings Date")
            next_ed = ed[0] if isinstance(ed, list) and ed else ed
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            try:
                next_ed = cal.loc["Earnings Date"].iloc[0]
            except Exception:
                pass

        if next_ed is not None:
            if hasattr(next_ed, "date"):
                next_ed = next_ed.date()
            elif isinstance(next_ed, str):
                next_ed = datetime.strptime(next_ed[:10], "%Y-%m-%d").date()
            if isinstance(next_ed, date):
                fund["next_earnings_date"] = next_ed.isoformat()
    except Exception:
        pass

    # 6. Write to cache
    _db.upsert_fundamental_cache(ticker, {**fund, "source": source})

    return fund


def _fetch_ticker_data(ticker: str, sector_rel_3m: float) -> dict | None:
    """Fetch price history, fundamentals, and earnings for one ticker."""
    result: dict = {
        # Technical
        "price_vs_50sma"    : np.nan,
        "price_vs_200sma"   : np.nan,
        "sma50_vs_200"      : np.nan,
        "ret_3m"            : np.nan,
        "ret_6m"            : np.nan,
        "adx_directional"   : np.nan,
        "proximity_52w"     : np.nan,
        # Fundamental
        "roe"               : np.nan,
        "gross_margin"      : np.nan,
        "revenue_growth"    : np.nan,
        "earnings_growth"   : np.nan,
        "fcf_yield"         : np.nan,
        "debt_equity_neg"   : np.nan,   # stored as -D/E so higher = less leverage
        "profit_margin"     : np.nan,
        "fwd_pe"            : np.nan,
        "pb"                : np.nan,
        "market_cap"        : np.nan,
        # Earnings
        "beat_rate"         : np.nan,
        "avg_eps_surprise"  : np.nan,
        "eps_qoq_growth"    : np.nan,
        "earnings_flag"     : False,
        "earnings_days_out" : None,
        "earnings_confidence": 1.0,
        # Macro (sector rel already resolved)
        "sector_rel_3m"     : sector_rel_3m,
        # Meta — current_price is a live/intraday overlay; prior_close is the
        # last fully-settled daily close (what all the technical features are
        # built against); pct_from_close is the intraday move so the user can
        # see how far a name has run before they consider entering.
        "current_price"     : None,
        "prior_close"       : None,
        "pct_from_close"    : 0.0,
    }

    t = yf.Ticker(ticker)

    # ── Technical (isolated) ──────────────────────────────────────────────────
    try:
        hist = t.history(period="400d", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist[["Close", "High", "Low"]].dropna()

        if len(hist) >= 200:
            close = hist["Close"]
            high  = hist["High"]
            low   = hist["Low"]

            sma50  = float(close.iloc[-50:].mean())
            sma200 = float(close.iloc[-200:].mean())
            # `cur` is the prior-daily-close — used by ALL technical features
            # below so the model logic stays exactly as it was.
            cur    = float(close.iloc[-1])
            h52    = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())

            # Live/intraday overlay for entry-price purposes only. Score and
            # features are unchanged.
            live = _fetch_live_price(t)
            display_price = float(live) if (live is not None and live > 0) else cur
            result["current_price"]   = round(display_price, 2)
            result["prior_close"]     = round(cur, 2)
            result["pct_from_close"]  = ((display_price - cur) / cur * 100.0) if cur > 0 else 0.0
            result["price_vs_50sma"]  = (cur - sma50)  / sma50
            result["price_vs_200sma"] = (cur - sma200) / sma200
            result["sma50_vs_200"]    = sma50 / sma200 - 1
            result["proximity_52w"]   = cur / h52

            if len(close) >= 63:
                result["ret_3m"] = float(close.iloc[-1] / close.iloc[-63] - 1)
            if len(close) >= 126:
                result["ret_6m"] = float(close.iloc[-1] / close.iloc[-126] - 1)

            adx = _compute_adx(high, low, close)
            if not np.isnan(adx):
                result["adx_directional"] = adx * (1.0 if cur > sma200 else -1.0)
    except Exception:
        pass

    # ── Fundamentals + earnings history (cached) ──────────────────────────────
    # Single call: checks Supabase cache first (7-day TTL), falls back to
    # yfinance t.info, and supplements with Alpha Vantage REST API for any
    # ticker where yfinance is missing critical fundamental fields.
    _cached = _fetch_and_cache_fundamentals(ticker, t)

    _SKIP = {"next_earnings_date"}  # handled separately below
    for _k, _v in _cached.items():
        if _k in _SKIP or _k not in result:
            continue
        if _v is None or (isinstance(_v, float) and np.isnan(_v)):
            continue
        result[_k] = _v

    # Derive earnings flag from cached absolute date so it stays correct
    # even when the cache was written on a different day.
    _ned = _cached.get("next_earnings_date")
    if _ned:
        try:
            _ned_date = date.fromisoformat(str(_ned))
            _days_out = (_ned_date - date.today()).days
            result["earnings_days_out"] = _days_out
            if 0 <= _days_out <= EARNINGS_FLAG_DAYS:
                result["earnings_flag"]       = True
                result["earnings_confidence"] = EARNINGS_CONFIDENCE
        except Exception:
            pass

    if result["current_price"] is None:
        return None
    return result


# ── Universe scoring ───────────────────────────────────────────────────────────

def score_universe(tickers: list[str],
                   progress_cb=None) -> tuple[pd.DataFrame, str, dict]:
    """
    Score all tickers across four pillars. Returns (ranked DataFrame, regime, signals).

    Direction assigned only when composite clears MIN_LONG_SCORE / MIN_SHORT_SCORE,
    so the model may recommend fewer than MAX_LONG + MAX_SHORT positions.
    """
    regime, macro_flat_bonus, signals = detect_macro_extended()
    spy_3m = signals.get("spy_3m_return")

    sector_ret = _fetch_sector_returns(tickers, spy_3m)
    sector_rel_for = {
        tk: sector_ret.get(SECTOR_MAP.get(tk, _DEFAULT_SECTOR), 0.0)
        for tk in tickers
    }

    rows = []
    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i, len(tickers), ticker)
        data = _fetch_ticker_data(ticker, sector_rel_for.get(ticker, 0.0))
        if data is None:
            continue
        rows.append({"ticker": ticker, **data})

    if not rows:
        return pd.DataFrame(), regime, signals

    df = pd.DataFrame(rows).set_index("ticker")

    # ── Technical pillar ──────────────────────────────────────────────────────
    tech_cols = ["price_vs_50sma", "price_vs_200sma", "sma50_vs_200",
                 "ret_3m", "ret_6m", "adx_directional", "proximity_52w"]
    z_tech_df       = pd.DataFrame({c: _zscore(df[c]) for c in tech_cols})
    df["z_technical"] = z_tech_df.mean(axis=1, skipna=True)

    # ── Fundamental pillar ────────────────────────────────────────────────────
    z_pe = _zscore(-df["fwd_pe"])   # lower P/E → higher score
    z_pb = _zscore(-df["pb"])       # lower P/B → higher score

    large_cap = df["market_cap"] >= SMALL_CAP_THRESH
    value_z   = pd.Series(np.nan, index=df.index)
    value_z[large_cap]  = (z_pe[large_cap].fillna(0) + z_pb[large_cap].fillna(0)) / 2
    value_z[~large_cap] = z_pb[~large_cap].fillna(0)   # no P/E for pre-profit small caps

    fund_cols = ["roe", "gross_margin", "revenue_growth", "earnings_growth",
                 "fcf_yield", "debt_equity_neg", "profit_margin"]
    z_fund_df         = pd.DataFrame({c: _zscore(df[c]) for c in fund_cols})
    z_fund_core       = z_fund_df.mean(axis=1, skipna=True)
    # Blend quality/growth (65%) with value (35%)
    df["z_fundamental"] = z_fund_core.fillna(0) * 0.65 + value_z.fillna(0) * 0.35

    # ── Earnings pillar ───────────────────────────────────────────────────────
    earn_cols = ["beat_rate", "avg_eps_surprise", "eps_qoq_growth"]
    z_earn_df     = pd.DataFrame({c: _zscore(df[c]) for c in earn_cols})
    z_earn_raw    = z_earn_df.mean(axis=1, skipna=True)
    # Penalise near-term earnings risk (score confidence reduced)
    df["z_earnings"] = z_earn_raw.fillna(0) * df["earnings_confidence"]

    # ── Macro pillar ──────────────────────────────────────────────────────────
    # Cross-sectional: sector rotation vs SPY (ticker-specific)
    # + macro_flat_bonus: regime + VIX + TLT (market-wide constant)
    z_sector      = _zscore(df["sector_rel_3m"])
    df["z_macro"] = z_sector.fillna(0) + macro_flat_bonus

    # ── Composite ─────────────────────────────────────────────────────────────
    df["composite_score"] = (
        W_TECHNICAL   * df["z_technical"].fillna(0) +
        W_FUNDAMENTAL * df["z_fundamental"].fillna(0) +
        W_EARNINGS    * df["z_earnings"].fillna(0) +
        W_MACRO       * df["z_macro"].fillna(0)
    )

    # ── Directional regime adjustment (boost longs in Expansion, shorts in Contraction) ──
    # This widens the spread between long and short candidates based on macro regime
    directional_adj = 0.30  # magnitude of directional bias
    if regime == "Expansion":
        df["composite_score"] = df["composite_score"] + df["composite_score"].apply(lambda x: directional_adj if x > 0 else -directional_adj)
    elif regime == "Contraction":
        df["composite_score"] = df["composite_score"] + df["composite_score"].apply(lambda x: -directional_adj if x > 0 else directional_adj)

    df = df.sort_values("composite_score", ascending=False).reset_index()
    n  = len(df)
    df["percentile"] = (n - df.index) / n
    df["rank"]       = df.index + 1
    df["scored_at"]  = str(date.today())
    df["regime"]     = regime

    # ── Direction — slots and ratio scale with macro regime ───────────────────
    max_long  = MAX_LONG_BY_REGIME[regime]
    max_short = MAX_SHORT_BY_REGIME[regime]
    df["direction"] = "NEUTRAL"
    long_idx  = df[df["composite_score"] >= MIN_LONG_SCORE].head(max_long).index
    short_idx = df[df["composite_score"] <= MIN_SHORT_SCORE].tail(max_short).index
    df.loc[long_idx,  "direction"] = "LONG"
    df.loc[short_idx, "direction"] = "SHORT"
    df["max_long"]  = max_long
    df["max_short"] = max_short

    return df, regime, signals


# ── Open-position management ───────────────────────────────────────────────────

def assess_open_positions(positions: list[dict],
                           scored_df: pd.DataFrame) -> list[dict]:
    """
    For each open conviction position, refresh price / P&L and check:
      1. Hard stop hit (≥ 15% adverse move)?
      2. Factor score deteriorated past EXIT_DECILE?
      3. Earnings within EARNINGS_FLAG_DAYS?
    """
    if not positions:
        return positions

    score_map  = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df["composite_score"]))
    pct_map    = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df["percentile"]))
    eflag_map  = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df.get("earnings_flag", pd.Series(False, index=scored_df.index))))
    edays_map  = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df.get("earnings_days_out", pd.Series(None, index=scored_df.index))))
    z_tech_map = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df.get("z_technical",   pd.Series(dtype=float))))
    z_fund_map = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df.get("z_fundamental", pd.Series(dtype=float))))
    z_earn_map = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df.get("z_earnings",    pd.Series(dtype=float))))
    z_mac_map  = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df.get("z_macro",       pd.Series(dtype=float))))

    updated = []
    for pos in positions:
        if pos.get("status") != "open":
            updated.append(pos)
            continue

        ticker      = pos["ticker"]
        direction   = pos["direction"]
        entry_price = pos["entry_price"]

        try:
            hist = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            current_price = round(float(hist["Close"].dropna().iloc[-1]), 2)
        except Exception:
            current_price = entry_price

        pnl_pct = (
            (current_price - entry_price) / entry_price if direction == "LONG"
            else (entry_price - current_price) / entry_price
        )

        pos["current_price"]           = current_price
        pos["pnl_pct"]                 = round(pnl_pct * 100, 2)
        pos["days_held"]               = (date.today() - date.fromisoformat(pos["entry_date"])).days
        pos["current_score"]           = score_map.get(ticker)
        pos["exit_signal"]             = None
        pos["reassess_signal"]         = None
        pos["earnings_flag"]           = eflag_map.get(ticker, False)
        pos["earnings_days_out"]       = edays_map.get(ticker)
        # Backfill score_at_entry for positions opened before score tracking was added.
        # Uses current score as a one-time proxy; preserved on all future saves.
        if pos.get("score_at_entry") is None and score_map.get(ticker) is not None:
            pos["score_at_entry"] = score_map.get(ticker)
        # Current pillar sub-scores (snapshot at each reassessment — enables drift analysis)
        pos["z_technical_current"]     = z_tech_map.get(ticker)
        pos["z_fundamental_current"]   = z_fund_map.get(ticker)
        pos["z_earnings_current"]      = z_earn_map.get(ticker)
        pos["z_macro_current"]         = z_mac_map.get(ticker)

        if pnl_pct <= -HARD_STOP_PCT:
            pos["exit_signal"] = "HARD_STOP"
        else:
            current_score = score_map.get(ticker)
            # Primary exit gate: score dropped back below the threshold that
            # qualified it for entry. This fires before the percentile check
            # so a name like SLB (LONG, score -0.35) gets flagged immediately.
            if current_score is not None:
                if direction == "LONG" and current_score < MIN_LONG_SCORE:
                    pos["reassess_signal"] = (
                        f"Score {current_score:+.3f} below LONG threshold ({MIN_LONG_SCORE:+.2f})"
                    )
                elif direction == "SHORT" and current_score > MIN_SHORT_SCORE:
                    pos["reassess_signal"] = (
                        f"Score {current_score:+.3f} above SHORT threshold ({MIN_SHORT_SCORE:+.2f})"
                    )

            # Secondary exit gate: percentile rank deterioration
            if pos.get("reassess_signal") is None:
                pct = pct_map.get(ticker)
                if pct is not None:
                    if direction == "LONG" and pct < EXIT_DECILE:
                        pos["reassess_signal"] = f"Fallen to {pct:.0%} pct — thesis weakened"
                    elif direction == "SHORT" and pct > (1 - EXIT_DECILE):
                        pos["reassess_signal"] = f"Risen to {pct:.0%} pct — thesis weakened"

            # Earnings trim: unrealized gain > threshold + earnings imminent
            edays = edays_map.get(ticker)
            if (pos.get("reassess_signal") is None
                    and pnl_pct >= EARNINGS_TRIM_GAIN
                    and edays is not None
                    and edays <= EARNINGS_TRIM_DAYS):
                pos["reassess_signal"] = (
                    f"EARNINGS_TRIM — up {pnl_pct*100:.1f}% with earnings in {edays}d"
                )

        updated.append(pos)
    return updated


# ── Weekly performance tracking ────────────────────────────────────────────────

def calculate_weekly_performance(positions: list[dict]) -> dict | None:
    """
    Calculate weekly portfolio performance vs SPY, weighted by capital at risk.
    Uses end-of-week prices for open positions and exit price for closed positions.
    """
    if not positions:
        return None

    active = [p for p in positions if p.get("status") in ("open", "closed")]
    if not active:
        return None

    dates = []
    for p in active:
        try:
            dates.append(date.fromisoformat(p["entry_date"]))
            if p.get("status") == "closed" and p.get("exit_date"):
                dates.append(date.fromisoformat(p["exit_date"]))
        except (ValueError, TypeError):
            pass

    if not dates:
        return None

    start_date = min(dates)
    end_date = max(dates)

    # If there are open positions, extend end_date to today
    if any(p.get("status") == "open" for p in active):
        end_date = max(end_date, date.today())

    # Download SPY and all ticker data (extend range to ensure we get full trading weeks)
    try:
        spy_data = yf.download("SPY", start=start_date - pd.Timedelta(days=7), end=end_date + pd.Timedelta(days=1),
                               auto_adjust=True, progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        spy_close = spy_data["Close"] if "Close" in spy_data.columns else None
        if spy_close is None or len(spy_close) == 0:
            return None
    except Exception:
        return None

    # Download all position tickers (extend range to ensure we get full trading weeks)
    tickers_to_fetch = list(set(p["ticker"] for p in active))
    ticker_prices = {}
    try:
        ticker_data = yf.download(tickers_to_fetch, start=start_date - pd.Timedelta(days=7), end=end_date + pd.Timedelta(days=1),
                                  auto_adjust=True, progress=False)

        # Handle different data structure depending on number of tickers
        for tk in tickers_to_fetch:
            try:
                if len(tickers_to_fetch) == 1:
                    # Single ticker: ticker_data is a DataFrame with Close column
                    ticker_prices[tk] = ticker_data["Close"].dropna()
                else:
                    # Multiple tickers: ticker_data has MultiIndex columns
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_prices[tk] = ticker_data[("Close", tk)].dropna() if ("Close", tk) in ticker_data.columns else ticker_data[(tk, "Close")].dropna()
                    else:
                        ticker_prices[tk] = ticker_data[tk].dropna()
            except (KeyError, TypeError):
                pass
    except Exception:
        pass

    # Calculate total portfolio capital (sum of all position costs)
    total_portfolio_capital = sum(float(p.get("cost", p.get("shares", 0) * p["entry_price"])) for p in active)

    # Build weekly performance - track cumulative P&L to handle partial weeks correctly
    weeks_data = []
    # Get the Monday of the week AFTER the first position entry
    days_until_monday = (7 - start_date.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    first_week_monday = start_date + pd.Timedelta(days=days_until_monday)
    current_week_monday = first_week_monday

    prev_week_cumul_pnl = 0.0  # Track cumulative P&L from previous week
    prev_week_spy_price = None  # Track SPY close from previous week

    while current_week_monday <= end_date:
        week_end = current_week_monday + pd.Timedelta(days=6)
        week_dates = pd.date_range(current_week_monday, week_end, freq='D')
        week_dates_list = [d.date() for d in week_dates if d.date() <= end_date]

        if not week_dates_list:
            current_week_monday += pd.Timedelta(days=7)
            continue

        # Calculate portfolio's cumulative P&L from entry to end of this week
        cumul_pnl = 0.0
        week_has_positions = False

        for pos in active:
            try:
                entry_date = date.fromisoformat(pos["entry_date"])
                exit_date = None
                if pos.get("exit_date"):
                    exit_date = date.fromisoformat(pos["exit_date"])

                # Check if position is active in this week
                if week_end < entry_date or (exit_date and week_dates_list[0] > exit_date):
                    continue

                week_has_positions = True

                entry_price = float(pos["entry_price"])
                shares = float(pos.get("shares", 0))
                direction = pos["direction"]

                # Get end-of-week price
                if exit_date and week_end >= exit_date:
                    week_price = float(pos.get("exit_price", entry_price))
                else:
                    # Use latest available price in this week
                    tk = pos["ticker"]
                    if tk in ticker_prices:
                        week_close_dates = [d for d in ticker_prices[tk].index if d.date() in set(week_dates_list)]
                        if week_close_dates:
                            week_price = float(ticker_prices[tk].loc[week_close_dates[-1]])
                        else:
                            week_price = entry_price
                    else:
                        week_price = entry_price

                # Calculate cumulative P&L from entry price
                if direction == "LONG":
                    pos_pnl = (week_price - entry_price) * shares
                elif direction == "SHORT":
                    pos_pnl = (entry_price - week_price) * shares
                else:
                    pos_pnl = 0.0

                cumul_pnl += pos_pnl

            except (ValueError, TypeError, KeyError, AttributeError):
                continue

        # Only include weeks with active positions
        if not week_has_positions:
            current_week_monday += pd.Timedelta(days=7)
            continue

        # Week's P&L = cumulative change from entry minus what was already earned last week
        week_pnl = cumul_pnl - prev_week_cumul_pnl
        prev_week_cumul_pnl = cumul_pnl

        # Calculate week return as % of total portfolio capital
        week_return = week_pnl / total_portfolio_capital if total_portfolio_capital > 0 else 0.0

        # Calculate SPY return for this week (including partial weeks)
        matching_dates = [d for d in spy_close.index if d.date() in set(week_dates_list)]
        if len(matching_dates) >= 1:
            current_spy_price = float(spy_close[matching_dates[-1]])  # Last price in this week
            if prev_week_spy_price is None:
                # First week: use opening price of the week
                spy_week_return = (current_spy_price - spy_close[matching_dates[0]]) / spy_close[matching_dates[0]]
            else:
                # Subsequent weeks: use previous week's closing price
                spy_week_return = (current_spy_price - prev_week_spy_price) / prev_week_spy_price
        else:
            spy_week_return = 0.0
            current_spy_price = prev_week_spy_price

        weeks_data.append({
            "week_monday": current_week_monday,
            "portfolio_return": week_return,
            "spy_return": spy_week_return,
        })

        # Update prev_week_spy_price for next iteration
        if len(matching_dates) >= 1:
            prev_week_spy_price = float(spy_close[matching_dates[-1]])

        current_week_monday += pd.Timedelta(days=7)

    if not weeks_data:
        return None

    # Calculate cumulative returns (simple sum since each week's return is % of total capital)
    cumulative_portfolio = []
    cumulative_spy = []
    cum_port = 0.0
    cum_spy = 0.0

    for w in weeks_data:
        cum_port += w["portfolio_return"]
        cum_spy += w["spy_return"]
        cumulative_portfolio.append(cum_port)
        cumulative_spy.append(cum_spy)

    return {
        "weeks": [w["week_monday"].strftime("%Y-%m-%d") for w in weeks_data],
        "portfolio_returns": [w["portfolio_return"] for w in weeks_data],
        "spy_returns": [w["spy_return"] for w in weeks_data],
        "cumulative_portfolio": cumulative_portfolio,
        "cumulative_spy": cumulative_spy,
    }


# ── Storage passthrough ────────────────────────────────────────────────────────

def load_conviction_positions() -> list[dict]:
    return _db.load_conviction_positions()


def save_conviction_positions(positions: list[dict]) -> None:
    _db.save_conviction_positions(positions)


def run_conviction_scan(tickers: list[str], progress_cb=None) -> pd.DataFrame:
    """Convenience wrapper used by app.py."""
    df, _, _ = score_universe(tickers, progress_cb)
    return df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SUMMARY_FILE = Path(__file__).parent / "ticker_summary.csv"
    if not SUMMARY_FILE.exists():
        print("ERROR: ticker_summary.csv not found.")
        sys.exit(1)

    tickers = pd.read_csv(SUMMARY_FILE)["Ticker"].tolist()
    print(f"\nScoring {len(tickers)} tickers — technical · fundamental · earnings · macro…")

    def _cb(i, n, t):
        print(f"  [{i+1}/{n}] {t:<8}", end="\r")

    df, regime, signals = score_universe(tickers, _cb)
    print()

    if df.empty:
        print("No results.")
        sys.exit(0)

    vix      = signals.get("vix_level", "—")
    vix_r    = signals.get("vix_regime", "—")
    tlt_ret  = signals.get("tlt_3m_return", "—")
    spy_ret  = signals.get("spy_3m_return")
    spy_str  = f"{spy_ret*100:.1f}%" if spy_ret is not None else "—"

    print(f"{'─'*84}")
    print(f"  Macro Regime : {regime}")
    print(f"  VIX          : {vix} ({vix_r})")
    print(f"  TLT 3m       : {tlt_ret}%   |   SPY 3m : {spy_str}")
    print(f"{'─'*84}\n")

    longs     = df[df["direction"] == "LONG"]
    shorts    = df[df["direction"] == "SHORT"]
    max_long  = MAX_LONG_BY_REGIME[regime]
    max_short = MAX_SHORT_BY_REGIME[regime]
    print(f"  LONG  : {len(longs)} / {max_long} slots filled   "
          f"(threshold ≥ {MIN_LONG_SCORE:+.2f})")
    print(f"  SHORT : {len(shorts)} / {max_short} slots filled  "
          f"(threshold ≤ {MIN_SHORT_SCORE:+.2f})\n")

    show_cols = ["rank", "ticker", "direction", "composite_score",
                 "z_technical", "z_fundamental", "z_earnings", "z_macro",
                 "current_price", "earnings_flag"]
    show_cols = [c for c in show_cols if c in df.columns]

    print("TOP 15:")
    pd.set_option("display.float_format", "{:+.3f}".format)
    print(df[show_cols].head(15).to_string(index=False))
    print("\nBOTTOM 5 (SHORT candidates):")
    print(df[show_cols].tail(5).to_string(index=False))
