"""
stocks/longterm.py — Factor-based long/short portfolio scanner

Scores tickers across three factors:
  • Momentum  — 12-1 month return (intermediate), winsorized at 5th/95th pct,
                minus a small penalty on 1-month return (short-term reversal)
  • Quality   — ROE + gross margin (profitable, efficient businesses)
  • Value     — inverse P/B for all; inverse forward P/E only for ≥$5B market cap
                (small caps excluded from P/E to avoid penalising pre-profit growth)

Top N by composite score → LONG candidates.
Bottom N by composite score → SHORT candidates.
Exits are reassessment-based (monthly re-score) + 15% hard stop.
"""

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
LT_POSITIONS_FILE  = Path(__file__).parent.parent / "lt_positions.json"
MAX_LONG           = 5       # max long positions recommended
MAX_SHORT          = 5       # max short positions recommended
HARD_STOP_PCT      = 0.15    # 15% hard stop (long down OR short adverse move)
SMALL_CAP_THRESH   = 5e9     # $5B — skip forward P/E penalty below this

# Factor weights
W_MOM_INTERMEDIATE = 0.40    # 12-1 month momentum (positive)
W_MOM_REVERSAL     = 0.10    # 1-month return penalty (subtracted)
W_QUALITY          = 0.30
W_VALUE            = 0.30

# Winsorisation bounds for momentum
MOM_WINSOR_LOW  = 0.05       # 5th percentile
MOM_WINSOR_HIGH = 0.95       # 95th percentile

# Exit threshold — leave top/bottom X% before flagging reassessment
EXIT_DECILE = 0.30
# ─────────────────────────────────────────────────────────────────────────────


def load_lt_positions() -> list[dict]:
    if LT_POSITIONS_FILE.exists():
        with open(LT_POSITIONS_FILE) as f:
            return json.load(f)
    return []


def save_lt_positions(positions: list[dict]) -> None:
    with open(LT_POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def _zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def _winsorise(series: pd.Series, low: float = 0.05, high: float = 0.95) -> pd.Series:
    lo = series.quantile(low)
    hi = series.quantile(high)
    return series.clip(lo, hi)


def _fetch_ticker_data(ticker: str) -> dict | None:
    """
    Single yfinance call per ticker: price history + fundamentals.
    Returns None if data is insufficient.
    """
    result = {
        "mom_12_1"    : np.nan,
        "mom_1m"      : np.nan,
        "roe"         : np.nan,
        "gross_margin": np.nan,
        "fwd_pe"      : np.nan,
        "pb"          : np.nan,
        "market_cap"  : np.nan,
        "current_price": None,
    }

    try:
        t = yf.Ticker(ticker)

        # ── Price history ──
        df = t.history(period="400d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna()

        if len(df) >= 252:
            close = df["Close"]
            # Intermediate momentum: return from 252 days ago to 22 days ago
            result["mom_12_1"] = float(close.iloc[-22] / close.iloc[-252] - 1)
            # Short-term (1-month) return for reversal penalty
            result["mom_1m"]   = float(close.iloc[-1] / close.iloc[-22] - 1)
            result["current_price"] = round(float(close.iloc[-1]), 2)

        # ── Fundamentals ──
        info = t.info
        roe          = info.get("returnOnEquity")
        gross_margin = info.get("grossMargins")
        fwd_pe       = info.get("forwardPE")
        pb           = info.get("priceToBook")
        market_cap   = info.get("marketCap")

        if roe          is not None: result["roe"]          = float(roe)
        if gross_margin is not None: result["gross_margin"] = float(gross_margin)
        if market_cap   is not None: result["market_cap"]   = float(market_cap)
        if pb is not None and float(pb) > 0:
            result["pb"] = float(pb)
        if fwd_pe is not None and float(fwd_pe) > 0:
            result["fwd_pe"] = float(fwd_pe)

    except Exception:
        pass

    # Need at least a price signal to be useful
    if pd.isna(result["mom_12_1"]):
        return None

    return result


def score_universe(tickers: list[str], progress_cb=None) -> pd.DataFrame:
    """
    Score all tickers. Returns a ranked DataFrame with LONG/SHORT/NEUTRAL labels.
    """
    rows = []
    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i, len(tickers), ticker)
        data = _fetch_ticker_data(ticker)
        if data is None:
            continue
        rows.append({"ticker": ticker, **data})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")

    # ── Momentum factor ──
    # Winsorise intermediate momentum before z-scoring
    mom_w = _winsorise(df["mom_12_1"].dropna(), MOM_WINSOR_LOW, MOM_WINSOR_HIGH)
    df["mom_12_1_w"] = df["mom_12_1"].clip(
        df["mom_12_1"].quantile(MOM_WINSOR_LOW),
        df["mom_12_1"].quantile(MOM_WINSOR_HIGH),
    )
    z_mom_intermediate = _zscore(df["mom_12_1_w"])
    z_mom_reversal     = _zscore(df["mom_1m"])    # high short-term → penalise

    # Net momentum z-score: intermediate positive, reversal negative
    # Scale weights so they still sum to W_MOM_INTERMEDIATE+W_MOM_REVERSAL
    total_mom_w = W_MOM_INTERMEDIATE + W_MOM_REVERSAL
    df["z_mom"] = (
        (W_MOM_INTERMEDIATE * z_mom_intermediate.fillna(0) -
         W_MOM_REVERSAL     * z_mom_reversal.fillna(0))
        / total_mom_w
    )

    # ── Quality factor ──
    z_roe = _zscore(df["roe"])
    z_gm  = _zscore(df["gross_margin"])
    df["z_quality"] = z_roe.add(z_gm, fill_value=0) / 2

    # ── Value factor (P/E excluded for small caps) ──
    # For tickers ≥$5B: value = mean(−P/E z-score, −P/B z-score)
    # For tickers  <$5B: value = −P/B z-score only
    z_pe = _zscore(-df["fwd_pe"])   # lower P/E = higher z
    z_pb = _zscore(-df["pb"])       # lower P/B = higher z

    large_cap_mask = df["market_cap"] >= SMALL_CAP_THRESH
    df["z_value"] = np.nan
    # Large cap: average P/E and P/B z-scores
    df.loc[large_cap_mask,  "z_value"] = (
        z_pe[large_cap_mask].fillna(0) + z_pb[large_cap_mask].fillna(0)
    ) / 2
    # Small cap: P/B only
    df.loc[~large_cap_mask, "z_value"] = z_pb[~large_cap_mask].fillna(0)

    # ── Composite score ──
    df["composite_score"] = (
        total_mom_w * df["z_mom"].fillna(0) +
        W_QUALITY   * df["z_quality"].fillna(0) +
        W_VALUE     * df["z_value"].fillna(0)
    )

    df = df.sort_values("composite_score", ascending=False).reset_index()

    n = len(df)
    df["percentile"] = (n - df.index) / n   # 1.0 = top scorer

    df["direction"] = "NEUTRAL"
    df.loc[df.index < MAX_LONG,       "direction"] = "LONG"
    df.loc[df.index >= n - MAX_SHORT, "direction"] = "SHORT"

    df["rank"]      = df.index + 1
    df["scored_at"] = str(date.today())

    return df


def assess_open_positions(positions: list[dict], scored_df: pd.DataFrame) -> list[dict]:
    """
    For each open long-term position, update current price/P&L and check:
      1. Hard stop hit?
      2. Factor score has deteriorated past EXIT_DECILE threshold?
    Returns positions with exit_signal / reassess_signal fields updated.
    """
    if not positions:
        return positions

    score_map = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df["composite_score"]))
    pct_map   = {} if scored_df.empty else dict(zip(scored_df["ticker"], scored_df["percentile"]))

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

        pos["current_price"]  = current_price
        pos["pnl_pct"]        = round(pnl_pct * 100, 2)
        pos["days_held"]      = (date.today() - date.fromisoformat(pos["entry_date"])).days
        pos["current_score"]  = score_map.get(ticker)
        pos["exit_signal"]    = None
        pos["reassess_signal"] = None

        if pnl_pct <= -HARD_STOP_PCT:
            pos["exit_signal"] = "HARD_STOP"
        else:
            pct = pct_map.get(ticker)
            if pct is not None:
                if direction == "LONG" and pct < EXIT_DECILE:
                    pos["reassess_signal"] = f"Fallen to {pct:.0%} pct — thesis weakened"
                elif direction == "SHORT" and pct > (1 - EXIT_DECILE):
                    pos["reassess_signal"] = f"Risen to {pct:.0%} pct — thesis weakened"

        updated.append(pos)

    return updated


def run_lt_scan(tickers: list[str], progress_cb=None) -> pd.DataFrame:
    """Convenience wrapper used by app.py."""
    return score_universe(tickers, progress_cb)


if __name__ == "__main__":
    import sys

    SUMMARY_FILE = Path(__file__).parent / "ticker_summary.csv"
    if not SUMMARY_FILE.exists():
        print("ERROR: ticker_summary.csv not found. Run features.py first.")
        sys.exit(1)

    tickers = pd.read_csv(SUMMARY_FILE)["Ticker"].tolist()
    print(f"\nScoring {len(tickers)} tickers across momentum, quality, value factors...")

    def _cb(i, n, t):
        print(f"  [{i+1}/{n}] {t:<8}", end="\r")

    df = run_lt_scan(tickers, _cb)
    print()

    if df.empty:
        print("No results.")
        sys.exit(0)

    print(f"\n{'Rank':>4}  {'Ticker':<8}  {'Score':>7}  {'Dir':<7}  "
          f"{'Mom 12-1':>9}  {'1m ret':>7}  {'ROE':>7}  {'Margin':>8}  {'P/B':>5}")
    print("-" * 78)
    for _, row in df.iterrows():
        mom   = f"{row['mom_12_1']*100:+.1f}%"    if pd.notna(row["mom_12_1"])     else "  —  "
        m1    = f"{row['mom_1m']*100:+.1f}%"       if pd.notna(row["mom_1m"])       else "  —  "
        roe   = f"{row['roe']*100:.1f}%"            if pd.notna(row["roe"])          else "  —  "
        gm    = f"{row['gross_margin']*100:.1f}%"  if pd.notna(row["gross_margin"]) else "  —  "
        pb    = f"{row['pb']:.2f}x"                if pd.notna(row["pb"])           else "  —  "
        tag   = "★ LONG " if row["direction"] == "LONG" else ("✗ SHORT" if row["direction"] == "SHORT" else "      ")
        print(f"{int(row['rank']):>4}  {row['ticker']:<8}  {row['composite_score']:>7.3f}  "
              f"{tag}  {mom:>9}  {m1:>7}  {roe:>7}  {gm:>8}  {pb:>5}")
