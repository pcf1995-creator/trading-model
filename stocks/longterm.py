"""
stocks/longterm.py — Factor-based long/short portfolio scanner

Scores tickers across three factors:
  • Momentum  — 12-1 month return (intermediate), winsorized at 5th/95th pct,
                minus a small penalty on 1-month return (short-term reversal)
  • Quality   — ROE + gross margin (profitable, efficient businesses)
  • Value     — inverse P/B for all; inverse forward P/E only for ≥$5B market cap
                (small caps excluded from P/E to avoid penalising pre-profit growth)

  Plus a macro regime overlay:
  • Regime    — detects Contraction / Caution / Expansion using SPY vs 200dma
                and SPY drawdown from 52-week high. In Contraction/Caution,
                defensive tickers receive a score bonus to offset value penalties.

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

# ── Macro regime overlay ──────────────────────────────────────────────────────
# Tickers considered defensive (benefit in slowdowns / recessions)
DEFENSIVE_TICKERS = {
    "WMT", "COST", "MCD", "JNJ", "PFE", "ABBV", "AMGN",
    "UNH", "GLD", "TLT", "KO", "PG", "CVX", "XOM",
}

# Score bonus added to defensive tickers based on regime
MACRO_BONUS = {
    "Contraction": 0.40,   # both signals firing — full defensive tilt
    "Caution":     0.20,   # one signal firing — partial tilt
    "Expansion":   0.00,   # no adjustment
}

# Thresholds for regime signals
SPY_DRAWDOWN_THRESHOLD = 0.10   # 10% off 52-week high triggers one signal
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


def detect_macro_regime() -> tuple[str, dict]:
    """
    Returns (regime_label, signals_dict) using two SPY-based signals:
      1. SPY price vs 200-day MA  (below = risk-off)
      2. SPY drawdown from 52-week high  (> 10% = stress)

    Contraction : both signals firing
    Caution     : one signal firing
    Expansion   : neither firing
    """
    signals = {"spy_below_200dma": False, "spy_drawdown_10pct": False}
    try:
        spy = yf.Ticker("SPY")
        df  = spy.history(period="400d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()

        if len(close) >= 200:
            ma200        = close.iloc[-200:].mean()
            current      = float(close.iloc[-1])
            high_52w     = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
            drawdown     = (high_52w - current) / high_52w

            signals["spy_below_200dma"]   = current < ma200
            signals["spy_drawdown_10pct"] = drawdown >= SPY_DRAWDOWN_THRESHOLD
            signals["spy_price"]          = round(current, 2)
            signals["spy_ma200"]          = round(ma200, 2)
            signals["spy_drawdown_pct"]   = round(drawdown * 100, 1)
    except Exception:
        pass

    n_firing = sum([signals["spy_below_200dma"], signals["spy_drawdown_10pct"]])
    if n_firing == 2:
        regime = "Contraction"
    elif n_firing == 1:
        regime = "Caution"
    else:
        regime = "Expansion"

    return regime, signals


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


def score_universe(tickers: list[str], progress_cb=None) -> tuple[pd.DataFrame, str, dict]:
    """
    Score all tickers. Returns (ranked DataFrame, regime_label, regime_signals).
    DataFrame includes LONG/SHORT/NEUTRAL labels and macro adjustment column.
    """
    regime, signals = detect_macro_regime()
    bonus           = MACRO_BONUS[regime]

    rows = []
    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i, len(tickers), ticker)
        data = _fetch_ticker_data(ticker)
        if data is None:
            continue
        rows.append({"ticker": ticker, **data})

    if not rows:
        return pd.DataFrame(), regime, signals

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

    # ── Macro regime adjustment ──
    df["is_defensive"]   = df.index.isin(DEFENSIVE_TICKERS)
    df["macro_adj"]      = df["is_defensive"].apply(lambda x: bonus if x else 0.0)
    df["composite_score"] = df["composite_score"] + df["macro_adj"]

    df = df.sort_values("composite_score", ascending=False).reset_index()

    n = len(df)
    df["percentile"] = (n - df.index) / n   # 1.0 = top scorer

    df["direction"] = "NEUTRAL"
    df.loc[df.index < MAX_LONG,       "direction"] = "LONG"
    df.loc[df.index >= n - MAX_SHORT, "direction"] = "SHORT"

    df["rank"]      = df.index + 1
    df["scored_at"] = str(date.today())
    df["regime"]    = regime

    return df, regime, signals


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
    """Convenience wrapper used by app.py. Returns just the DataFrame."""
    df, _, _ = score_universe(tickers, progress_cb)
    return df


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

    df, regime, signals = score_universe(tickers, _cb)
    print()

    if df.empty:
        print("No results.")
        sys.exit(0)

    # ── Regime summary ──
    spy_price   = signals.get("spy_price", "—")
    spy_ma200   = signals.get("spy_ma200", "—")
    drawdown    = signals.get("spy_drawdown_pct", "—")
    below_flag  = "YES" if signals.get("spy_below_200dma") else "no"
    drawdown_flag = "YES" if signals.get("spy_drawdown_10pct") else "no"
    bonus       = MACRO_BONUS[regime]

    regime_icons = {"Contraction": "⚠ ", "Caution": "~ ", "Expansion": "  "}
    print(f"{'─'*78}")
    print(f"  Macro Regime : {regime_icons[regime]}{regime}")
    print(f"  SPY          : ${spy_price}  |  200dma: ${spy_ma200}  |  Below 200dma: {below_flag}")
    print(f"  Drawdown     : {drawdown}% from 52w high  |  >10% threshold: {drawdown_flag}")
    if bonus > 0:
        print(f"  Defensive bonus applied: +{bonus:.2f} to {', '.join(sorted(DEFENSIVE_TICKERS & set(df['ticker'])))}")
    print(f"{'─'*78}\n")

    print(f"\n{'Rank':>4}  {'Ticker':<8}  {'Score':>7}  {'Adj':>5}  {'Dir':<7}  "
          f"{'Mom 12-1':>9}  {'1m ret':>7}  {'ROE':>7}  {'Margin':>8}  {'P/B':>5}")
    print("-" * 86)
    for _, row in df.iterrows():
        mom   = f"{row['mom_12_1']*100:+.1f}%"    if pd.notna(row["mom_12_1"])     else "  —  "
        m1    = f"{row['mom_1m']*100:+.1f}%"       if pd.notna(row["mom_1m"])       else "  —  "
        roe   = f"{row['roe']*100:.1f}%"            if pd.notna(row["roe"])          else "  —  "
        gm    = f"{row['gross_margin']*100:.1f}%"  if pd.notna(row["gross_margin"]) else "  —  "
        pb    = f"{row['pb']:.2f}x"                if pd.notna(row["pb"])           else "  —  "
        adj   = f"+{row['macro_adj']:.2f}" if row["macro_adj"] > 0 else "     "
        tag   = "★ LONG " if row["direction"] == "LONG" else ("✗ SHORT" if row["direction"] == "SHORT" else "      ")
        print(f"{int(row['rank']):>4}  {row['ticker']:<8}  {row['composite_score']:>7.3f}  "
              f"{adj:>5}  {tag}  {mom:>9}  {m1:>7}  {roe:>7}  {gm:>8}  {pb:>5}")

    # ── Open positions P&L ────────────────────────────────────────────────────
    positions = load_lt_positions()
    open_pos  = [p for p in positions if p.get("status") == "open"]

    if open_pos:
        print(f"\n{'─'*78}")
        print(f"  OPEN POSITIONS  ({len(open_pos)} trades)")
        print(f"{'─'*78}")
        print(f"  {'Dir':<5}  {'Ticker':<6}  {'Entry':>8}  {'Current':>8}  "
              f"{'P&L%':>7}  {'P&L$':>8}  {'Days':>5}  {'Alert'}")
        print(f"  {'-'*74}")

        updated = assess_open_positions(open_pos, df)
        save_lt_positions(updated + [p for p in positions if p.get("status") != "open"])

        total_pnl = 0.0
        for pos in updated:
            current = pos.get("current_price", pos["entry_price"])
            pnl_pct = pos.get("pnl_pct", 0.0)
            pnl_usd = round(pos["cost"] * pnl_pct / 100, 2)
            total_pnl += pnl_usd
            days    = pos.get("days_held", 0)
            alert   = pos.get("exit_signal") or pos.get("reassess_signal") or ""
            flag    = "🛑 " if pos.get("exit_signal") else ("⚠  " if pos.get("reassess_signal") else "")
            tag     = "LONG " if pos["direction"] == "LONG" else "SHORT"
            sign    = "+" if pnl_pct >= 0 else ""
            print(f"  {tag:<5}  {pos['ticker']:<6}  ${pos['entry_price']:>7}  ${current:>7}  "
                  f"{sign}{pnl_pct:>6.2f}%  ${pnl_usd:>+8.2f}  {days:>5}d  {flag}{alert}")

        sign = "+" if total_pnl >= 0 else ""
        print(f"  {'-'*74}")
        print(f"  {'Total P&L':>47}  ${total_pnl:>+8.2f}")
