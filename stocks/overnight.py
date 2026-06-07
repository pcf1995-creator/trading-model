"""
stocks/overnight.py — Overnight drift backtest and scanner

The overnight drift is the documented tendency of stocks to earn positive
returns close-to-open while losing intraday (open-to-close).

Reference: Lou, Polk & Skouras (2019) "A Tug of War: Overnight Versus
Intraday Expected Returns."
"""

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PERIOD_DAYS = {"1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825, "10Y": 3650}

# Curated universe of liquid names — used when no summary CSV is available
DEFAULT_TICKERS = [
    "MU", "NVDA", "AMD", "INTC", "AMAT", "LRCX", "KLAC", "MRVL", "AVGO", "QCOM",
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "CRM", "ORCL",
    "PLTR", "RBLX", "SHOP", "SNOW", "DDOG", "NET", "ZS", "CRWD", "MSTR",
    "JPM", "GS", "BAC", "MS", "C", "XOM", "CVX", "LLY", "UNH", "COST",
    "WMT", "HD", "NKE", "MCD", "V", "MA", "JNJ", "PFE", "ABBV",
]


def compute_overnight_stats(
    ticker: str,
    period_days: int = 1825,
    cost_bps: float = 3.0,
) -> dict | None:
    """
    Download OHLC history and return overnight vs intraday statistics.

    Returns a dict with both scalar summary fields and '_'-prefixed pandas
    Series for charting (dates, cumulative return curves).
    Returns None if there is insufficient data.
    """
    end   = date.today()
    start = end - timedelta(days=period_days + 90)

    try:
        raw = yf.download(ticker, start=str(start), end=str(end),
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "Close"]].dropna()
    except Exception:
        return None

    if len(df) < 60:
        return None

    # Core return series
    df = df.copy()
    df["overnight"] = df["Open"] / df["Close"].shift(1) - 1   # close[t-1] → open[t]
    df["intraday"]  = df["Close"] / df["Open"] - 1             # open[t] → close[t]
    df["daily"]     = df["Close"] / df["Close"].shift(1) - 1
    df = df.dropna()

    if len(df) < 50:
        return None

    on = df["overnight"]
    id_ = df["intraday"]

    on_mean_bps = on.mean() * 10_000
    id_mean_bps = id_.mean() * 10_000

    on_sharpe = (on.mean() / on.std() * np.sqrt(252)) if on.std() > 0 else 0.0
    id_sharpe = (id_.mean() / id_.std() * np.sqrt(252)) if id_.std() > 0 else 0.0

    # t-test: overnight mean significantly != 0
    try:
        from scipy import stats as _st
        t_stat, p_val = float(_st.ttest_1samp(on, 0).statistic), float(_st.ttest_1samp(on, 0).pvalue)
    except Exception:
        n = len(on)
        t_stat = float(on.mean() / (on.std() / np.sqrt(n))) if on.std() > 0 else 0.0
        p_val  = float("nan")

    on_win_rate = float((on > 0).mean() * 100)

    cost_daily = 2 * cost_bps / 10_000
    on_net = on - cost_daily
    on_net_sharpe = (on_net.mean() / on_net.std() * np.sqrt(252)) if on_net.std() > 0 else 0.0

    breakeven_bps = max(0.0, float(on.mean() * 10_000 / 2))

    # Cumulative return series (multiplied up from 1.0)
    on_cum     = (1 + on).cumprod()
    on_net_cum = (1 + on_net).cumprod()
    id_cum     = (1 + id_).cumprod()
    bah_cum    = (1 + df["daily"]).cumprod()

    # Monthly overnight returns for heatmap
    monthly = (
        on.groupby([on.index.year, on.index.month])
          .apply(lambda x: float((1 + x).prod() - 1) * 100)
    )

    return {
        # Scalars
        "ticker"            : ticker,
        "n_days"            : len(df),
        "cur_price"         : round(float(df["Close"].iloc[-1]), 2),
        "overnight_mean_bps": round(on_mean_bps, 2),
        "intraday_mean_bps" : round(id_mean_bps, 2),
        "edge_bps"          : round(on_mean_bps - id_mean_bps, 2),
        "overnight_sharpe"  : round(on_sharpe, 3),
        "intraday_sharpe"   : round(id_sharpe, 3),
        "net_sharpe"        : round(on_net_sharpe, 3),
        "win_rate"          : round(on_win_rate, 1),
        "t_stat"            : round(t_stat, 2),
        "p_val"             : round(p_val, 4) if not np.isnan(p_val) else None,
        "breakeven_bps"     : round(breakeven_bps, 1),
        "gross_cum_pct"     : round(float(on_cum.iloc[-1] - 1) * 100, 1),
        "net_cum_pct"       : round(float(on_net_cum.iloc[-1] - 1) * 100, 1),
        "intraday_cum_pct"  : round(float(id_cum.iloc[-1] - 1) * 100, 1),
        "bah_cum_pct"       : round(float(bah_cum.iloc[-1] - 1) * 100, 1),
        # Time series for charting (prefixed with _ so scanner strips them)
        "_dates"            : df.index,
        "_on_cum"           : on_cum,
        "_on_net_cum"       : on_net_cum,
        "_id_cum"           : id_cum,
        "_bah_cum"          : bah_cum,
        "_on_series"        : on,
        "_monthly"          : monthly,
    }


def scan_tickers(
    tickers: list[str],
    period_days: int = 1825,
    cost_bps: float = 3.0,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Compute overnight stats for each ticker.
    Returns a DataFrame ranked by net_sharpe (best first).
    progress_cb(ticker, i, n) is called before each fetch.
    """
    rows = []
    n = len(tickers)
    for i, tk in enumerate(tickers):
        if progress_cb:
            progress_cb(tk, i, n)
        result = compute_overnight_stats(tk, period_days, cost_bps)
        if result:
            rows.append({k: v for k, v in result.items() if not k.startswith("_")})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df
