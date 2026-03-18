"""
predict.py — Daily signal generator and position tracker

Run this each day ~30 min before market close.

Rules:
  Entry : Buy at close (MOC order) when model probability > CV threshold
  Exit  : Sell at close after 5 trading days  OR  if down >=3% from entry
  Sizing: Equal weight across max 5 simultaneous open positions
          New signals ranked by probability; skipped if already at max positions

Usage:
  python predict.py                    # use tickers from ticker_summary.csv
  python predict.py --portfolio 10000  # set portfolio size
  python predict.py --dry-run          # show signals without updating positions
"""

import argparse
import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
POSITIONS_FILE  = "positions.json"
SUMMARY_FILE    = "ticker_summary.csv"
MAX_POSITIONS   = 8
STOP_LOSS_PCT   = 0.03    # 3% hard stop
HOLD_DAYS       = 5       # trading days
HISTORY_DAYS    = 300     # enough for 200-day MA + buffer
DEFAULT_CAPITAL = 10_000
MIN_ROC_AUC     = 0.55    # only trade tickers with CV ROC-AUC above this
MIN_PROB        = 0.50    # only buy when model is majority-confident
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_positions() -> list[dict]:
    if Path(POSITIONS_FILE).exists():
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return []


def save_positions(positions: list[dict]):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def trading_days_between(start: date, end: date) -> int:
    """Count trading days (Mon–Fri) between two dates inclusive of end."""
    days = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:   # Mon=0 … Fri=4
            days += 1
        current += timedelta(days=1)
    return days


def load_model_and_features(ticker: str):
    model_path    = Path(f"model_{ticker}.joblib")
    features_path = Path(f"features_{ticker}.csv")
    if not model_path.exists() or not features_path.exists():
        return None, None
    model         = joblib.load(model_path)
    feature_names = pd.read_csv(features_path, header=None)[0].tolist()
    return model, feature_names


# ── Feature computation (mirrors features.py) ─────────────────────────────────
def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = series.copy() * np.nan
    result.iloc[period - 1] = series.iloc[:period].mean()
    for i in range(period, len(series)):
        result.iloc[i] = result.iloc[i - 1] * (1 - 1 / period) + series.iloc[i] / period
    return result


def _true_range(high, low, close) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low,
                      (high - prev_close).abs(),
                      (low  - prev_close).abs()], axis=1).max(axis=1)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    tr = _true_range(h, l, c)

    for p in [5, 10, 20, 50, 100, 200]:
        sma = c.rolling(p).mean()
        feat[f"SMA_{p}"]       = sma
        feat[f"Price_SMA_{p}"] = c / sma

    for p in [5, 10, 20, 50, 100, 200]:
        ema = c.ewm(span=p, adjust=False).mean()
        feat[f"EMA_{p}"]       = ema
        feat[f"Price_EMA_{p}"] = c / ema

    feat["SMA_5_20_cross"]   = c.rolling(5).mean()  - c.rolling(20).mean()
    feat["SMA_10_50_cross"]  = c.rolling(10).mean() - c.rolling(50).mean()
    feat["SMA_20_200_cross"] = c.rolling(20).mean() - c.rolling(200).mean()
    feat["EMA_12_26_cross"]  = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()

    delta = c.diff()
    for p in [7, 14, 21]:
        gain     = delta.clip(lower=0)
        loss     = (-delta.clip(upper=0))
        avg_gain = _wilder_smooth(gain, p)
        avg_loss = _wilder_smooth(loss, p)
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        feat[f"RSI_{p}"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    feat["MACD_line"]      = ema12 - ema26
    feat["MACD_signal"]    = feat["MACD_line"].ewm(span=9, adjust=False).mean()
    feat["MACD_histogram"] = feat["MACD_line"] - feat["MACD_signal"]

    for p in [10, 20, 50]:
        sma   = c.rolling(p).mean()
        std   = c.rolling(p).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bw    = upper - lower
        feat[f"BB_pct_{p}"]   = (c - lower) / bw.replace(0, np.nan)
        feat[f"BB_width_{p}"] = bw / sma

    for p in [7, 14, 21]:
        atr = _wilder_smooth(tr, p)
        feat[f"ATR_{p}"]     = atr
        feat[f"ATR_pct_{p}"] = atr / c

    for p in [14, 21]:
        ll = l.rolling(p).min()
        hh = h.rolling(p).max()
        k  = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        feat[f"Stoch_K_{p}"] = k
        feat[f"Stoch_D_{p}"] = k.rolling(3).mean()

    for p in [14, 21]:
        hh = h.rolling(p).max()
        ll = l.rolling(p).min()
        feat[f"Williams_R_{p}"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)

    tp = (h + l + c) / 3
    for p in [14, 20]:
        sma_tp = tp.rolling(p).mean()
        mad    = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        feat[f"CCI_{p}"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    for p in [1, 5, 10, 20, 60]:
        feat[f"ROC_{p}"] = c.pct_change(p)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    feat["OBV"]       = obv
    feat["OBV_SMA20"] = obv.rolling(20).mean()
    feat["OBV_ratio"] = obv / obv.rolling(20).mean().replace(0, np.nan)

    feat["Vol_ratio_10"] = v / v.rolling(10).mean().replace(0, np.nan)
    feat["Vol_ratio_20"] = v / v.rolling(20).mean().replace(0, np.nan)

    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm[plus_dm   < minus_dm] = 0
    minus_dm[minus_dm < plus_dm]  = 0
    p      = 14
    atr14  = _wilder_smooth(tr, p)
    pdi14  = 100 * _wilder_smooth(plus_dm,  p) / atr14.replace(0, np.nan)
    mdi14  = 100 * _wilder_smooth(minus_dm, p) / atr14.replace(0, np.nan)
    di_sum = (pdi14 + mdi14).replace(0, np.nan)
    dx     = 100 * (pdi14 - mdi14).abs() / di_sum
    feat["ADX_14"]      = _wilder_smooth(dx, p)
    feat["Plus_DI_14"]  = pdi14
    feat["Minus_DI_14"] = mdi14

    body   = (c - o).abs()
    candle = (h - l).replace(0, np.nan)
    feat["Body_pct"]     = body / o
    feat["HL_range"]     = candle / o
    feat["Upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / candle
    feat["Lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / candle
    feat["Gap"]          = (o - c.shift(1)) / c.shift(1)

    ret = c.pct_change()
    for p in [5, 10, 20]:
        feat[f"Ret_mean_{p}"] = ret.rolling(p).mean()
        feat[f"Ret_std_{p}"]  = ret.rolling(p).std()
        feat[f"Ret_skew_{p}"] = ret.rolling(p).skew()

    for p in [20, 52]:
        feat[f"Pct_from_high_{p}"] = (c - h.rolling(p).max()) / h.rolling(p).max()
        feat[f"Pct_from_low_{p}"]  = (c - l.rolling(p).min()) / l.rolling(p).min()

    mid = (h + l) / 2
    feat["SAR_proxy"] = c / mid.ewm(span=14, adjust=False).mean() - 1

    raw_mf  = tp * v
    pos_mf  = raw_mf.where(tp > tp.shift(1), 0)
    neg_mf  = raw_mf.where(tp < tp.shift(1), 0)
    mf_ratio = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    feat["MFI_14"] = 100 - 100 / (1 + mf_ratio)

    vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    feat["VWAP_dev"] = (c - vwap) / vwap

    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    feat["CMF_20"] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    dm  = ((h + l) / 2) - ((h.shift(1) + l.shift(1)) / 2)
    br  = v / 1e6 / (h - l).replace(0, np.nan)
    feat["EOM_14"] = (dm / br.replace(0, np.nan)).rolling(14).mean()

    feat["Force_13"] = (c.diff() * v).ewm(span=13, adjust=False).mean()

    kc_mid  = c.ewm(span=20, adjust=False).mean()
    kc_band = atr14 * 2
    feat["KC_pct"]   = (c - (kc_mid - kc_band)) / (2 * kc_band).replace(0, np.nan)
    feat["KC_width"] = (2 * kc_band) / kc_mid

    feat["DC_width_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c

    ema1 = c.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    ema3 = ema2.ewm(span=20, adjust=False).mean()
    feat["TEMA_20"] = c / (3*ema1 - 3*ema2 + ema3) - 1
    feat["DEMA_20"] = c / (2*ema1 - ema2) - 1

    half_ema = c.ewm(span=10, adjust=False).mean()
    full_ema = c.ewm(span=20, adjust=False).mean()
    hull_raw = 2*half_ema - full_ema
    feat["HMA_proxy"] = c / hull_raw.ewm(span=4, adjust=False).mean() - 1

    feat["PPO"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) \
                  / c.ewm(span=26, adjust=False).mean()

    t1 = c.ewm(span=15, adjust=False).mean()
    t2 = t1.ewm(span=15, adjust=False).mean()
    t3 = t2.ewm(span=15, adjust=False).mean()
    feat["TRIX_15"] = t3.pct_change()

    bp   = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr_uo = pd.concat([h, c.shift(1)], axis=1).max(axis=1) \
           - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    feat["UltOsc"] = 100 * (
        4 * bp.rolling(7).sum()  / tr_uo.rolling(7).sum()  +
        2 * bp.rolling(14).sum() / tr_uo.rolling(14).sum() +
            bp.rolling(28).sum() / tr_uo.rolling(28).sum()
    ) / 7

    ema9_hl   = (h - l).ewm(span=9, adjust=False).mean()
    ema9_ema9 = ema9_hl.ewm(span=9, adjust=False).mean()
    feat["MassIndex_25"] = (ema9_hl / ema9_ema9.replace(0, np.nan)).rolling(25).sum()

    vm_plus   = (h - l.shift(1)).abs().rolling(14).sum()
    vm_minus  = (l - h.shift(1)).abs().rolling(14).sum()
    atr14_sum = tr.rolling(14).sum()
    feat["VI_plus_14"]  = vm_plus  / atr14_sum.replace(0, np.nan)
    feat["VI_minus_14"] = vm_minus / atr14_sum.replace(0, np.nan)

    for p in [14, 25]:
        aroon_up   = h.rolling(p+1).apply(lambda x: x.argmax(), raw=True) / p * 100
        aroon_down = l.rolling(p+1).apply(lambda x: x.argmin(), raw=True) / p * 100
        feat[f"Aroon_osc_{p}"] = aroon_up - aroon_down

    feat["Coppock"] = (c.pct_change(11) + c.pct_change(14)).rolling(10).mean()

    ema14 = c.ewm(span=14, adjust=False).mean()
    feat["Bull_power"] = h - ema14
    feat["Bear_power"] = l - ema14

    feat["DPO_20"] = c - c.rolling(11).mean().shift(11)

    for lag in [1, 2, 3, 5, 10]:
        feat[f"Lag_ret_{lag}"] = c.pct_change(lag)

    return feat


# ── Core logic ────────────────────────────────────────────────────────────────
def get_latest_signal(ticker: str, model, feature_names: list[str],
                      threshold: float) -> dict | None:
    """Download recent data, compute features, return signal dict or None."""
    df = yf.download(ticker, period=f"{HISTORY_DAYS}d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 210:
        return None

    feat = compute_features(df)
    row  = feat.iloc[[-1]][feature_names]
    if row.isnull().any(axis=1).values[0]:
        return None

    prob       = model.predict_proba(row)[0][1]
    close      = float(df["Close"].iloc[-1])
    as_of_date = df.index[-1].date()

    return {
        "ticker"   : ticker,
        "date"     : str(as_of_date),
        "close"    : round(close, 4),
        "prob"     : round(prob, 4),
        "threshold": threshold,
        "signal"   : prob >= threshold,
    }


def check_exits(positions: list[dict], today: date) -> tuple[list[dict], list[dict]]:
    """
    Check each open position for exit conditions.
    Returns (still_open, to_close) where to_close includes the exit reason.
    """
    still_open = []
    to_close   = []

    for pos in positions:
        if pos["status"] != "open":
            still_open.append(pos)
            continue

        entry_date  = date.fromisoformat(pos["entry_date"])
        entry_price = pos["entry_price"]
        ticker      = pos["ticker"]

        # Fetch current price
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        current_price = float(df["Close"].iloc[-1])
        pnl_pct       = (current_price - entry_price) / entry_price

        days_held = trading_days_between(entry_date, today)

        if pnl_pct <= -STOP_LOSS_PCT:
            pos.update({"status": "closed_stop", "exit_date": str(today),
                        "exit_price": round(current_price, 4),
                        "pnl_pct": round(pnl_pct * 100, 2)})
            to_close.append(pos)
        elif days_held >= HOLD_DAYS:
            pos.update({"status": "closed_time", "exit_date": str(today),
                        "exit_price": round(current_price, 4),
                        "pnl_pct": round(pnl_pct * 100, 2)})
            to_close.append(pos)
        else:
            pos["current_price"] = round(current_price, 4)
            pos["days_held"]     = days_held
            pos["pnl_pct"]       = round(pnl_pct * 100, 2)
            still_open.append(pos)

    return still_open, to_close


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print signals without saving position changes")
    args = parser.parse_args()

    today     = date.today()
    portfolio = args.portfolio

    # Load ticker list + thresholds
    if not Path(SUMMARY_FILE).exists():
        print(f"ERROR: {SUMMARY_FILE} not found. Run features.py first.")
        return
    summary   = pd.read_csv(SUMMARY_FILE)
    tickers   = summary["Ticker"].tolist()
    threshold_map = dict(zip(summary["Ticker"], summary["CV_Threshold"]))
    roc_auc_map   = dict(zip(summary["Ticker"], summary["CV_ROC_AUC"]))

    eligible = [t for t in tickers if roc_auc_map.get(t, 0) >= MIN_ROC_AUC]
    skipped  = [t for t in tickers if roc_auc_map.get(t, 0) <  MIN_ROC_AUC]
    print(f"\n  Quality filter (CV ROC-AUC >= {MIN_ROC_AUC}): "
          f"{len(eligible)} eligible, {len(skipped)} excluded")
    print(f"  Excluded: {', '.join(skipped)}")
    tickers = eligible

    print(f"\n{'='*60}")
    print(f"  Daily Signal Report  —  {today}  (portfolio ${portfolio:,.0f})")
    print(f"{'='*60}")

    # ── 1. Check exits ──
    positions          = load_positions()
    open_positions     = [p for p in positions if p["status"] == "open"]
    closed_positions   = [p for p in positions if p["status"] != "open"]

    print(f"\n[EXITS]")
    if open_positions:
        open_positions, to_close = check_exits(open_positions, today)
        for pos in to_close:
            reason = "STOP LOSS" if pos["status"] == "closed_stop" else "TIME EXIT"
            pnl    = pos["pnl_pct"]
            sign   = "+" if pnl >= 0 else ""
            print(f"  SELL {pos['ticker']:6s}  {reason}  "
                  f"entry={pos['entry_price']}  exit={pos['exit_price']}  "
                  f"P&L={sign}{pnl}%  →  PLACE MOC SELL ORDER")
        closed_positions.extend(to_close)
        if not to_close:
            print("  No exits today.")
    else:
        print("  No open positions.")

    # ── 2. Show open positions ──
    print(f"\n[OPEN POSITIONS]  ({len(open_positions)}/{MAX_POSITIONS})")
    if open_positions:
        for pos in open_positions:
            pnl  = pos.get("pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(f"  {pos['ticker']:6s}  entry={pos['entry_date']}  "
                  f"@ {pos['entry_price']}  now={pos.get('current_price','?')}  "
                  f"P&L={sign}{pnl}%  day {pos.get('days_held','?')}/{HOLD_DAYS}")
    else:
        print("  None")

    # ── 3. Scan for new signals ──
    slots_available = MAX_POSITIONS - len(open_positions)
    open_tickers    = {p["ticker"] for p in open_positions}

    print(f"\n[SCANNING {len(tickers)} TICKERS FOR BUY SIGNALS ...]")
    signals = []
    for ticker in tickers:
        model, feature_names = load_model_and_features(ticker)
        if model is None:
            print(f"  {ticker}: no saved model, skipping")
            continue
        if ticker in open_tickers:
            continue    # already holding this one
        threshold = threshold_map.get(ticker, 0.30)
        result    = get_latest_signal(ticker, model, feature_names, threshold)
        if result:
            signals.append(result)
            above_thresh = result["signal"]
            above_min    = result["prob"] >= MIN_PROB
            if above_thresh and above_min:
                flag = "  BUY"
            elif above_thresh and not above_min:
                flag = " WEAK"   # passed ticker threshold but prob < 0.50
            else:
                flag = "     "
            print(f"  {flag} {ticker:6s}  prob={result['prob']:.3f}  "
                  f"thresh={threshold:.2f}  close={result['close']}")

    # ── 4. New entries ──
    buy_signals = sorted([s for s in signals if s["signal"] and s["prob"] >= MIN_PROB],
                         key=lambda x: x["prob"], reverse=True)

    print(f"\n[NEW ENTRIES]  ({slots_available} slot(s) available)")
    new_entries = []
    if not buy_signals:
        print("  No buy signals today.")
    elif slots_available == 0:
        print("  Portfolio full — no new entries.")
    else:
        position_size = round(portfolio / MAX_POSITIONS, 2)
        taken         = 0
        for sig in buy_signals:
            if taken >= slots_available:
                print(f"  SKIP   {sig['ticker']:6s}  prob={sig['prob']:.3f}  "
                      f"(no slots remaining)")
                continue
            whole_shares    = int(position_size / sig["close"])
            frac_shares     = round(position_size / sig["close"], 4)
            whole_cost      = round(whole_shares * sig["close"], 2)
            # Warn if can't afford even 1 whole share
            frac_note = ""
            if whole_shares == 0:
                frac_note = f"  [fractional: {frac_shares} shares @ ${position_size:.2f}]"
                shares, cost = frac_shares, position_size
            else:
                shares, cost = whole_shares, whole_cost
            print(f"  BUY    {sig['ticker']:6s}  prob={sig['prob']:.3f}  "
                  f"close=${sig['close']}  "
                  f"shares={shares}  cost=${cost:,.2f}"
                  f"{frac_note}  →  PLACE MOC BUY ORDER")
            new_entries.append({
                "ticker"      : sig["ticker"],
                "entry_date"  : sig["date"],
                "entry_price" : sig["close"],
                "shares"      : shares,
                "cost"        : cost,
                "status"      : "open",
            })
            taken += 1

    # ── 5. P&L summary ──
    all_closed = closed_positions + [p for p in to_close] \
        if open_positions else closed_positions
    closed_with_pnl = [p for p in all_closed if "pnl_pct" in p]
    if closed_with_pnl:
        avg_pnl = sum(p["pnl_pct"] for p in closed_with_pnl) / len(closed_with_pnl)
        wins    = sum(1 for p in closed_with_pnl if p["pnl_pct"] > 0)
        print(f"\n[HISTORICAL P&L]  {len(closed_with_pnl)} closed trades  |  "
              f"Win rate {wins/len(closed_with_pnl):.0%}  |  Avg P&L {avg_pnl:+.2f}%")

    # ── 6. Save updated positions ──
    if not args.dry_run:
        all_positions = open_positions + closed_positions + new_entries
        save_positions(all_positions)
        print(f"\nPositions saved → {POSITIONS_FILE}")
    else:
        print("\n[DRY RUN] Position file not updated.")

    print()


if __name__ == "__main__":
    main()
