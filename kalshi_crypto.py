"""
kalshi_crypto.py — Crypto prediction pipeline for Kalshi markets

Two separate models based on contract time-to-expiry:
  - Daily model   (>24h):  trained on daily OHLC, horizons 1/3/5/7/10 days
  - Intraday model (≤24h): trained on hourly OHLC, horizons 1/2/3/4/8/12/24 hours

Full pipeline:
  1. Train model(s) on BTC-USD + ETH-USD data
  2. For each open Kalshi crypto contract:
       a. Select daily or intraday model based on hours_to_expiry
       b. Compute base rate (historical frequency of hitting the strike)
       c. Get model probability
       d. Bayesian calibration (log-odds update: swap training prior for actual prior)
       e. Compute EV
       f. Compute half-Kelly position size
  3. Print ranked recommendations

Usage:
  python kalshi_crypto.py              # score live contracts (dry-run if no creds)
  python kalshi_crypto.py --train      # retrain daily model then score
  python kalshi_crypto.py --train-all  # retrain both daily + intraday models
"""

import argparse
import json
import logging
import math
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from kalshi_api import KalshiClient
from features import compute_features, _wilder_smooth, _true_range

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CRYPTO_ASSETS      = ["BTC-USD", "ETH-USD"]
KALSHI_SERIES      = {"BTC-USD": "KXBTCD", "ETH-USD": "KXETHD"}
STRIKE_OFFSETS     = [-0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.15]

# Daily model
HORIZONS           = [1, 3, 5, 7, 10]   # trading days
HISTORY_PERIOD     = "5y"
INFERENCE_PERIOD   = "2y"

# Intraday model
HOURLY_HISTORY     = "730d"             # max for yfinance hourly data
HOURLY_HORIZONS    = [1, 2, 3, 4, 8, 12, 24]  # hours

HIST_VOL_WINDOW    = 30                  # days for rolling daily vol
BASE_RATE_LOOKBACK = 3                   # years for base-rate estimation

# Daily model paths (backward-compatible names)
MODEL_PATH         = "model_crypto.joblib"
FEATURES_PATH      = "features_crypto.csv"
METADATA_PATH      = "model_crypto_meta.json"

# Intraday model paths
MODEL_INTRADAY_PATH    = "model_crypto_intraday.joblib"
FEATURES_INTRADAY_PATH = "features_crypto_intraday.csv"
METADATA_INTRADAY_PATH = "model_crypto_intraday_meta.json"

# Paper-trade calibration (Platt scaling fitted on live outcomes)
CALIBRATION_PATH   = "model_crypto_calibration.json"

RF_TREES           = 300
RF_MAX_DEPTH       = 12
RF_MIN_LEAF        = 5
RANDOM_STATE       = 42
N_FOLDS            = 5
MIN_TRAIN_FRAC     = 0.5
MIN_EV             = 0.05
MIN_EDGE           = 0.05
MAX_KELLY          = 0.25
DEFAULT_BANKROLL   = 500
# ──────────────────────────────────────────────────────────────────────────────


# ── Data ──────────────────────────────────────────────────────────────────────
def download_crypto(symbol: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def download_crypto_hourly(symbol: str) -> pd.DataFrame:
    """Download hourly OHLC (up to 730 days, yfinance limit)."""
    df = yf.download(symbol, period=HOURLY_HISTORY, interval="1h",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def hist_vol(close: pd.Series, window: int = HIST_VOL_WINDOW) -> pd.Series:
    """Annualised rolling volatility from log returns (daily)."""
    return close.pct_change().apply(np.log1p).rolling(window).std() * math.sqrt(252)


# ── Hourly feature engineering ────────────────────────────────────────────────
def compute_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from hourly OHLC data.
    Returns DataFrame with same index as df.
    """
    close  = df["Close"]
    volume = df["Volume"]
    feats  = pd.DataFrame(index=df.index)

    # Returns at multiple timeframes
    for h in [1, 4, 8, 24, 48, 168]:
        feats[f"ret_{h}h"] = close.pct_change(h)

    # Rolling annualised vol (8760 trading hours/year for crypto)
    log_ret = close.pct_change().apply(np.log1p)
    feats["vol_24h"]  = log_ret.rolling(24).std()  * math.sqrt(8760)
    feats["vol_168h"] = log_ret.rolling(168).std() * math.sqrt(8760)

    # RSI 14-period on hourly candles
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    feats["rsi_14h"] = 100 - 100 / (1 + rs)

    # MACD (12/26/9) normalized by price
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = (ema12 - ema26) / close
    sig   = macd.ewm(span=9, adjust=False).mean()
    feats["macd_h"]      = macd
    feats["macd_hist_h"] = macd - sig

    # Bollinger band position (0 = at lower band, 0.5 = at mean, 1 = at upper)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feats["bb_pct_h"] = (close - sma20) / (2 * std20 + 1e-10)

    # Volume ratio vs 24h rolling average
    feats["vol_ratio_24h"] = volume / (volume.rolling(24).mean() + 1e-10)

    # Time-of-day cyclical encoding (crypto trades 24/7 — intraday patterns exist)
    hour_utc = pd.DatetimeIndex(df.index).hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour_utc / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour_utc / 24)

    return feats


# ── Training data generation ──────────────────────────────────────────────────
def generate_training_samples(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Daily training samples. For each date t, strike offset, and horizon:
      - tech features from compute_features(df).iloc[t]
      - contract features: days_to_expiry, strike_distance, strike_z_score
      - label: 1 if close[t+horizon] > close[t]*(1+offset)
    """
    logger.info(f"  Generating daily training samples for {symbol} ...")

    feat_df = compute_features(df).dropna(how="all")
    vol_s   = hist_vol(df["Close"])
    close   = df["Close"]

    max_horizon = max(HORIZONS)
    rows = []

    valid_idx = feat_df.dropna().index
    valid_idx = valid_idx[valid_idx <= close.index[-max_horizon - 1]]

    for dt in valid_idx:
        feat_row = feat_df.loc[dt]
        if feat_row.isnull().any():
            continue

        c0  = float(close.loc[dt])
        vol = float(vol_s.loc[dt]) if not pd.isna(vol_s.loc[dt]) else 0.5

        for horizon in HORIZONS:
            future_idx = close.index.get_loc(dt) + horizon
            if future_idx >= len(close):
                continue
            c_future = float(close.iloc[future_idx])

            for offset in STRIKE_OFFSETS:
                strike = c0 * (1 + offset)
                label  = int(c_future > strike)

                denom         = vol * math.sqrt(horizon / 252)
                strike_z_score = offset / denom if denom > 0 else 0.0

                row = feat_row.to_dict()
                row["days_to_expiry"]  = horizon
                row["strike_distance"] = offset
                row["strike_z_score"]  = strike_z_score
                row["label"]           = label
                row["_date"]           = dt
                rows.append(row)

    result = pd.DataFrame(rows).sort_values("_date").reset_index(drop=True)
    logger.info(f"    {len(result):,} samples, label mean = {result['label'].mean():.3f}")
    return result


def generate_training_samples_hourly(df: pd.DataFrame, symbol: str,
                                      horizons: list[int]) -> pd.DataFrame:
    """
    Hourly training samples. For each hourly bar t, strike offset, and horizon (hours):
      - hourly tech features from compute_features_hourly(df).iloc[t]
      - contract features: hours_to_expiry, strike_distance, strike_z_score
      - label: 1 if close[t+horizon_hours] > close[t]*(1+offset)
    """
    logger.info(f"  Generating hourly training samples for {symbol} ...")

    feat_df = compute_features_hourly(df).dropna(how="all")
    close   = df["Close"]

    max_horizon = max(horizons)
    rows = []

    valid_idx = feat_df.dropna().index
    valid_idx = valid_idx[valid_idx <= close.index[-max_horizon - 1]]

    for dt in valid_idx:
        feat_row = feat_df.loc[dt]
        if feat_row.isnull().any():
            continue

        c0  = float(close.loc[dt])
        vol = float(feat_df.loc[dt, "vol_24h"])
        if pd.isna(vol):
            vol = 0.5

        for horizon_h in horizons:
            future_loc = close.index.get_loc(dt) + horizon_h
            if future_loc >= len(close):
                continue
            c_future = float(close.iloc[future_loc])

            for offset in STRIKE_OFFSETS:
                strike   = c0 * (1 + offset)
                label    = int(c_future > strike)
                denom    = vol * math.sqrt(horizon_h / 8760)
                z_score  = offset / denom if denom > 0 else 0.0

                row = feat_row.to_dict()
                row["hours_to_expiry"] = horizon_h
                row["strike_distance"] = offset
                row["strike_z_score"]  = z_score
                row["label"]           = label
                row["_date"]           = dt
                rows.append(row)

    result = pd.DataFrame(rows).sort_values("_date").reset_index(drop=True)
    logger.info(f"    {len(result):,} hourly samples, label mean = {result['label'].mean():.3f}")
    return result


# ── Model ─────────────────────────────────────────────────────────────────────
def _time_decay_weights(dates: pd.Series, half_life: int = 180) -> np.ndarray:
    """Exponential decay weights so recent data has more influence."""
    today_ts = pd.Timestamp(datetime.now(timezone.utc).date())
    days_old = (today_ts - pd.to_datetime(dates).dt.tz_localize(None)).dt.days.clip(lower=0)
    weights  = np.exp(-np.log(2) * days_old / half_life).values
    weights /= weights.mean()
    return weights


def _walk_forward_cv(X: pd.DataFrame, y: pd.Series,
                     dates: pd.Series) -> float:
    """Expanding-window CV split on date; returns mean ROC-AUC."""
    unique_dates = dates.sort_values().unique()
    n            = len(unique_dates)
    min_train    = int(n * MIN_TRAIN_FRAC)
    fold_size    = (n - min_train) // N_FOLDS
    if fold_size < 1:
        return float("nan")

    aucs = []
    for fold in range(N_FOLDS):
        train_end_date = unique_dates[min_train + fold * fold_size - 1]
        test_end_date  = unique_dates[min(min_train + (fold + 1) * fold_size - 1, n - 1)]

        train_mask = dates <= train_end_date
        test_mask  = (dates > train_end_date) & (dates <= test_end_date)
        if test_mask.sum() < 20:
            continue

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_LEAF, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X[train_mask], y[train_mask])
        proba = rf.predict_proba(X[test_mask])[:, 1]
        if len(y[test_mask].unique()) > 1:
            aucs.append(roc_auc_score(y[test_mask], proba))
        logger.info(f"    Fold {fold+1}: {train_mask.sum():,} train, "
                    f"{test_mask.sum():,} test, "
                    f"ROC-AUC={aucs[-1]:.4f}" if aucs else "")

    return float(np.mean(aucs)) if aucs else float("nan")


def build_crypto_model() -> tuple[RandomForestClassifier, list[str], float]:
    """Train daily model on BTC + ETH, run walk-forward CV, save model."""
    logger.info("Building daily crypto model ...")
    half_life = 180

    all_samples = []
    for symbol in CRYPTO_ASSETS:
        df = download_crypto(symbol, HISTORY_PERIOD)
        logger.info(f"  {symbol}: {len(df)} days, "
                    f"{df.index[0].date()} → {df.index[-1].date()}")
        samples = generate_training_samples(df, symbol)
        all_samples.append(samples)

    data = pd.concat(all_samples, ignore_index=True)
    data = data.sort_values("_date").reset_index(drop=True)

    dates  = data["_date"]
    y      = data["label"]
    X      = data.drop(columns=["label", "_date"])
    feature_names = X.columns.tolist()

    weights = _time_decay_weights(dates, half_life)
    training_base_rate = float(np.average(y, weights=weights))
    logger.info(f"  Combined: {len(X):,} samples, "
                f"weighted base rate = {training_base_rate:.3f}  "
                f"(unweighted = {float(y.mean()):.3f})")

    logger.info("  Running walk-forward CV ...")
    cv_roc_auc = _walk_forward_cv(X, y, dates)
    logger.info(f"  CV ROC-AUC = {cv_roc_auc:.4f}")

    logger.info("  Training final model on all data ...")
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y, sample_weight=weights)

    joblib.dump(rf, MODEL_PATH)
    pd.Series(feature_names).to_csv(FEATURES_PATH, index=False, header=False)
    meta = {
        "training_base_rate"   : training_base_rate,
        "trained_on"           : str(date.today()),
        "n_samples"            : len(X),
        "cv_roc_auc"           : round(cv_roc_auc, 4),
        "weight_half_life_days": half_life,
        "feature_names"        : feature_names,
        "model_type"           : "daily",
        "horizons_days"        : HORIZONS,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Daily model saved → {MODEL_PATH}")

    return rf, feature_names, training_base_rate


def build_intraday_model() -> tuple[RandomForestClassifier, list[str], float]:
    """Train intraday model on BTC + ETH hourly data."""
    logger.info("Building intraday crypto model (hourly data) ...")
    half_life = 180

    all_samples = []
    for symbol in CRYPTO_ASSETS:
        df = download_crypto_hourly(symbol)
        logger.info(f"  {symbol}: {len(df)} hourly bars, "
                    f"{df.index[0]} → {df.index[-1]}")
        samples = generate_training_samples_hourly(df, symbol, HOURLY_HORIZONS)
        all_samples.append(samples)

    data = pd.concat(all_samples, ignore_index=True).sort_values("_date").reset_index(drop=True)

    dates = data["_date"]
    y     = data["label"]
    X     = data.drop(columns=["label", "_date"])
    feature_names = X.columns.tolist()

    weights = _time_decay_weights(dates, half_life)
    training_base_rate = float(np.average(y, weights=weights))
    logger.info(f"  Combined: {len(X):,} hourly samples, "
                f"weighted base rate = {training_base_rate:.3f}")

    logger.info("  Running walk-forward CV ...")
    cv_roc_auc = _walk_forward_cv(X, y, dates)
    logger.info(f"  Intraday CV ROC-AUC = {cv_roc_auc:.4f}")

    logger.info("  Training final intraday model on all data ...")
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y, sample_weight=weights)

    joblib.dump(rf, MODEL_INTRADAY_PATH)
    pd.Series(feature_names).to_csv(FEATURES_INTRADAY_PATH, index=False, header=False)
    meta = {
        "training_base_rate"   : training_base_rate,
        "trained_on"           : str(date.today()),
        "n_samples"            : len(X),
        "cv_roc_auc"           : round(cv_roc_auc, 4),
        "weight_half_life_days": half_life,
        "feature_names"        : feature_names,
        "model_type"           : "intraday_hourly",
        "horizons_hours"       : HOURLY_HORIZONS,
    }
    with open(METADATA_INTRADAY_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Intraday model saved → {MODEL_INTRADAY_PATH}")

    return rf, feature_names, training_base_rate


def load_crypto_model() -> tuple:
    """Load daily model. Returns (None, None, None) if missing."""
    if not Path(MODEL_PATH).exists() or not Path(METADATA_PATH).exists():
        return None, None, None
    rf            = joblib.load(MODEL_PATH)
    feature_names = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    training_base_rate = meta["training_base_rate"]
    logger.info(f"Loaded daily model trained on {meta['trained_on']} "
                f"({meta['n_samples']:,} samples, CV ROC-AUC={meta['cv_roc_auc']})")
    return rf, feature_names, training_base_rate


def load_crypto_models() -> dict:
    """
    Load all available models.
    Returns: {
        "daily":    (model, feature_names, training_base_rate) or None,
        "intraday": (model, feature_names, training_base_rate) or None,
    }
    """
    models = {}

    # Daily model
    daily = load_crypto_model()
    models["daily"] = daily if daily[0] is not None else None

    # Intraday model
    if (Path(MODEL_INTRADAY_PATH).exists()
            and Path(METADATA_INTRADAY_PATH).exists()
            and Path(FEATURES_INTRADAY_PATH).exists()):
        rf            = joblib.load(MODEL_INTRADAY_PATH)
        feature_names = pd.read_csv(FEATURES_INTRADAY_PATH, header=None)[0].tolist()
        with open(METADATA_INTRADAY_PATH) as f:
            meta = json.load(f)
        tbr = meta["training_base_rate"]
        logger.info(f"Loaded intraday model trained on {meta['trained_on']} "
                    f"({meta['n_samples']:,} samples, CV ROC-AUC={meta['cv_roc_auc']})")
        models["intraday"] = (rf, feature_names, tbr)
    else:
        models["intraday"] = None
        logger.info("No intraday model found — run --train-all to build it.")

    models["calibration"] = load_calibration()
    return models


# ── Paper-trade calibration (Platt scaling) ───────────────────────────────────
def load_calibration() -> dict | None:
    """Load Platt scaling params saved by recalibrate_from_paper_trades()."""
    if not Path(CALIBRATION_PATH).exists():
        return None
    try:
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def recalibrate_from_paper_trades(paper_trades_path: str) -> dict:
    """
    Fit Platt scaling (logistic regression on model_prob → settlement outcome)
    using settled paper trades, with exponential decay weights (half-life 30 days).

    For both YES and NO bets, the calibration target is always whether the market
    settled YES. This lets both sides contribute signal to the same curve.

    Saves calibration to CALIBRATION_PATH and returns the result dict.
    Minimum 5 settled trades with at least 1 win and 1 loss per bucket.
    """
    from sklearn.linear_model import LogisticRegression

    try:
        with open(paper_trades_path) as f:
            trades = json.load(f)
    except Exception as e:
        return {"error": f"Could not read paper trades: {e}"}

    settled = [t for t in trades
               if t.get("status") == "settled"
               and t.get("result") in ("yes", "no")]
    if not settled:
        return {"error": "No settled trades with known results yet."}

    now_utc  = datetime.now(timezone.utc)
    buckets  = {}

    for bucket in ("daily", "intraday"):
        bt = [t for t in settled if t.get("bucket") == bucket]
        if len(bt) < 5:
            buckets[bucket] = {"skipped": True, "reason": f"Only {len(bt)} settled trades (need ≥5)"}
            continue

        probs, outcomes, weights = [], [], []
        for t in bt:
            mp     = float(t.get("model_prob", 0.5))
            result = t.get("result", "").lower()
            # outcome = 1 if market settled YES (independent of side bet)
            outcome = 1 if result == "yes" else 0

            placed_str = t.get("placed_at", "")
            try:
                placed_dt = datetime.fromisoformat(placed_str)
                if placed_dt.tzinfo is None:
                    placed_dt = placed_dt.replace(tzinfo=timezone.utc)
                days_ago = max(0, (now_utc - placed_dt).days)
            except Exception:
                days_ago = 0
            w = math.exp(-days_ago / 30)   # half-life 30 days

            probs.append(mp)
            outcomes.append(outcome)
            weights.append(w)

        y_arr = np.array(outcomes)
        if len(set(y_arr)) < 2:
            buckets[bucket] = {
                "skipped": True,
                "reason": "Need at least 1 win and 1 loss to fit calibration",
            }
            continue

        X = np.array(probs).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, fit_intercept=True, max_iter=500)
        lr.fit(X, y_arr, sample_weight=np.array(weights))

        coef      = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        win_rate  = float(np.mean(y_arr))
        pred_rate = float(np.mean(probs))

        buckets[bucket] = {
            "coef"      : coef,
            "intercept" : intercept,
            "n_trades"  : len(bt),
            "win_rate"  : round(win_rate, 4),
            "pred_rate" : round(pred_rate, 4),
            "skipped"   : False,
        }
        logger.info(
            f"Calibration [{bucket}]: n={len(bt)}, "
            f"actual={win_rate:.1%}, predicted={pred_rate:.1%}, "
            f"coef={coef:.3f}, intercept={intercept:.3f}"
        )

    result = {
        "updated_at": now_utc.isoformat(),
        "buckets"   : buckets,
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Calibration saved → {CALIBRATION_PATH}")
    return result


def _apply_platt(model_prob: float, coef: float, intercept: float) -> float:
    """Apply Platt scaling: sigmoid(coef * model_prob + intercept)."""
    return float(1 / (1 + math.exp(-(coef * model_prob + intercept))))


# ── Base rate ─────────────────────────────────────────────────────────────────
def compute_base_rate(close: pd.Series, strike_distance: float,
                      horizon: int) -> float:
    """
    Fraction of historical days where close[t+horizon] > close[t]*(1+strike_distance).
    Always uses daily close data regardless of model type.
    """
    if close is None or len(close) == 0:
        return 0.5
    cutoff = close.index[-1] - pd.DateOffset(years=BASE_RATE_LOOKBACK)
    recent = close[close.index >= cutoff]
    hits   = 0
    total  = 0
    for i in range(len(recent) - horizon):
        c0      = float(recent.iloc[i])
        c_future = float(recent.iloc[i + horizon])
        strike  = c0 * (1 + strike_distance)
        total  += 1
        if c_future > strike:
            hits += 1
    rate = hits / total if total > 0 else 0.5
    return float(np.clip(rate, 0.01, 0.99))


# ── Bayesian calibration ──────────────────────────────────────────────────────
def calibrate_probability(model_prob: float, training_base_rate: float,
                           actual_base_rate: float) -> float:
    """
    Log-odds Bayesian update:
      calibrated_log_odds = model_log_odds - training_log_odds + actual_log_odds
    Removes the training prior and replaces with the actual prior.
    """
    p   = float(np.clip(model_prob,        0.001, 0.999))
    tbr = float(np.clip(training_base_rate, 0.001, 0.999))
    abr = float(np.clip(actual_base_rate,   0.001, 0.999))

    model_lo    = math.log(p   / (1 - p))
    training_lo = math.log(tbr / (1 - tbr))
    actual_lo   = math.log(abr / (1 - abr))

    calibrated_lo = model_lo - training_lo + actual_lo
    return float(1 / (1 + math.exp(-calibrated_lo)))


# ── Ticker parsing ────────────────────────────────────────────────────────────
def parse_kalshi_ticker(ticker: str) -> dict | None:
    """
    Parse Kalshi crypto tickers into expiry date + strike.

    Two formats observed:
      - "KXBTCD-27MAR2026-T85000"  → standard DDMONYYYY
      - "KXBTCD-26MAR2717-T68400"  → Kalshi compact YYMONDDH H
    """
    import re
    try:
        parts = ticker.split("-")
        if len(parts) < 3:
            return None

        series = parts[0].upper()
        if "BTC" in series:
            asset, symbol = "BTC", "BTC-USD"
        elif "ETH" in series:
            asset, symbol = "ETH", "ETH-USD"
        else:
            return None

        date_str = parts[1].upper()
        try:
            year_suffix = int(date_str[-4:])
        except ValueError:
            return None
        settlement_hour = None
        if 2020 <= year_suffix <= 2099:
            expiry = datetime.strptime(date_str, "%d%b%Y").date()
        else:
            m = re.match(r'^(\d{2})([A-Z]{3})(\d{2})(\d{2})$', date_str)
            if not m:
                return None
            year            = 2000 + int(m.group(1))
            day             = int(m.group(3))
            settlement_hour = int(m.group(4))   # UTC hour encoded in ticker
            expiry          = datetime.strptime(f"{day:02d}{m.group(2)}{year}", "%d%b%Y").date()

        strike_str = parts[2]
        if not strike_str.startswith("T"):
            return None
        strike = float(strike_str[1:])

        return {"asset": asset, "symbol": symbol,
                "expiry": expiry, "strike": strike,
                "settlement_hour": settlement_hour}
    except Exception:
        return None


# ── Contract features (daily model) ───────────────────────────────────────────
def contract_features(current_price: float, strike: float,
                       expiry: date, vol: float,
                       as_of: date | None = None) -> dict:
    as_of           = as_of or date.today()
    days            = max(1, (expiry - as_of).days)
    strike_distance = (strike / current_price) - 1
    denom           = vol * math.sqrt(days / 252)
    strike_z_score  = strike_distance / denom if denom > 0 else 0.0
    return {
        "days_to_expiry" : days,
        "strike_distance": strike_distance,
        "strike_z_score" : strike_z_score,
    }


# ── EV and Kelly ──────────────────────────────────────────────────────────────
def compute_ev(calibrated_prob: float, market_price_cents: int) -> float:
    mp = market_price_cents / 100
    return calibrated_prob * (1 - mp) - (1 - calibrated_prob) * mp


def compute_kelly(calibrated_prob: float, market_price_cents: int) -> float:
    mp = market_price_cents / 100
    if mp <= 0 or mp >= 1:
        return 0.0
    b = (1 - mp) / mp
    f = (calibrated_prob * b - (1 - calibrated_prob)) / b
    return float(np.clip(f / 2, 0.0, MAX_KELLY))


def compute_ev_no(calibrated_prob: float, no_price_cents: int) -> float:
    np_ = no_price_cents / 100
    p_no = 1 - calibrated_prob
    return p_no * (1 - np_) - calibrated_prob * np_


def compute_kelly_no(calibrated_prob: float, no_price_cents: int) -> float:
    np_ = no_price_cents / 100
    if np_ <= 0 or np_ >= 1:
        return 0.0
    p_no = 1 - calibrated_prob
    b    = (1 - np_) / np_
    f    = (p_no * b - calibrated_prob) / b
    return float(np.clip(f / 2, 0.0, MAX_KELLY))


# ── Score a single contract ───────────────────────────────────────────────────
def score_contract(market: dict, models: dict, asset_dfs: dict) -> list[dict]:
    """
    Score both YES and NO sides of a contract using the appropriate time-bucket model.

    models:    {"daily": (rf, features, tbr) or None,
                "intraday": (rf, features, tbr) or None}
    asset_dfs: {"daily": pd.DataFrame,          # daily OHLC for base rate + daily model
                "hourly": pd.DataFrame or None}  # hourly OHLC for intraday model
    """
    ticker = market.get("ticker", "")
    parsed = parse_kalshi_ticker(ticker)
    if parsed is None:
        return []

    expiry           = parsed["expiry"]
    strike           = parsed["strike"]
    settlement_hour  = parsed.get("settlement_hour")  # UTC hour from compact ticker

    # Compute hours to expiry and filter expired contracts
    close_time_str = market.get("close_time", "")
    hours_to_close = None
    now_utc        = datetime.now(timezone.utc)

    if close_time_str:
        try:
            close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            if close_dt <= now_utc:
                return []
            hours_to_close = (close_dt - now_utc).total_seconds() / 3600
        except Exception:
            pass

    if hours_to_close is None:
        # Fall back to ticker-encoded settlement hour (UTC) if available
        if settlement_hour is not None:
            close_dt = datetime(expiry.year, expiry.month, expiry.day,
                                settlement_hour, 0, 0, tzinfo=timezone.utc)
            if close_dt <= now_utc:
                return []
            hours_to_close = (close_dt - now_utc).total_seconds() / 3600
        elif expiry < date.today():
            return []

    hours_left = hours_to_close if hours_to_close is not None else max(1, (expiry - date.today()).days) * 24

    # Get prices
    yes_ask = market.get("yes_ask") or market.get("last_price")
    yes_bid = market.get("yes_bid")
    no_ask  = market.get("no_ask")
    if yes_ask is None:
        return []
    yes_ask_cents = int(yes_ask)
    if no_ask is not None:
        no_ask_cents = int(no_ask)
    elif yes_bid is not None:
        no_ask_cents = 100 - int(yes_bid)
    else:
        no_ask_cents = 100 - yes_ask_cents
    if not (1 <= yes_ask_cents <= 99) or not (1 <= no_ask_cents <= 99):
        return []

    # Select model based on time to expiry
    use_intraday = (
        hours_left <= 24
        and models.get("intraday") is not None
        and asset_dfs.get("hourly") is not None
    )

    if use_intraday:
        model, feature_names, training_base_rate = models["intraday"]
        df_hourly = asset_dfs["hourly"]

        feat_df  = compute_features_hourly(df_hourly)
        last_row = feat_df.iloc[[-1]]
        if last_row.isnull().any(axis=1).values[0]:
            return []

        current_price   = float(df_hourly["Close"].iloc[-1])
        vol             = float(feat_df["vol_24h"].iloc[-1])
        if math.isnan(vol):
            vol = 0.5

        strike_distance = (strike / current_price) - 1
        denom           = vol * math.sqrt(max(hours_left, 0.5) / 8760)
        z_score         = strike_distance / denom if denom > 0 else 0.0

        row = last_row.copy()
        row["hours_to_expiry"] = hours_left
        row["strike_distance"] = strike_distance
        row["strike_z_score"]  = z_score
        model_type = "intraday"

    else:
        daily_tuple = models.get("daily")
        if daily_tuple is None:
            return []
        model, feature_names, training_base_rate = daily_tuple
        df_daily = asset_dfs["daily"]

        feat_df  = compute_features(df_daily)
        last_row = feat_df.iloc[[-1]]
        if last_row.isnull().any(axis=1).values[0]:
            return []

        current_price = float(df_daily["Close"].iloc[-1])
        vol           = float(hist_vol(df_daily["Close"]).iloc[-1])
        if math.isnan(vol):
            vol = 0.5

        cf  = contract_features(current_price, strike, expiry, vol)
        row = last_row.copy()
        for k, v in cf.items():
            row[k] = v
        strike_distance = cf["strike_distance"]
        model_type = "daily"

    # Align feature columns
    try:
        row = row[feature_names]
    except KeyError:
        return []
    if row.isnull().any(axis=1).values[0]:
        return []

    model_prob   = float(model.predict_proba(row)[0][1])

    # Apply paper-trade Platt calibration if available for this bucket
    _cal = models.get("calibration")
    if _cal:
        _bp = _cal.get("buckets", {}).get(model_type, {})
        if _bp and not _bp.get("skipped") and "coef" in _bp:
            model_prob = _apply_platt(model_prob, _bp["coef"], _bp["intercept"])

    horizon_days = max(1, (expiry - date.today()).days)
    strike_distance = (strike / current_price) - 1

    # Base rate always computed on daily data (historical frequency)
    daily_df_for_br = asset_dfs.get("daily")
    daily_close = daily_df_for_br["Close"] if daily_df_for_br is not None and len(daily_df_for_br) > 0 else pd.Series(dtype=float)
    base_rate   = compute_base_rate(daily_close, strike_distance, horizon_days)
    cal_prob    = calibrate_probability(model_prob, training_base_rate, base_rate)

    base = {
        "ticker"         : ticker,
        "asset"          : parsed["asset"],
        "expiry"         : str(expiry),
        "strike"         : strike,
        "current_price"  : round(current_price, 2),
        "strike_distance": round(strike_distance * 100, 1),
        "days_to_expiry" : horizon_days,
        "hours_to_expiry": round(hours_left, 1),
        "model_prob"     : round(model_prob, 4),
        "base_rate"      : round(base_rate, 4),
        "calibrated_prob": round(cal_prob, 4),
        "model_type"     : model_type,
    }

    results = []

    ev_yes    = compute_ev(cal_prob, yes_ask_cents)
    kelly_yes = compute_kelly(cal_prob, yes_ask_cents)
    edge_yes  = cal_prob - yes_ask_cents / 100
    results.append({**base,
        "side"      : "YES",
        "price"     : yes_ask_cents,
        "edge"      : round(edge_yes, 4),
        "ev"        : round(ev_yes, 4),
        "kelly_pct" : round(kelly_yes * 100, 1),
    })

    ev_no    = compute_ev_no(cal_prob, no_ask_cents)
    kelly_no = compute_kelly_no(cal_prob, no_ask_cents)
    edge_no  = (1 - cal_prob) - no_ask_cents / 100
    results.append({**base,
        "side"      : "NO",
        "price"     : no_ask_cents,
        "edge"      : round(edge_no, 4),
        "ev"        : round(ev_no, 4),
        "kelly_pct" : round(kelly_no * 100, 1),
    })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",        action="store_true",
                        help="Retrain daily model before scoring")
    parser.add_argument("--train-all",    action="store_true",
                        help="Retrain both daily + intraday models")
    parser.add_argument("--train-intraday", action="store_true",
                        help="Retrain intraday model only")
    parser.add_argument("--bankroll",     type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--min-ev",       type=float, default=MIN_EV)
    parser.add_argument("--min-edge",     type=float, default=MIN_EDGE)
    args = parser.parse_args()

    # ── Kalshi client ──
    client = KalshiClient()
    if not client.dry_run:
        client.login()
    try:
        balance_cents = client.get_balance().get("balance", 0)
    except Exception:
        balance_cents = 0
    bankroll = args.bankroll or balance_cents / 100 or DEFAULT_BANKROLL
    mode     = "DRY RUN" if client.dry_run else "LIVE"

    print(f"\n{'='*65}")
    print(f"  Kalshi Crypto Signal Report  [{mode}]  —  {date.today()}")
    print(f"  Bankroll: ${bankroll:,.2f}")
    print(f"{'='*65}")

    # ── Train ──
    if args.train_all:
        build_crypto_model()
        build_intraday_model()
    elif args.train:
        build_crypto_model()
    elif args.train_intraday:
        build_intraday_model()

    # ── Load models ──
    models = load_crypto_models()
    if models.get("daily") is None:
        logger.error("No daily model found. Run with --train or --train-all first.")
        return

    has_intraday = models.get("intraday") is not None
    logger.info(f"Models loaded — daily: yes, intraday: {'yes' if has_intraday else 'no'}")

    # ── Asset data ──
    asset_dfs_by_symbol: dict[str, dict] = {}
    for symbol in CRYPTO_ASSETS:
        logger.info(f"Fetching {symbol} daily data ...")
        daily_df = download_crypto(symbol, INFERENCE_PERIOD)
        hourly_df = None
        if has_intraday:
            logger.info(f"Fetching {symbol} hourly data ...")
            hourly_df = download_crypto_hourly(symbol)
        asset_dfs_by_symbol[symbol] = {"daily": daily_df, "hourly": hourly_df}

    # ── Score contracts ──
    recommendations = []
    all_results     = []

    for symbol, series in KALSHI_SERIES.items():
        markets = client.get_markets(series_ticker=series, status="open")
        logger.info(f"{series}: {len(markets)} open contracts")
        asset_dfs = asset_dfs_by_symbol[symbol]

        for market in markets:
            for result in score_contract(market, models, asset_dfs):
                all_results.append(result)
                if result["ev"] >= args.min_ev and result["edge"] >= args.min_edge:
                    recommendations.append(result)

    # ── All contracts scanned ──
    print(f"\n[ALL CONTRACTS SCANNED]")
    print(f"  {'Ticker':<35} {'Side':>4} {'Mdl':>7} {'Price':>6} {'Cal%':>6} "
          f"{'Edge':>6} {'EV':>6} {'Kelly':>6}")
    print(f"  {'-'*90}")
    for r in sorted(all_results, key=lambda x: x["ev"], reverse=True):
        flag = " **" if (r["ev"] >= args.min_ev and r["edge"] >= args.min_edge) else "   "
        print(f"{flag} {r['ticker']:<35} "
              f"{r['side']:>4}  "
              f"{'intra' if r['model_type']=='intraday' else 'daily':>5}  "
              f"{r['price']:>4}¢  "
              f"{r['calibrated_prob']*100:>5.1f}%  "
              f"{r['edge']*100:>+5.1f}%  "
              f"{r['ev']:>+.3f}  "
              f"{r['kelly_pct']:>5.1f}%")

    # ── Recommendations ──
    print(f"\n[RECOMMENDATIONS]  (EV ≥ {args.min_ev}, edge ≥ {args.min_edge*100:.0f}pp)")
    if not recommendations:
        print("  No contracts meet the threshold today.")
    else:
        recommendations.sort(key=lambda x: x["ev"], reverse=True)
        for r in recommendations:
            contracts = max(1, int(bankroll * (r["kelly_pct"] / 100)))
            cost_usd  = contracts * r["price"] / 100
            side      = r["side"]
            print(f"\n  {r['ticker']}  [{side}]  [{r['model_type']} model]")
            print(f"    Strike      : ${r['strike']:,.0f}  "
                  f"({r['strike_distance']:+.1f}% from current ${r['current_price']:,.0f})")
            print(f"    Expires     : {r['expiry']}  "
                  f"({r['hours_to_expiry']:.1f}h)")
            print(f"    Price       : {r['price']}¢  "
                  f"(market implies {r['price'] if side=='YES' else 100-r['price']}% YES)")
            print(f"    Base rate   : {r['base_rate']*100:.1f}%")
            print(f"    Model prob  : {r['model_prob']*100:.1f}%")
            print(f"    Calibrated  : {r['calibrated_prob']*100:.1f}%  ← YES probability")
            print(f"    Edge        : {r['edge']*100:+.1f}pp")
            print(f"    EV          : {r['ev']:+.3f} per $1 risked")
            print(f"    Half-Kelly  : {r['kelly_pct']:.1f}% of bankroll")
            print(f"    Order       : BUY {contracts} {side} contract(s) "
                  f"@ {r['price']}¢  (cost ~${cost_usd:.2f})")

    print()


if __name__ == "__main__":
    main()
