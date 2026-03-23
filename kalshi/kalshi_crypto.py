"""
kalshi_crypto.py — Crypto prediction pipeline for Kalshi markets

Full pipeline:
  1. Train a Random Forest on BTC-USD + ETH-USD data with contract-specific features
  2. For each open Kalshi crypto contract:
       a. Compute base rate (historical frequency of hitting the strike)
       b. Get model probability
       c. Bayesian calibration (log-odds update: swap training prior for actual prior)
       d. Compute EV
       e. Compute half-Kelly position size
  3. Print ranked recommendations

Usage:
  python kalshi_crypto.py              # score live contracts (dry-run if no creds)
  python kalshi_crypto.py --train      # retrain model then score
  python kalshi_crypto.py --bankroll 1000
"""

import argparse
import json
import logging
import math
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Allow importing features.py from the stocks/ sibling directory
sys.path.insert(0, str(Path(__file__).parent.parent / "stocks"))
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
HORIZONS           = [1, 5, 7, 10]       # trading days to simulate contracts
HISTORY_PERIOD     = "5y"
INFERENCE_PERIOD   = "2y"
HIST_VOL_WINDOW    = 30                  # days for rolling volatility
BASE_RATE_LOOKBACK = 3                   # years for base-rate estimation
MODEL_PATH         = "model_crypto.joblib"
FEATURES_PATH      = "features_crypto.csv"
METADATA_PATH      = "model_crypto_meta.json"
PREDICTIONS_LOG    = "predictions_log.jsonl"
RF_TREES           = 300
RF_MAX_DEPTH       = 12
RF_MIN_LEAF        = 5
RANDOM_STATE       = 42
N_FOLDS            = 5
MIN_TRAIN_FRAC     = 0.5
MIN_EV             = 0.05   # minimum EV per dollar to recommend
MIN_EDGE           = 0.05   # minimum (calibrated_prob - market_price)
MAX_KELLY          = 0.25   # cap half-Kelly at 25% of bankroll
DEFAULT_BANKROLL   = 200
# ──────────────────────────────────────────────────────────────────────────────


# ── Data ──────────────────────────────────────────────────────────────────────
def download_crypto(symbol: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def hist_vol(close: pd.Series, window: int = HIST_VOL_WINDOW) -> pd.Series:
    """Annualised rolling volatility from log returns."""
    return close.pct_change().apply(np.log1p).rolling(window).std() * math.sqrt(252)


# ── Training data generation ──────────────────────────────────────────────────
def generate_training_samples(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    For each date t, strike offset, and horizon:
      - tech features from compute_features(df).iloc[t]  (114 cols)
      - contract features: days_to_expiry, strike_distance, strike_z_score
      - label: 1 if close[t+horizon] > close[t]*(1+offset), else 0
    Returns a DataFrame sorted by date index.
    """
    logger.info(f"  Generating training samples for {symbol} ...")

    feat_df = compute_features(df).dropna(how="all")
    vol_s   = hist_vol(df["Close"])
    close   = df["Close"]

    max_horizon = max(HORIZONS)
    rows = []

    valid_idx = feat_df.dropna().index
    # Need at least max_horizon future days
    valid_idx = valid_idx[valid_idx <= close.index[-max_horizon - 1]]

    for dt in valid_idx:
        feat_row = feat_df.loc[dt]
        if feat_row.isnull().any():
            continue

        c0  = float(close.loc[dt])
        vol = float(vol_s.loc[dt]) if not pd.isna(vol_s.loc[dt]) else 0.5

        for horizon in HORIZONS:
            # Find close price horizon trading-days ahead
            future_idx = close.index.get_loc(dt) + horizon
            if future_idx >= len(close):
                continue
            c_future = float(close.iloc[future_idx])

            for offset in STRIKE_OFFSETS:
                strike = c0 * (1 + offset)
                label  = int(c_future > strike)

                days_to_expiry = horizon
                strike_distance = offset
                denom = vol * math.sqrt(horizon / 252)
                strike_z_score = offset / denom if denom > 0 else 0.0

                row = feat_row.to_dict()
                row["days_to_expiry"]  = days_to_expiry
                row["strike_distance"] = strike_distance
                row["strike_z_score"]  = strike_z_score
                row["label"]           = label
                row["_date"]           = dt
                rows.append(row)

    result = pd.DataFrame(rows).sort_values("_date").reset_index(drop=True)
    logger.info(f"    {len(result):,} samples, label mean = {result['label'].mean():.3f}")
    return result


# ── Model ─────────────────────────────────────────────────────────────────────
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
    """Train on BTC + ETH, run walk-forward CV, save model and metadata."""
    logger.info("Building crypto model ...")

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

    training_base_rate = float(y.mean())
    logger.info(f"  Combined: {len(X):,} samples, "
                f"base rate = {training_base_rate:.3f}")

    logger.info("  Running walk-forward CV ...")
    cv_roc_auc = _walk_forward_cv(X, y, dates)
    logger.info(f"  CV ROC-AUC = {cv_roc_auc:.4f}")

    logger.info("  Training final model on all data ...")
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y)

    # Save
    joblib.dump(rf, MODEL_PATH)
    pd.Series(feature_names).to_csv(FEATURES_PATH, index=False, header=False)
    meta = {
        "training_base_rate": training_base_rate,
        "trained_on"        : str(date.today()),
        "n_samples"         : len(X),
        "cv_roc_auc"        : round(cv_roc_auc, 4),
        "feature_names"     : feature_names,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Model saved → {MODEL_PATH}")

    return rf, feature_names, training_base_rate


def load_crypto_model() -> tuple:
    """Load saved model + metadata. Returns (None, None, None) if missing."""
    if not Path(MODEL_PATH).exists() or not Path(METADATA_PATH).exists():
        return None, None, None
    rf            = joblib.load(MODEL_PATH)
    feature_names = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    training_base_rate = meta["training_base_rate"]
    logger.info(f"Loaded model trained on {meta['trained_on']} "
                f"({meta['n_samples']:,} samples, "
                f"CV ROC-AUC={meta['cv_roc_auc']})")
    return rf, feature_names, training_base_rate


# ── Base rate ─────────────────────────────────────────────────────────────────
def compute_base_rate(close: pd.Series, strike_distance: float,
                      horizon: int) -> float:
    """
    Fraction of historical days where close[t+horizon] > close[t]*(1+strike_distance)
    using the last BASE_RATE_LOOKBACK years of data.
    """
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

    This removes the training prior and replaces it with the actual prior,
    preserving the model's discriminative signal.
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
    Parse e.g. 'KXBTCD-26MAR2026-T85000' into:
      {"asset": "BTC", "symbol": "BTC-USD", "expiry": date, "strike": 85000.0}
    """
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

        expiry = datetime.strptime(parts[1].upper(), "%d%b%Y").date()

        strike_str = parts[2]
        if not strike_str.startswith("T"):
            return None
        strike = float(strike_str[1:])

        return {"asset": asset, "symbol": symbol,
                "expiry": expiry, "strike": strike}
    except Exception:
        return None


# ── Contract features ─────────────────────────────────────────────────────────
def contract_features(current_price: float, strike: float,
                       expiry: date, vol: float,
                       as_of: date | None = None,
                       days_override: float | None = None) -> dict:
    as_of           = as_of or date.today()
    days            = days_override if days_override is not None else max(0, (expiry - as_of).days)
    strike_distance = (strike / current_price) - 1
    denom           = vol * math.sqrt(max(days, 1/24) / 252)
    strike_z_score  = strike_distance / denom if denom > 0 else 0.0
    return {
        "days_to_expiry" : days,
        "strike_distance": strike_distance,
        "strike_z_score" : strike_z_score,
    }


# ── EV and Kelly ──────────────────────────────────────────────────────────────
def compute_ev(calibrated_prob: float, market_price_cents: int) -> float:
    """
    True EV per dollar staked on the yes side.
    EV = (calibrated_prob - market_price) / market_price
    e.g. 35% cal prob at 20¢ market → EV = 0.75 (75c profit per $1 risked)
         95% cal prob at 80¢ market → EV = 0.19 (19c profit per $1 risked)
    """
    mp = market_price_cents / 100
    if mp <= 0:
        return 0.0
    return (calibrated_prob - mp) / mp


def compute_kelly(calibrated_prob: float, market_price_cents: int) -> float:
    """Half-Kelly fraction, capped at MAX_KELLY."""
    mp = market_price_cents / 100
    if mp <= 0 or mp >= 1:
        return 0.0
    b = (1 - mp) / mp           # net odds on a $1 yes bet
    f = (calibrated_prob * b - (1 - calibrated_prob)) / b
    return float(np.clip(f / 2, 0.0, MAX_KELLY))


# ── Score a single contract ───────────────────────────────────────────────────
def score_contract(market: dict, model: RandomForestClassifier,
                   feature_names: list[str], df_asset: pd.DataFrame,
                   training_base_rate: float,
                   feat_cache: dict | None = None) -> dict | None:
    ticker = market.get("ticker", "")

    # Determine asset from series prefix (KXBTCD → BTC, KXETHD → ETH)
    series = ticker.split("-")[0].upper() if ticker else ""
    if "BTC" in series:
        asset = "BTC"
    elif "ETH" in series:
        asset = "ETH"
    else:
        return None

    # --- Expiry: parse close_time as full datetime to get fractional days ---
    close_time_str = market.get("close_time")
    close_datetime = None
    if close_time_str:
        try:
            close_datetime = datetime.fromisoformat(
                close_time_str.replace("Z", "+00:00")
            )
            expiry = close_datetime.date()
        except Exception:
            expiry = None
    else:
        expiry = None

    if expiry is None:
        parsed = parse_kalshi_ticker(ticker)
        expiry = parsed["expiry"] if parsed else None

    if expiry is None or expiry < date.today():
        return None

    # Fractional days remaining (hours / 24), minimum 1 hour
    if close_datetime is not None:
        now_utc = datetime.now(tz=close_datetime.tzinfo)
        hours_remaining = max(1.0, (close_datetime - now_utc).total_seconds() / 3600)
        days_remaining = hours_remaining / 24
    else:
        days_remaining = max(1.0, (expiry - date.today()).days)

    # --- Strike: prefer floor_strike field, fall back to ticker parsing ---
    strike = market.get("floor_strike")
    if strike is None:
        parsed = parse_kalshi_ticker(ticker)
        strike = parsed["strike"] if parsed else None
    if strike is None:
        return None
    strike = float(strike)

    # --- Market price in cents: try dollars fields first, then fp, then cents ---
    yes_ask_dollars = market.get("yes_ask_dollars")
    if yes_ask_dollars is not None:
        try:
            market_price_cents = round(float(yes_ask_dollars) * 100)
        except (ValueError, TypeError):
            market_price_cents = None
    else:
        # Fixed-point (×100) → cents
        yes_ask_fp = market.get("yes_ask_fp")
        if yes_ask_fp is not None:
            market_price_cents = round(yes_ask_fp / 100)
        else:
            # Already in cents
            yes_ask = market.get("yes_ask") or market.get("last_price")
            market_price_cents = int(yes_ask) if yes_ask is not None else None

    if market_price_cents is None or market_price_cents <= 0:
        return None

    # Technical features from latest close (use cache if provided)
    if feat_cache is not None and id(df_asset) in feat_cache:
        last_row, current_price, vol = feat_cache[id(df_asset)]
    else:
        feat_df = compute_features(df_asset)
        last_row = feat_df.iloc[[-1]]
        if last_row.isnull().any(axis=1).values[0]:
            return None
        current_price = float(df_asset["Close"].iloc[-1])
        vol           = float(hist_vol(df_asset["Close"]).iloc[-1])
        if math.isnan(vol):
            vol = 0.5
        if feat_cache is not None:
            feat_cache[id(df_asset)] = (last_row, current_price, vol)


    # Contract-specific features (use fractional days for accuracy)
    cf = contract_features(current_price, strike, expiry, vol,
                           days_override=days_remaining)

    # Assemble feature vector
    row = last_row.copy()
    for k, v in cf.items():
        row[k] = v
    row = row[feature_names]
    if row.isnull().any(axis=1).values[0]:
        return None

    # Model probability
    model_prob = float(model.predict_proba(row)[0][1])

    # Base rate (historical frequency of close > strike at this distance/horizon)
    strike_distance = cf["strike_distance"]
    base_rate       = compute_base_rate(df_asset["Close"], strike_distance, max(1, int(round(days_remaining))))

    # Bayesian calibration
    cal_prob = calibrate_probability(model_prob, training_base_rate, base_rate)

    # EV and Kelly
    ev    = compute_ev(cal_prob, market_price_cents)
    kelly = compute_kelly(cal_prob, market_price_cents)
    edge  = cal_prob - market_price_cents / 100

    hours_left  = round(days_remaining * 24, 1)
    ev_per_day  = round(ev / max(days_remaining, 1/24), 4)
    return {
        "ticker"          : ticker,
        "asset"           : asset,
        "expiry"          : str(expiry),
        "close_time"      : close_time_str or "",
        "hours_left"      : hours_left,
        "ev_per_day"      : ev_per_day,
        "strike"          : strike,
        "current_price"   : round(current_price, 2),
        "strike_distance" : round(strike_distance * 100, 1),   # as %
        "days_to_expiry"  : round(days_remaining, 2),
        "market_price"    : market_price_cents,                 # cents
        "model_prob"      : round(model_prob, 4),
        "base_rate"       : round(base_rate, 4),
        "calibrated_prob" : round(cal_prob, 4),
        "edge"            : round(edge, 4),
        "ev"              : round(ev, 4),
        "kelly_pct"       : round(kelly * 100, 1),
    }


# ── Prediction logging ────────────────────────────────────────────────────────
def log_predictions(results: list[dict], recommended_tickers: set[str]) -> None:
    """Append scored contracts to predictions_log.jsonl for later back-testing."""
    run_ts = datetime.now(timezone.utc).isoformat()
    path   = Path(PREDICTIONS_LOG)
    with open(path, "a") as f:
        for r in results:
            entry = {
                "run_ts"          : run_ts,
                "ticker"          : r["ticker"],
                "asset"           : r["asset"],
                "expiry"          : r["expiry"],
                "hours_left"      : r["hours_left"],
                "strike"          : r["strike"],
                "current_price"   : r["current_price"],
                "strike_distance" : r["strike_distance"],
                "market_price"    : r["market_price"],
                "model_prob"      : r["model_prob"],
                "base_rate"       : r["base_rate"],
                "calibrated_prob" : r["calibrated_prob"],
                "edge"            : r["edge"],
                "ev"              : r["ev"],
                "ev_per_day"      : r["ev_per_day"],
                "kelly_pct"       : r["kelly_pct"],
                "recommended"     : r["ticker"] in recommended_tickers,
            }
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Logged {len(results)} predictions → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",            action="store_true",
                        help="Retrain model before scoring")
    parser.add_argument("--bankroll",         type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--bankroll-daily",   type=float, default=None,
                        help="Bankroll allocation for <24h contracts")
    parser.add_argument("--bankroll-weekly",  type=float, default=None,
                        help="Bankroll allocation for 5+ day contracts")
    parser.add_argument("--min-ev",           type=float, default=MIN_EV)
    parser.add_argument("--min-edge",         type=float, default=MIN_EDGE)
    args = parser.parse_args()


    # ── Kalshi client ──
    client = KalshiClient()
    if not client.dry_run:
        client.login()
    balance_cents   = client.get_balance().get("balance", 0)
    total_bankroll  = args.bankroll or balance_cents / 100
    bankroll_daily  = args.bankroll_daily  or total_bankroll * 0.40
    bankroll_weekly = args.bankroll_weekly or total_bankroll * 0.60
    mode            = "DRY RUN" if client.dry_run else "LIVE"

    print(f"\n{'='*65}")
    print(f"  Kalshi Crypto Signal Report  [{mode}]  —  {date.today()}")
    print(f"  Bankroll: ${total_bankroll:,.2f}  "
          f"(Daily: ${bankroll_daily:,.2f}  |  Weekly: ${bankroll_weekly:,.2f})")
    print(f"{'='*65}")

    # ── Model ──
    if args.train or not Path(MODEL_PATH).exists():
        model, feature_names, training_base_rate = build_crypto_model()
    else:
        model, feature_names, training_base_rate = load_crypto_model()
        if model is None:
            logger.error("No model found. Run with --train first.")
            return

    # ── Asset data ──
    asset_data = {}
    for symbol in CRYPTO_ASSETS:
        logger.info(f"Fetching {symbol} ...")
        asset_data[symbol] = download_crypto(symbol, INFERENCE_PERIOD)

    # ── Score contracts ──
    recommendations = []
    all_results     = []
    feat_cache      = {}   # cache compute_features per asset — big speedup

    for symbol, series in KALSHI_SERIES.items():
        markets = client.get_markets(series_ticker=series, status="open")
        logger.info(f"{series}: {len(markets)} open contracts")
        df_asset = asset_data[symbol]

        for market in markets:
            result = score_contract(
                market, model, feature_names, df_asset, training_base_rate,
                feat_cache=feat_cache
            )
            if result is None:
                continue
            all_results.append(result)
            if result["ev"] >= args.min_ev and result["edge"] >= args.min_edge:
                recommendations.append(result)

    # ── All contracts scanned ──
    print(f"\n[ALL CONTRACTS SCANNED]")
    print(f"  {'Ticker':<35} {'Mkt%':>5} {'Base%':>6} {'Cal%':>6} "
          f"{'Edge':>6} {'EV':>6} {'Kelly':>6}")
    print(f"  {'-'*80}")
    for r in sorted(all_results, key=lambda x: x["ev"], reverse=True):
        flag = " **" if (r["ev"] >= args.min_ev and r["edge"] >= args.min_edge) else "   "
        print(f"{flag} {r['ticker']:<35} "
              f"{r['market_price']:>4}¢  "
              f"{r['base_rate']*100:>5.1f}%  "
              f"{r['calibrated_prob']*100:>5.1f}%  "
              f"{r['edge']*100:>+5.1f}%  "
              f"{r['ev']:>+.3f}  "
              f"{r['kelly_pct']:>5.1f}%")

    # ── Recommendations: two sections (<24h and 5+ days) ──
    print(f"\n[RECOMMENDATIONS]  (EV ≥ {args.min_ev}, edge ≥ {args.min_edge*100:.0f}pp)")

    def print_table(contracts_list):
        if not contracts_list:
            print("  None")
            return
        hdr = f"  {'Ticker':<38} {'Strike':>10} {'Dist':>6} {'Mkt':>4} {'Cal':>5} {'Edge':>6} {'EV':>6} {'Bet$':>6}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in contracts_list:
            br       = bankroll_daily if r["hours_left"] <= 24 else bankroll_weekly
            n        = max(1, int(br * (r["kelly_pct"] / 100)))
            cost_usd = n * r["market_price"] / 100
            print(f"  {r['ticker']:<38} "
                  f"${r['strike']:>9,.0f} "
                  f"{r['strike_distance']:>+5.1f}% "
                  f"{r['market_price']:>3}¢ "
                  f"{r['calibrated_prob']*100:>4.0f}% "
                  f"{r['edge']*100:>+5.1f}% "
                  f"{r['ev']:>+.2f} "
                  f"${cost_usd:>5.2f}")

    if not recommendations:
        print("  No contracts meet the threshold today.")
    else:
        short = [r for r in recommendations if r["hours_left"] <= 24]
        long_ = [r for r in recommendations if r["hours_left"] >  24]

        for label, group in [("< 24 HOURS", short), ("> 24 HOURS", long_)]:
            high  = sorted([r for r in group if r["market_price"] >= 50], key=lambda x: -x["kelly_pct"])
            shots = sorted([r for r in group if r["market_price"] <  50], key=lambda x: -x["kelly_pct"])
            if not high and not shots:
                continue
            print(f"\n  {'─'*10} {label} {'─'*10}")
            if high:
                print(f"\n  [High Confidence >50¢]")
                print_table(high)
            if shots:
                print(f"\n  [Long Shots <50¢]")
                print_table(shots)

    # ── Top 5 Summaries ──
    def print_top5(label, pool, br, key="ev"):
        print(f"\n{'='*65}")
        print(f"  {label}  (≥20¢)  —  Pool: ${br:,.2f}")
        print(f"{'='*65}")
        picks = sorted(
            [r for r in pool if r["market_price"] >= 20],
            key=lambda x: -x[key]
        )[:5]
        if not picks:
            print("  None meet the ≥20¢ filter.")
            return
        total_cost = 0
        for i, r in enumerate(picks, 1):
            n        = max(1, int(br * (r["kelly_pct"] / 100)))
            cost_usd = n * r["market_price"] / 100
            total_cost += cost_usd
            print(f"\n  #{i}  {r['ticker']}")
            print(f"      Strike {r['strike_distance']:+.1f}% away  |  "
                  f"Mkt {r['market_price']}¢ → Cal {r['calibrated_prob']*100:.0f}%  |  "
                  f"Edge {r['edge']*100:+.1f}pp  |  EV {r['ev']:+.2f}  |  EV/day {r['ev_per_day']:+.2f}  |  Kelly {r['kelly_pct']:.1f}%")
            print(f"      BUY {n} YES @ {r['market_price']}¢  (~${cost_usd:.2f})  |  "
                  f"Expires in {r['hours_left']:.0f}h")
        print(f"\n  Total cost if all placed: ~${total_cost:.2f} of ${br:,.2f} pool")

    # ── Log predictions for back-testing ──
    recommended_tickers = {r["ticker"] for r in recommendations}
    log_predictions(all_results, recommended_tickers)

    short_rec = [r for r in recommendations if r["hours_left"] <= 24]
    long_rec  = [r for r in recommendations if r["hours_left"] >  24]

    print_top5("TOP 5 OVERALL  (ranked by EV/day)", recommendations, total_bankroll, key="ev_per_day")
    print_top5("TOP 5  —  < 24 HOURS", short_rec, bankroll_daily, key="ev")
    print_top5("TOP 5  —  > 24 HOURS", long_rec, bankroll_weekly, key="ev")
    print()


if __name__ == "__main__":
    main()
