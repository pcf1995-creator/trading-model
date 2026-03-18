"""
features.py — Stock indicator feature engineering + Random Forest buy/sell classifier

Indicators (38 groups, 100+ signals):
  1-2.   SMA / EMA (6 periods each + price ratios)
  3.     SMA & EMA crossovers
  4.     RSI (3 periods)
  5.     MACD (line, signal, histogram)
  6.     Bollinger Bands (3 periods: %B, width)
  7.     ATR (3 periods, raw + % of price)
  8.     Stochastic %K / %D (2 periods)
  9.     Williams %R (2 periods)
  10.    CCI (2 periods)
  11.    Rate of Change / Momentum (5 periods)
  12.    OBV + OBV moving average + ratio
  13.    Volume ratio (2 periods)
  14.    ADX + Plus/Minus DI
  15.    Candlestick structure (body, shadows, gap, HL range)
  16.    Rolling return stats — mean, std, skew (3 periods)
  17.    Distance from N-day high/low (2 periods)
  18.    Parabolic SAR proxy
  19.    Money Flow Index (MFI)
  20.    VWAP deviation
  21.    Chaikin Money Flow (CMF)
  22.    Ease of Movement (EOM)
  23.    Force Index
  24.    Keltner Channel (position + width)
  25.    Donchian Channel width
  26.    TEMA (Triple EMA)
  27.    DEMA (Double EMA)
  28.    Hull Moving Average proxy
  29.    Price oscillator (PPO)
  30.    Trix
  31.    Ultimate Oscillator
  32.    Mass Index
  33.    Vortex Indicator
  34.    Aroon Oscillator (2 periods)
  35.    Coppock Curve
  36.    Elder Ray (Bull/Bear Power)
  37.    DPO (Detrended Price Oscillator)
  38.    Lagged close returns (5 lags)

Label: Buy = 1 if close rises >2% in 5 trading days, else 0
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    precision_recall_curve, f1_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
TICKERS       = [
    # ── Original ──────────────────────────────────────────
    "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META", "NFLX",
    "JPM", "JNJ", "XOM", "WMT",
    # ── Broad ETFs ────────────────────────────────────────
    "SPY", "QQQ", "IWM",           # S&P 500, Nasdaq 100, Russell 2000
    "GLD", "TLT",                  # gold, long bonds
    # ── Semiconductors / Tech ─────────────────────────────
    "AMD", "QCOM", "ORCL", "CRM", "ADBE",
    # ── Finance ───────────────────────────────────────────
    "GS", "BAC", "V", "MA",
    # ── Healthcare / Biotech ──────────────────────────────
    "UNH", "PFE", "ABBV", "AMGN",
    # ── Industrials / Aerospace ───────────────────────────
    "CAT", "BA", "HON", "GE",
    # ── Energy ────────────────────────────────────────────
    "CVX", "COP",
    # ── Consumer ──────────────────────────────────────────
    "MCD", "NKE", "HD", "COST",
]
PERIOD        = "10y"       # history to download
FUTURE_DAYS   = 5           # days ahead for label
BUY_THRESHOLD = 0.02        # 2 % gain → Buy
TEST_SIZE     = 0.2         # fraction held out for evaluation
RF_TREES      = 200
RF_MAX_DEPTH  = 12
RANDOM_STATE  = 42
# ──────────────────────────────────────────────────────────────────────────────


def download_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ── Helper: Wilder smoothing (used in RSI, ATR, ADX) ─────────────────────────
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


# ── Feature engineering ───────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    tr = _true_range(h, l, c)

    # 1. SMA + price/SMA ratio
    for p in [5, 10, 20, 50, 100, 200]:
        sma = c.rolling(p).mean()
        feat[f"SMA_{p}"]       = sma
        feat[f"Price_SMA_{p}"] = c / sma

    # 2. EMA + price/EMA ratio
    for p in [5, 10, 20, 50, 100, 200]:
        ema = c.ewm(span=p, adjust=False).mean()
        feat[f"EMA_{p}"]       = ema
        feat[f"Price_EMA_{p}"] = c / ema

    # 3. Crossovers (SMA & EMA)
    feat["SMA_5_20_cross"]   = c.rolling(5).mean()  - c.rolling(20).mean()
    feat["SMA_10_50_cross"]  = c.rolling(10).mean() - c.rolling(50).mean()
    feat["SMA_20_200_cross"] = c.rolling(20).mean() - c.rolling(200).mean()
    feat["EMA_12_26_cross"]  = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()

    # 4. RSI (3 periods — Wilder method)
    delta = c.diff()
    for p in [7, 14, 21]:
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = _wilder_smooth(gain, p)
        avg_loss = _wilder_smooth(loss, p)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        feat[f"RSI_{p}"] = 100 - 100 / (1 + rs)

    # 5. MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    feat["MACD_line"]      = ema12 - ema26
    feat["MACD_signal"]    = feat["MACD_line"].ewm(span=9, adjust=False).mean()
    feat["MACD_histogram"] = feat["MACD_line"] - feat["MACD_signal"]

    # 6. Bollinger Bands (3 periods)
    for p in [10, 20, 50]:
        sma = c.rolling(p).mean()
        std = c.rolling(p).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        band_width = upper - lower
        feat[f"BB_pct_{p}"]   = (c - lower) / band_width.replace(0, np.nan)
        feat[f"BB_width_{p}"] = band_width / sma

    # 7. ATR (3 periods, raw + % of price)
    for p in [7, 14, 21]:
        atr = _wilder_smooth(tr, p)
        feat[f"ATR_{p}"]     = atr
        feat[f"ATR_pct_{p}"] = atr / c

    # 8. Stochastic %K / %D (2 periods)
    for p in [14, 21]:
        ll = l.rolling(p).min()
        hh = h.rolling(p).max()
        k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        feat[f"Stoch_K_{p}"] = k
        feat[f"Stoch_D_{p}"] = k.rolling(3).mean()

    # 9. Williams %R (2 periods)
    for p in [14, 21]:
        hh = h.rolling(p).max()
        ll = l.rolling(p).min()
        feat[f"Williams_R_{p}"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)

    # 10. CCI (2 periods)
    tp = (h + l + c) / 3
    for p in [14, 20]:
        sma_tp = tp.rolling(p).mean()
        mad    = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        feat[f"CCI_{p}"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # 11. Rate of Change (5 periods)
    for p in [1, 5, 10, 20, 60]:
        feat[f"ROC_{p}"] = c.pct_change(p)

    # 12. OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    feat["OBV"]          = obv
    feat["OBV_SMA20"]    = obv.rolling(20).mean()
    feat["OBV_ratio"]    = obv / obv.rolling(20).mean().replace(0, np.nan)

    # 13. Volume ratio (2 periods)
    feat["Vol_ratio_10"] = v / v.rolling(10).mean().replace(0, np.nan)
    feat["Vol_ratio_20"] = v / v.rolling(20).mean().replace(0, np.nan)

    # 14. ADX + DI (14-period)
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm[plus_dm  < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    p = 14
    atr14      = _wilder_smooth(tr, p)
    plus_di14  = 100 * _wilder_smooth(plus_dm,  p) / atr14.replace(0, np.nan)
    minus_di14 = 100 * _wilder_smooth(minus_dm, p) / atr14.replace(0, np.nan)
    di_sum     = (plus_di14 + minus_di14).replace(0, np.nan)
    dx         = 100 * (plus_di14 - minus_di14).abs() / di_sum
    feat["ADX_14"]      = _wilder_smooth(dx, p)
    feat["Plus_DI_14"]  = plus_di14
    feat["Minus_DI_14"] = minus_di14

    # 15. Candlestick structure
    body    = (c - o).abs()
    candle  = (h - l).replace(0, np.nan)
    feat["Body_pct"]      = body / o
    feat["HL_range"]      = candle / o
    feat["Upper_shadow"]  = (h - pd.concat([c, o], axis=1).max(axis=1)) / candle
    feat["Lower_shadow"]  = (pd.concat([c, o], axis=1).min(axis=1) - l) / candle
    feat["Gap"]           = (o - c.shift(1)) / c.shift(1)

    # 16. Rolling return stats (mean, std, skew) for 3 periods
    ret = c.pct_change()
    for p in [5, 10, 20]:
        feat[f"Ret_mean_{p}"] = ret.rolling(p).mean()
        feat[f"Ret_std_{p}"]  = ret.rolling(p).std()
        feat[f"Ret_skew_{p}"] = ret.rolling(p).skew()

    # 17. Distance from N-day high/low
    for p in [20, 52]:
        feat[f"Pct_from_high_{p}"] = (c - h.rolling(p).max()) / h.rolling(p).max()
        feat[f"Pct_from_low_{p}"]  = (c - l.rolling(p).min()) / l.rolling(p).min()

    # 18. Parabolic SAR proxy (close vs 14-day EMA of midpoint)
    mid = (h + l) / 2
    feat["SAR_proxy"] = c / mid.ewm(span=14, adjust=False).mean() - 1

    # 19. Money Flow Index (MFI, 14-period)
    raw_mf  = tp * v
    pos_mf  = raw_mf.where(tp > tp.shift(1), 0)
    neg_mf  = raw_mf.where(tp < tp.shift(1), 0)
    mf_ratio = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    feat["MFI_14"] = 100 - 100 / (1 + mf_ratio)

    # 20. VWAP deviation (rolling 20-day)
    vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    feat["VWAP_dev"] = (c - vwap) / vwap

    # 21. Chaikin Money Flow (20-period)
    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    feat["CMF_20"] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # 22. Ease of Movement
    dm    = ((h + l) / 2) - ((h.shift(1) + l.shift(1)) / 2)
    br    = v / 1e6 / (h - l).replace(0, np.nan)
    feat["EOM_14"] = (dm / br.replace(0, np.nan)).rolling(14).mean()

    # 23. Force Index
    feat["Force_13"] = (c.diff() * v).ewm(span=13, adjust=False).mean()

    # 24. Keltner Channel
    kc_mid   = c.ewm(span=20, adjust=False).mean()
    kc_band  = atr14 * 2
    feat["KC_pct"]   = (c - (kc_mid - kc_band)) / (2 * kc_band).replace(0, np.nan)
    feat["KC_width"] = (2 * kc_band) / kc_mid

    # 25. Donchian Channel width (20-period)
    feat["DC_width_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c

    # 26. TEMA (Triple EMA, 20-period)
    ema1 = c.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    ema3 = ema2.ewm(span=20, adjust=False).mean()
    feat["TEMA_20"] = c / (3*ema1 - 3*ema2 + ema3) - 1

    # 27. DEMA (Double EMA, 20-period)
    feat["DEMA_20"] = c / (2*ema1 - ema2) - 1

    # 28. Hull MA proxy (WMA approximation using EMAs)
    half_ema = c.ewm(span=10, adjust=False).mean()
    full_ema = c.ewm(span=20, adjust=False).mean()
    hull_raw = 2*half_ema - full_ema
    feat["HMA_proxy"] = c / hull_raw.ewm(span=4, adjust=False).mean() - 1

    # 29. Price Percentage Oscillator (PPO)
    feat["PPO"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) \
                  / c.ewm(span=26, adjust=False).mean()

    # 30. Trix (15-period)
    t1 = c.ewm(span=15, adjust=False).mean()
    t2 = t1.ewm(span=15, adjust=False).mean()
    t3 = t2.ewm(span=15, adjust=False).mean()
    feat["TRIX_15"] = t3.pct_change()

    # 31. Ultimate Oscillator
    buying_pressure = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    true_range_uo   = pd.concat([h, c.shift(1)], axis=1).max(axis=1) \
                    - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    bp_7  = buying_pressure.rolling(7).sum()
    bp_14 = buying_pressure.rolling(14).sum()
    bp_28 = buying_pressure.rolling(28).sum()
    tr_7  = true_range_uo.rolling(7).sum()
    tr_14 = true_range_uo.rolling(14).sum()
    tr_28 = true_range_uo.rolling(28).sum()
    feat["UltOsc"] = 100 * (4 * bp_7/tr_7 + 2 * bp_14/tr_14 + bp_28/tr_28) / 7

    # 32. Mass Index (9/25-period)
    ema9_hl   = (h - l).ewm(span=9, adjust=False).mean()
    ema9_ema9 = ema9_hl.ewm(span=9, adjust=False).mean()
    feat["MassIndex_25"] = (ema9_hl / ema9_ema9.replace(0, np.nan)).rolling(25).sum()

    # 33. Vortex Indicator (14-period)
    vm_plus  = (h - l.shift(1)).abs().rolling(14).sum()
    vm_minus = (l - h.shift(1)).abs().rolling(14).sum()
    atr14_sum = tr.rolling(14).sum()
    feat["VI_plus_14"]  = vm_plus  / atr14_sum.replace(0, np.nan)
    feat["VI_minus_14"] = vm_minus / atr14_sum.replace(0, np.nan)

    # 34. Aroon Oscillator (2 periods)
    for p in [14, 25]:
        aroon_up   = h.rolling(p+1).apply(lambda x: x.argmax(), raw=True) / p * 100
        aroon_down = l.rolling(p+1).apply(lambda x: x.argmin(), raw=True) / p * 100
        feat[f"Aroon_osc_{p}"] = aroon_up - aroon_down

    # 35. Coppock Curve (10-month ROC + 11-month ROC, WMA 10)
    feat["Coppock"] = (c.pct_change(11) + c.pct_change(14)).rolling(10).mean()

    # 36. Elder Ray (14-period EMA basis)
    ema14 = c.ewm(span=14, adjust=False).mean()
    feat["Bull_power"] = h - ema14
    feat["Bear_power"] = l - ema14

    # 37. DPO (20-period)
    feat["DPO_20"] = c - c.rolling(11).mean().shift(11)

    # 38. Lagged close returns (5 lags)
    for lag in [1, 2, 3, 5, 10]:
        feat[f"Lag_ret_{lag}"] = c.pct_change(lag)

    return feat


# ── Label creation ────────────────────────────────────────────────────────────
def create_labels(df: pd.DataFrame,
                  future_days: int = FUTURE_DAYS,
                  threshold: float = BUY_THRESHOLD) -> pd.Series:
    future_ret = df["Close"].shift(-future_days) / df["Close"] - 1
    return (future_ret > threshold).astype(int).rename("label")


# ── Model ─────────────────────────────────────────────────────────────────────
def find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    """Return the threshold that maximises Buy F1 on the given set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls),
    )
    best = f1s.argmax()
    return float(thresholds[best]), float(f1s[best])


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, ticker: str = ""):
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:, 1]

    # ── Default threshold (0.5) ──
    y_pred_default = (y_proba >= 0.5).astype(int)

    # ── Optimal threshold (maximises Buy F1 on test set) ──
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, y_proba)
    y_pred_opt = (y_proba >= opt_thresh).astype(int)

    label = f" [{ticker}]" if ticker else ""
    print(f"\n{'='*60}")
    print(f"  Results{label}")
    print(f"{'='*60}")
    print(f"  Test samples    : {len(y_test)}")
    print(f"  ROC-AUC         : {roc_auc_score(y_test, y_proba):.4f}")

    print(f"\n  -- Default threshold (0.50) --")
    print(f"  Accuracy        : {accuracy_score(y_test, y_pred_default):.4f}")
    print(classification_report(y_test, y_pred_default,
                                target_names=["Sell/Hold", "Buy"], digits=3))

    print(f"  -- Optimal threshold ({opt_thresh:.2f}) — max Buy F1 --")
    print(f"  Accuracy        : {accuracy_score(y_test, y_pred_opt):.4f}")
    print(classification_report(y_test, y_pred_opt,
                                target_names=["Sell/Hold", "Buy"], digits=3))

    importance = (
        pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return rf, importance, opt_thresh


# ── Walk-forward cross-validation ────────────────────────────────────────────
def walk_forward_cv(X: pd.DataFrame, y: pd.Series, ticker: str = "",
                    n_folds: int = 5, min_train_frac: float = 0.5,
                    val_frac: float = 0.15) -> pd.DataFrame:
    """
    Expanding-window walk-forward CV.

    For each fold:
      - Train  : all data up to fold boundary (expanding)
      - Val    : last val_frac of the train window → used ONLY to pick threshold
      - Test   : next unseen chunk

    This avoids the leakage from our earlier approach where the optimal threshold
    was chosen on the same test set used for evaluation.
    """
    n = len(X)
    min_train  = int(n * min_train_frac)
    remaining  = n - min_train
    fold_size  = remaining // n_folds

    print(f"\n{'='*60}")
    print(f"  Walk-Forward CV [{ticker}]  —  {n_folds} folds")
    print(f"  Min train: {min_train} samples | Fold size: {fold_size} samples")
    print(f"{'='*60}")

    rows = []
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end  = train_end + fold_size
        if test_end > n:
            break

        # Split
        val_start  = int(train_end * (1 - val_frac))
        X_train    = X.iloc[:val_start]
        y_train    = y.iloc[:val_start]
        X_val      = X.iloc[val_start:train_end]
        y_val      = y.iloc[val_start:train_end]
        X_test     = X.iloc[train_end:test_end]
        y_test     = y.iloc[train_end:test_end]

        # Train
        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Find threshold on val set (no leakage into test)
        val_proba = rf.predict_proba(X_val)[:, 1]
        if len(y_val.unique()) > 1:
            opt_thresh, _ = find_optimal_threshold(y_val, val_proba)
        else:
            opt_thresh = 0.5   # fallback if val has only one class

        # Evaluate on test set
        test_proba    = rf.predict_proba(X_test)[:, 1]
        y_pred_def    = (test_proba >= 0.50).astype(int)
        y_pred_opt    = (test_proba >= opt_thresh).astype(int)

        roc = roc_auc_score(y_test, test_proba) if len(y_test.unique()) > 1 else np.nan
        acc_def   = accuracy_score(y_test, y_pred_def)
        acc_opt   = accuracy_score(y_test, y_pred_opt)
        buy_f1_def = f1_score(y_test, y_pred_def, pos_label=1, zero_division=0)
        buy_f1_opt = f1_score(y_test, y_pred_opt, pos_label=1, zero_division=0)

        date_start = X.index[train_end].date()
        date_end   = X.index[test_end - 1].date()

        print(f"\n  Fold {fold+1}  ({date_start} → {date_end})  "
              f"thresh={opt_thresh:.2f}")
        print(f"    ROC-AUC : {roc:.4f}  |  "
              f"Acc@0.50={acc_def:.3f}  BuyF1@0.50={buy_f1_def:.3f}  |  "
              f"Acc@opt={acc_opt:.3f}  BuyF1@opt={buy_f1_opt:.3f}")

        rows.append({
            "Fold"         : fold + 1,
            "Test_start"   : date_start,
            "Test_end"     : date_end,
            "Threshold"    : round(opt_thresh, 4),
            "ROC_AUC"      : round(roc, 4),
            "Acc_default"  : round(acc_def, 4),
            "BuyF1_default": round(buy_f1_def, 4),
            "Acc_opt"      : round(acc_opt, 4),
            "BuyF1_opt"    : round(buy_f1_opt, 4),
        })

    results = pd.DataFrame(rows)

    print(f"\n  Mean across folds:")
    for col in ["ROC_AUC", "BuyF1_default", "BuyF1_opt"]:
        print(f"    {col:20s}: {results[col].mean():.4f}  ±  {results[col].std():.4f}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_importance(importance: pd.DataFrame, ticker: str, top_n: int = 30):
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {ticker}")
    plt.tight_layout()
    fname = f"feature_importance_{ticker}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Chart saved → {fname}")


def plot_cv_results(cv_results: dict[str, pd.DataFrame]):
    """Line chart of fold-by-fold ROC-AUC and BuyF1 for each ticker."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ticker, df in cv_results.items():
        axes[0].plot(df["Fold"], df["ROC_AUC"],    marker="o", label=ticker)
        axes[1].plot(df["Fold"], df["BuyF1_opt"],  marker="o", label=ticker)
    for ax, title in zip(axes, ["ROC-AUC per fold", "Buy F1 (opt threshold) per fold"]):
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Fold")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig("walk_forward_cv.png", dpi=150)
    plt.close()
    print("Walk-forward CV chart saved → walk_forward_cv.png")


def plot_summary(summary: list[dict]):
    """Bar chart comparing ROC-AUC and optimal Buy F1 across tickers."""
    df = pd.DataFrame(summary)
    x  = np.arange(len(df))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, df["CV_ROC_AUC"],   w, label="ROC-AUC")
    ax.bar(x + w/2, df["CV_BuyF1_opt"], w, label="Opt Buy F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Ticker"])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.legend()
    ax.set_title("Model comparison across tickers")
    plt.tight_layout()
    plt.savefig("ticker_comparison.png", dpi=150)
    plt.close()
    print("\nComparison chart saved → ticker_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary    = []
    cv_results = {}

    for ticker in TICKERS:
        print(f"\nDownloading {ticker} ({PERIOD}) ...")
        df = download_data(ticker, PERIOD)
        print(f"  {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")

        features = compute_features(df)
        labels   = create_labels(df)

        data = features.join(labels).dropna()
        X    = data.drop(columns=["label"])
        y    = data["label"]

        print(f"  {X.shape[1]} features | {len(X)} samples | "
              f"Buy {y.mean():.1%} / Sell {(1-y.mean()):.1%}")

        # ── Walk-forward CV (primary evaluation) ──
        cv_df = walk_forward_cv(X, y, ticker)
        cv_results[ticker] = cv_df
        cv_df.to_csv(f"cv_results_{ticker}.csv", index=False)

        # ── Final model on full data (for feature importance + predict.py) ──
        model, importance, _ = train_and_evaluate(X, y, ticker)
        importance.to_csv(f"feature_importances_{ticker}.csv", index=False)
        plot_importance(importance, ticker)

        # Save model and feature list so predict.py can load them
        joblib.dump(model, f"model_{ticker}.joblib")
        pd.Series(X.columns.tolist()).to_csv(f"features_{ticker}.csv", index=False, header=False)
        print(f"  Model saved → model_{ticker}.joblib")

        cv_threshold = round(cv_df["Threshold"].mean(), 4)
        summary.append({
            "Ticker"          : ticker,
            "CV_ROC_AUC"      : round(cv_df["ROC_AUC"].mean(), 4),
            "CV_BuyF1_default": round(cv_df["BuyF1_default"].mean(), 4),
            "CV_BuyF1_opt"    : round(cv_df["BuyF1_opt"].mean(), 4),
            "CV_Threshold"    : cv_threshold,
        })

    # ── Cross-ticker summary ──
    print(f"\n{'='*60}")
    print("  Walk-Forward CV Summary (mean across folds)")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("ticker_summary.csv", index=False)
    print("\nSummary saved → ticker_summary.csv")

    plot_cv_results(cv_results)
    plot_summary(summary)
