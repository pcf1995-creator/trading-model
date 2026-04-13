# Trading Model — CLAUDE.md

## Project Overview

A machine-learning-based trading system with two main components:

1. **Stock signal generator** — scikit-learn models predict 5-day price direction for ~40 equities/ETFs; signals are acted on via MOC orders.
2. **Kalshi event trading** — crypto price prediction contracts (BTC/ETH) traded on [Kalshi](https://kalshi.com) using intraday and daily models.

A Streamlit dashboard (`app.py`) surfaces both components with tabs for live positions, long-term signals, and performance.

---

## Architecture

```
app.py                  Streamlit dashboard (main entry point)
predict.py              Daily stock signal generator (CLI)
features.py             Technical indicator computation (shared)
db.py                   Persistence layer (Supabase or local JSON fallback)
kalshi_api.py           Kalshi REST API v2 client (RSA-PSS auth)
kalshi_crypto.py        Crypto Kalshi model + position logic (root-level)
upload_models.py        One-time upload of stock model files to Supabase Storage

kalshi/                 Kalshi-specific scripts and data
  kalshi_crypto.py      Crypto prediction logic
  kalshi_api.py         (symlinked or copy)
  analyze.py            Position analysis
  monitor.py            Live monitoring loop
  longterm.py           Longer-horizon Kalshi signals
  predict.py            Kalshi-specific prediction entry point
  positions_kalshi.json Local fallback for Kalshi positions
  predictions_log.jsonl Append-only log of past predictions
  features_*.csv        Per-ticker feature name lists

model_*.joblib          Trained scikit-learn models (one per stock ticker)
features_*.csv          Feature name lists per ticker (used by predict.py)
model_crypto*.joblib    Crypto models (daily + intraday)
model_crypto*_meta.json Crypto model metadata (thresholds, calibration)
ticker_summary.csv      Ticker list with CV ROC-AUC and CV threshold per ticker
positions.json          Open/closed stock positions (local fallback)
```

---

## Common Commands

### Run the dashboard
```bash
streamlit run app.py
```

### Run daily stock signals (~30 min before market close)
```bash
python predict.py                    # use tickers from ticker_summary.csv
python predict.py --portfolio 50000  # set portfolio size
python predict.py --dry-run          # preview signals without saving
```

### Upload stock model files to Supabase Storage
```bash
SUPABASE_URL=... SUPABASE_KEY=... python3 upload_models.py
```

---

## Environment Variables / Secrets

The app reads credentials from Streamlit secrets (`st.secrets`) first, then falls back to environment variables.

| Variable | Description |
|---|---|
| `KALSHI_KEY_ID` | Kalshi API key ID |
| `KALSHI_KEY_PATH` | Path to Kalshi RSA private key `.pem` file |
| `KALSHI_KEY_CONTENT` | Full PEM string (alternative to `KEY_PATH`, used on Streamlit Cloud) |
| `KALSHI_DEMO` | Set to `"true"` to use the Kalshi demo environment |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role or anon key |

**Never commit `.pem` files or secrets to the repository.**

---

## Trading Logic

### Stock model (`predict.py`)
- **Entry**: Buy at close (MOC order) when `model.predict_proba() >= CV_Threshold` AND `prob >= MIN_PROB` (0.60)
- **Exit**: Sell at close after `HOLD_DAYS` (5 trading days) OR if P&L ≤ `-STOP_LOSS_PCT` (−2%)
- **Sizing**: Equal weight across `MAX_POSITIONS` (8) slots; position size = `portfolio / MAX_POSITIONS`
- **Quality filter**: Only trade tickers with `CV_ROC_AUC >= MIN_ROC_AUC` (0.55) in `ticker_summary.csv`
- **Intraday awareness**: Before 3:30 PM ET, signals off yesterday's confirmed close; after 3:30 PM ET, uses today's mature bar

### Kalshi crypto model
- Predicts BTC/ETH price direction for Kalshi event contracts
- Separate intraday and daily models (`model_crypto_intraday.joblib` / `model_crypto.joblib`)
- Position data stored in Supabase (`position_overrides` table) with local JSON fallback

---

## Persistence Layer (`db.py`)

All data is stored in **Supabase** when credentials are present, with automatic fallback to local JSON files for local development.

| Supabase table / bucket | Local fallback |
|---|---|
| `paper_trades` | `paper_trades.json` |
| `stock_paper_trades` | `stock_paper_trades.json` |
| `position_overrides` | `kalshi/positions_kalshi.json` |
| `calibration` | `model_crypto_calibration.json` |
| Storage bucket `stock-models` | `model_*.joblib`, `features_*.csv` in repo root |

Models are downloaded from Supabase Storage on demand and cached to `/tmp/trading_models/`.

---

## Feature Engineering

`features.py` (and the mirrored `compute_features()` in `predict.py`) builds ~150 technical indicators:

- Moving averages: SMA/EMA (5, 10, 20, 50, 100, 200), crossovers, TEMA, DEMA, HMA proxy
- Momentum: RSI (7/14/21), MACD, Stochastic, Williams %R, CCI, ROC, PPO, TRIX, Ultimate Oscillator
- Volatility: ATR (7/14/21), Bollinger Bands, Keltner Channels, Donchian Channels, ADX
- Volume: OBV, MFI, CMF, EOM, Force Index, VWAP deviation
- Candlestick: body %, HL range, upper/lower shadows, gap
- Statistical: rolling return mean/std/skew, pct from rolling high/low, lagged returns

**Important**: `predict.py` contains a verbatim copy of `compute_features()` to be self-contained. If you modify the feature logic in `features.py`, apply the same change to `predict.py`.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Dashboard UI |
| `scikit-learn` + `joblib` | ML models, serialization |
| `yfinance` | Historical price data |
| `supabase` | Cloud database and file storage |
| `requests` + `cryptography` | Kalshi API (RSA-PSS signing) |
| `pandas` + `numpy` | Data processing |
| `matplotlib` | Charts in dashboard |

Install: `pip install -r requirements.txt`
