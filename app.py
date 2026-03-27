"""
app.py — Trading Dashboard
Run: streamlit run app.py
"""

import importlib.util
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "kalshi"))
sys.path.insert(0, str(ROOT))

from kalshi_api import KalshiClient, KALSHI_CONFIG  # noqa: E402


def make_kalshi_client() -> KalshiClient:
    """Build KalshiClient from st.secrets (Streamlit Cloud) or env vars (local)."""
    try:
        key_id      = st.secrets.get("KALSHI_KEY_ID")
        key_content = st.secrets.get("KALSHI_KEY_CONTENT")  # full PEM string
        key_path    = st.secrets.get("KALSHI_KEY_PATH")     # fallback path
        if key_id and (key_content or key_path):
            cfg = {**KALSHI_CONFIG, "key_id": key_id,
                   "key_content": key_content, "key_path": key_path}
            return KalshiClient(config=cfg)
    except Exception:
        pass
    return KalshiClient()  # falls back to env vars

POSITIONS_KALSHI = ROOT / "kalshi" / "positions_kalshi.json"
POSITIONS_STOCKS = ROOT / "positions.json"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Trading Dashboard")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def hours_left(close_time_str: str) -> float | None:
    if not close_time_str:
        return None
    try:
        dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).total_seconds() / 3600
    except Exception:
        return None


def parse_ticker(ticker: str) -> tuple[str, str, str]:
    parts  = ticker.split("-")
    asset  = "BTC" if "BTC" in parts[0].upper() else "ETH"
    expiry = parts[1] if len(parts) > 1 else ""
    strike = parts[2].lstrip("T") if len(parts) > 2 else ""
    return asset, expiry, strike


def color_pnl(val: str) -> str:
    if isinstance(val, str) and val.startswith("+"):
        return "color: #2ecc71; font-weight: bold"
    if isinstance(val, str) and val.startswith("-"):
        return "color: #e74c3c; font-weight: bold"
    return ""


def get_bid_cents(market: dict) -> int | None:
    bid_fp      = market.get("yes_bid_fp")
    bid_dollars = market.get("yes_bid_dollars")
    bid         = market.get("yes_bid")
    if bid_dollars is not None:
        return round(float(bid_dollars) * 100)
    if bid_fp is not None:
        return round(bid_fp / 100)
    if bid is not None:
        return int(bid)
    return None


# ── Cached API calls ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_prices(tickers: tuple) -> dict[str, int | None]:
    """Live yes_bid for each ticker. Auto-refreshes every 60s."""
    client = make_kalshi_client()
    if client.dry_run:
        return {}
    prices = {}
    for ticker in tickers:
        try:
            prices[ticker] = get_bid_cents(client.get_market(ticker))
        except Exception:
            prices[ticker] = None
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_settlements(tickers: tuple) -> dict[str, str | None]:
    """Actual settlement result ('yes'/'no'/None) for expired contracts. Cached 1h."""
    client = make_kalshi_client()
    if client.dry_run:
        return {}
    results = {}
    for ticker in tickers:
        try:
            market = client._request("GET", f"/markets/{ticker}").get("market", {})
            results[ticker] = market.get("result")
        except Exception:
            results[ticker] = None
    return results


# ══════════════════════════════════════════════════════════════════════════════
# KALSHI — OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
st.header("Kalshi — Open Positions")

c1, c2 = st.columns([1, 8])
with c1:
    if st.button("↻ Refresh", type="primary"):
        st.cache_data.clear()
        st.rerun()

# ── Load positions: API is source of truth for open positions ─────────────────
_client      = make_kalshi_client()
all_kalshi   = load_json(POSITIONS_KALSHI)
# Local JSON keyed by ticker — used for entry/stop prices and closed positions
_local_by_ticker = {p["ticker"]: p for p in all_kalshi}

if not _client.dry_run:
    try:
        _api_positions = _client.get_positions()
        open_kalshi = []
        for _pos in _api_positions:
            _tkr = _pos.get("ticker", "")
            if not _tkr:
                continue
            _local = _local_by_ticker.get(_tkr, {})
            _mkt   = _client.get_market(_tkr)
            open_kalshi.append({
                "ticker"      : _tkr,
                "status"      : "open",
                "side"        : "yes" if _pos.get("position", 0) > 0 else "no",
                "contracts"   : abs(_pos.get("position", 1)),
                "entry_cents" : _local.get("entry_cents", 0),
                "stop_cents"  : _local.get("stop_cents", 0),
                "close_time"  : _mkt.get("close_time", _local.get("close_time", "")),
            })
    except Exception as e:
        st.warning(f"Could not fetch live positions from Kalshi: {e}")
        open_kalshi = [p for p in all_kalshi if p["status"] == "open"]
else:
    open_kalshi = [p for p in all_kalshi if p["status"] == "open"]

if open_kalshi:
    open_tickers = tuple(p["ticker"] for p in open_kalshi)
    with st.spinner("Loading live prices..."):
        live = fetch_live_prices(open_tickers)

    rows = []
    for p in open_kalshi:
        ticker    = p["ticker"]
        entry     = p["entry_cents"]
        stop      = p["stop_cents"]
        contracts = p["contracts"]
        hrs       = hours_left(p.get("close_time", ""))
        asset, expiry, strike = parse_ticker(ticker)
        current   = live.get(ticker)
        pnl_pct   = ((current - entry) / entry * 100
                     if current is not None and entry > 0 else None)

        rows.append({
            "Ticker"   : ticker,
            "Asset"    : asset,
            "Strike"   : f"${float(strike):,.0f}" if strike else "",
            "Hrs Left" : (f"{int(hrs * 60)}m" if hrs is not None and hrs < 1
                          else f"{hrs:.0f}h" if hrs is not None else "—"),
            "Contracts": contracts,
            "Entry ¢"  : entry,
            "Stop ¢"   : stop,
            "Live Bid" : f"{current}¢" if current is not None else "—",
            "P&L"      : (f"+{pnl_pct:.1f}%" if pnl_pct is not None and pnl_pct >= 0
                          else f"{pnl_pct:.1f}%" if pnl_pct is not None else "—"),
        })

    df_open = pd.DataFrame(rows).reset_index(drop=True)

    edited = st.data_editor(
        df_open.drop(columns=["Ticker"]),
        column_config={
            "Entry ¢": st.column_config.NumberColumn("Entry ¢", min_value=0, max_value=99, step=1),
            "Stop ¢" : st.column_config.NumberColumn("Stop ¢",  min_value=0, max_value=99, step=1),
        },
        disabled=["Asset", "Strike", "Hrs Left", "Contracts", "Live Bid", "P&L"],
        hide_index=True,
        use_container_width=True,
    )

    if st.button("💾 Save entry & stop prices"):
        import json as _json
        for i, row in edited.iterrows():
            tkr = df_open.iloc[i]["Ticker"]
            _local_by_ticker.setdefault(tkr, {"ticker": tkr, "status": "open"})
            _local_by_ticker[tkr]["entry_cents"] = int(row["Entry ¢"])
            _local_by_ticker[tkr]["stop_cents"]  = int(row["Stop ¢"])
        with open(POSITIONS_KALSHI, "w") as _f:
            _json.dump(list(_local_by_ticker.values()), _f, indent=2)
        st.success("Saved!")

    m1, m2, m3 = st.columns(3)
    m1.metric("Open Positions", len(open_kalshi))
    expiring_soon = sum(1 for p in open_kalshi
                        if (h := hours_left(p.get("close_time", ""))) is not None and h < 12)
    m2.metric("Expiring < 12h", expiring_soon)
    total_exp = sum(p["entry_cents"] * p["contracts"] for p in open_kalshi)
    m3.metric("Total Exposure", f"${total_exp/100:.2f}")
else:
    st.info("No open Kalshi positions.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# KALSHI — CLOSED POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Filter to positions with a real entry (skip zero-cost stubs)
closed_kalshi = [p for p in all_kalshi
                 if p["status"] not in ("open",) and p.get("entry_cents", 0) > 0]

if closed_kalshi:
    with st.expander(f"Closed / Settled Positions ({len(closed_kalshi)})"):
        # Fetch actual settlement for contracts held to expiry
        need_settlement = tuple(
            p["ticker"] for p in closed_kalshi
            if p["status"] in ("settled", "stop_triggered")
        )
        settlement_map = fetch_settlements(need_settlement) if need_settlement else {}

        rows = []
        for p in closed_kalshi:
            asset, expiry, strike = parse_ticker(p["ticker"])
            entry  = p.get("entry_cents", 0)
            ctrs   = p["contracts"]
            status = p["status"]

            if status == "closed":
                # Manually sold
                exit_c = p.get("exit_cents", 0)
                pnl    = (exit_c - entry) * ctrs / 100
                exit_label = f"{exit_c}¢ (sold)"
            else:
                # Held to expiry — use actual settlement result
                result = settlement_map.get(p["ticker"])
                if result == "yes":
                    pnl        = (100 - entry) * ctrs / 100
                    exit_label = "100¢ (won ✓)"
                elif result == "no":
                    pnl        = -entry * ctrs / 100
                    exit_label = "0¢ (expired)"
                else:
                    pnl        = None
                    exit_label = "—"

            rows.append({
                "Asset"    : asset,
                "Strike"   : f"${float(strike):,.0f}" if strike else "",
                "Expiry"   : expiry,
                "Contracts": ctrs,
                "Entry"    : f"{entry}¢",
                "Exit"     : exit_label,
                "P&L $"    : (f"${pnl:+.2f}" if pnl is not None else "—"),
            })

        df_closed = pd.DataFrame(rows)
        st.dataframe(df_closed.style.map(color_pnl, subset=["P&L $"]),
                     width="stretch", hide_index=True)

        resolved = [r for r in rows if r["P&L $"] != "—"]
        if resolved:
            total_pnl = sum(
                float(r["P&L $"].replace("$", "").replace("+", ""))
                for r in resolved
            )
            wins = sum(1 for r in resolved
                       if isinstance(r["P&L $"], str) and r["P&L $"].startswith("$+"))
            c1, c2 = st.columns(2)
            c1.metric("Win Rate", f"{wins/len(resolved):.0%}")
            c2.metric("Total P&L", f"${total_pnl:+.2f}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# KALSHI — SCAN OPPORTUNITIES
# ══════════════════════════════════════════════════════════════════════════════
st.header("Kalshi — Scan Opportunities")
st.caption("Top 5 by EV across both YES and NO sides, split by time to expiry.")

if st.button("Run Kalshi Scan", type="primary", key="scan_kalshi"):
    try:
        from kalshi_crypto import (
            load_crypto_models, score_contract,
            download_crypto, download_crypto_hourly,
            CRYPTO_ASSETS, KALSHI_SERIES, INFERENCE_PERIOD, MIN_EV, MIN_EDGE,
        )

        with st.spinner("Loading model..."):
            models = load_crypto_models()

        if models.get("daily") is None:
            st.error("No trained model. Run `python kalshi_crypto.py --train` first.")
        else:
            has_intraday = models.get("intraday") is not None
            client = make_kalshi_client()
            if client.dry_run:
                st.warning("Running in dry-run mode — set KALSHI_KEY_ID and KALSHI_KEY_PATH "
                           "to scan live contracts.")

            asset_dfs_by_symbol = {}
            with st.spinner("Fetching crypto prices..."):
                for symbol in CRYPTO_ASSETS:
                    daily_df  = download_crypto(symbol, INFERENCE_PERIOD)
                    hourly_df = download_crypto_hourly(symbol) if has_intraday else None
                    asset_dfs_by_symbol[symbol] = {"daily": daily_df, "hourly": hourly_df}

            all_results = []
            scan_debug  = []
            with st.spinner("Scoring contracts..."):
                for symbol, series in KALSHI_SERIES.items():
                    markets   = client.get_markets(series_ticker=series, status="open")
                    scored    = 0
                    asset_dfs = asset_dfs_by_symbol[symbol]
                    for market in markets:
                        results = score_contract(market, models, asset_dfs)
                        all_results.extend(results)
                        if results:
                            scored += 1
                    scan_debug.append(f"{series}: {len(markets)} markets, {scored} scored")
            st.caption(" · ".join(scan_debug)
                       + f" · {sum(1 for r in all_results if r['hours_to_expiry'] > 24)} sides >24h")

            DAILY_BUDGET  = 50.0
            WEEKLY_BUDGET = 200.0

            def build_bucket(results: list, budget: float) -> list:
                """
                Pick best signal per (asset, expiry) group by EV.
                Add correlated second pick at half Kelly weight if EV >= 80% of best in group.
                Normalize Kelly weights so total spend = budget.
                """
                valid = [r for r in results if r["ev"] >= MIN_EV and r["edge"] >= MIN_EDGE]
                valid.sort(key=lambda x: x["ev"], reverse=True)

                groups: dict[tuple, list] = {}
                for r in valid:
                    groups.setdefault((r["asset"], r["expiry"]), []).append(r)

                picks = []  # (result, kelly_weight)
                for group in sorted(groups.values(), key=lambda g: g[0]["ev"], reverse=True):
                    if len(picks) >= 4:
                        break
                    best = group[0]
                    picks.append((best, best["kelly_pct"]))

                if not picks:
                    return []

                total_weight = sum(w for _, w in picks)
                portfolio = []
                for r, weight in picks:
                    dollars   = round(budget * (weight / total_weight), 2)
                    contracts = max(1, int(dollars / (r["price"] / 100)))
                    portfolio.append({
                        **r,
                        "kelly_dollars"      : dollars,
                        "contracts_suggested": contracts,
                        "correlated"         : weight < r["kelly_pct"],
                    })
                return portfolio

            def make_portfolio_table(portfolio: list) -> pd.DataFrame:
                rows = []
                for p in portfolio:
                    rows.append({
                        "Ticker"   : p["ticker"],
                        "Side"     : p["side"],
                        "Hrs Left" : (f"{int(p['hours_to_expiry'] * 60)}m"
                                      if p['hours_to_expiry'] < 1
                                      else f"{p['hours_to_expiry']:.0f}h"),
                        "Price"    : f"{p['price']}¢",
                        "Cal Prob" : f"{p['calibrated_prob']*100:.1f}%",
                        "EV"       : f"{p['ev']:+.3f}",
                        "Bet $"    : f"${p['kelly_dollars']:.0f}",
                        "Contracts": p["contracts_suggested"],
                        "Model"    : p.get("model_type", "daily"),
                    })
                return pd.DataFrame(rows)

            def make_scan_table(results: list) -> pd.DataFrame:
                rows = []
                for r in results:
                    rows.append({
                        "Asset"    : r["asset"],
                        "Side"     : r["side"],
                        "Strike"   : f"${r['strike']:,.0f}",
                        "Expiry"   : r["expiry"],
                        "Price"    : f"{r['price']}¢",
                        "Cal Prob" : f"{r['calibrated_prob']*100:.1f}%",
                        "Edge"     : f"{r['edge']*100:+.1f}pp",
                        "EV"       : f"{r['ev']:+.3f}",
                        "Kelly %"  : f"{r['kelly_pct']:.1f}%",
                    })
                return pd.DataFrame(rows)

            under24 = [r for r in all_results if r["hours_to_expiry"] <= 24]
            over24  = [r for r in all_results if r["hours_to_expiry"] > 24]

            daily_port  = build_bucket(under24, DAILY_BUDGET)
            weekly_port = build_bucket(over24,  WEEKLY_BUDGET)

            # ── Daily plays ──
            st.subheader(f"Daily Plays — ${DAILY_BUDGET:.0f} budget (≤24h)")
            if daily_port:
                st.dataframe(make_portfolio_table(daily_port), width="stretch", hide_index=True)
            else:
                st.info("No daily contracts meet the thresholds right now.")

            # ── Weekly plays ──
            st.subheader(f"Weekly Plays — ${WEEKLY_BUDGET:.0f} budget (>24h)")
            if weekly_port:
                st.dataframe(make_portfolio_table(weekly_port), width="stretch", hide_index=True)
            else:
                st.info("No weekly contracts available right now.")

            # ── Full scan results ──
            with st.expander("All scored contracts"):
                if under24:
                    st.markdown("**≤ 24h to expiry**")
                    st.dataframe(make_scan_table(
                        sorted(under24, key=lambda x: x["ev"], reverse=True)[:10]),
                        width="stretch", hide_index=True)
                if over24:
                    st.markdown("**> 24h to expiry**")
                    st.dataframe(make_scan_table(
                        sorted(over24, key=lambda x: x["ev"], reverse=True)[:10]),
                        width="stretch", hide_index=True)

            st.caption(f"Scanned {len(all_results)//2} contracts · {len(all_results)} sides scored")

    except Exception as e:
        st.error(f"Scan error: {e}")
        import traceback; st.code(traceback.format_exc())

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STOCKS — OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
st.header("Stocks — Open Positions")

stock_positions = load_json(POSITIONS_STOCKS)
open_stocks     = [p for p in stock_positions if p["status"] == "open"]
closed_stocks   = [p for p in stock_positions if p["status"] != "open"]

if open_stocks:
    stock_rows = []
    for p in open_stocks:
        pnl = p.get("pnl_pct", 0)
        stock_rows.append({
            "Ticker"    : p["ticker"],
            "Entry Date": p["entry_date"],
            "Entry $"   : f"${p['entry_price']:.2f}",
            "Current $" : f"${p.get('current_price', p['entry_price']):.2f}",
            "Shares"    : p["shares"],
            "Cost"      : f"${p['cost']:.2f}",
            "Days Held" : f"{p.get('days_held', '—')}/5",
            "P&L"       : f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%",
        })
    st.dataframe(
        pd.DataFrame(stock_rows).style.map(color_pnl, subset=["P&L"]),
        width="stretch", hide_index=True,
    )
else:
    st.info("No open stock positions.")

if closed_stocks:
    with st.expander(f"Closed Stock Positions ({len(closed_stocks)})"):
        cl_rows = []
        for p in closed_stocks:
            pnl = p.get("pnl_pct", 0)
            cl_rows.append({
                "Ticker"    : p["ticker"],
                "Entry Date": p["entry_date"],
                "Exit Date" : p.get("exit_date", "—"),
                "Entry $"   : f"${p['entry_price']:.2f}",
                "Exit $"    : f"${p.get('exit_price', 0):.2f}",
                "P&L"       : f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%",
                "Reason"    : p["status"].replace("closed_", "").upper(),
            })
        df_cl = pd.DataFrame(cl_rows)
        st.dataframe(df_cl.style.map(color_pnl, subset=["P&L"]),
                     width="stretch", hide_index=True)
        wins    = sum(1 for p in closed_stocks if p.get("pnl_pct", 0) > 0)
        avg_pnl = sum(p.get("pnl_pct", 0) for p in closed_stocks) / len(closed_stocks)
        c1, c2  = st.columns(2)
        c1.metric("Win Rate", f"{wins/len(closed_stocks):.0%}")
        c2.metric("Avg P&L",  f"{avg_pnl:+.2f}%")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STOCKS — SCAN
# ══════════════════════════════════════════════════════════════════════════════
st.header("Stocks — Daily Scan")
st.caption("Top 5 buy signals from the Random Forest model.")

if st.button("Run Stock Scan", type="primary", key="scan_stocks"):
    try:
        import joblib
        import numpy as np
        import yfinance as yf

        # Load predict module from root (where models live)
        spec = importlib.util.spec_from_file_location(
            "stock_predict", ROOT / "predict.py"
        )
        if not (ROOT / "predict.py").exists():
            spec = importlib.util.spec_from_file_location(
                "stock_predict", ROOT / "stocks" / "predict.py"
            )
        pm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm)

        summary_path = ROOT / "ticker_summary.csv"
        if not summary_path.exists():
            summary_path = ROOT / "stocks" / "ticker_summary.csv"

        if not summary_path.exists():
            st.error("No ticker_summary.csv found. Run features.py first.")
        else:
            summary   = pd.read_csv(summary_path)
            thresh_map = dict(zip(summary["Ticker"], summary["CV_Threshold"]))
            roc_map    = dict(zip(summary["Ticker"], summary["CV_ROC_AUC"]))
            eligible   = [t for t in summary["Ticker"]
                          if roc_map.get(t, 0) >= pm.MIN_ROC_AUC]

            signals = []
            prog = st.progress(0, text="Scanning...")
            for i, ticker in enumerate(eligible):
                prog.progress((i + 1) / len(eligible), text=f"Scanning {ticker}...")
                model_path    = ROOT / f"model_{ticker}.joblib"
                features_path = ROOT / f"features_{ticker}.csv"
                if not model_path.exists():
                    continue
                model         = joblib.load(model_path)
                feature_names = pd.read_csv(features_path, header=None)[0].tolist()
                threshold     = thresh_map.get(ticker, 0.30)
                result        = pm.get_latest_signal(ticker, model, feature_names, threshold)
                if result:
                    signals.append(result)
            prog.empty()

            buy_signals = sorted(
                [s for s in signals if s["signal"] and s["prob"] >= pm.MIN_PROB],
                key=lambda x: x["prob"], reverse=True,
            )[:5]

            if not buy_signals:
                st.info("No buy signals today.")
            else:
                scan_rows = [
                    {
                        "Ticker"   : s["ticker"],
                        "Close $"  : f"${s['close']:.2f}",
                        "Prob"     : f"{s['prob']*100:.1f}%",
                        "Threshold": f"{s['threshold']*100:.0f}%",
                        "Signal"   : "✓ BUY",
                    }
                    for s in buy_signals
                ]
                st.dataframe(pd.DataFrame(scan_rows), width="stretch", hide_index=True)
                st.caption(f"Scanned {len(eligible)} eligible tickers "
                           f"({len(signals)} returned signals)")

    except Exception as e:
        st.error(f"Stock scan error: {e}")
        import traceback; st.code(traceback.format_exc())
