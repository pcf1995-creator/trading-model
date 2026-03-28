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
import db


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
PAPER_TRADES     = ROOT / "paper_trades.json"

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
_client          = make_kalshi_client()
_local_by_ticker = db.load_position_overrides()   # {ticker: {entry_cents, stop_cents, contracts}}

if not _client.dry_run:
    try:
        _api_positions = _client.get_positions()
        _all_api = []
        for _pos in _api_positions:
            _tkr = _pos.get("ticker", "")
            # Only show crypto positions in this dashboard
            if not _tkr or not (_tkr.startswith("KXBTC") or _tkr.startswith("KXETH")):
                continue
            _local = _local_by_ticker.get(_tkr, {})
            _mkt   = _client.get_market(_tkr)
            _hrs   = hours_left(_mkt.get("close_time", ""))
            _all_api.append({
                "ticker"      : _tkr,
                "status"      : "open",
                "side"        : "yes" if _pos.get("position", 0) > 0 else "no",
                # Prefer manually saved contracts; fall back to API position count
                "contracts"   : _local.get("contracts", abs(_pos.get("position", 1))),
                "entry_cents" : _local.get("entry_cents", 0),
                "stop_cents"  : _local.get("stop_cents", 0),
                "close_time"  : _mkt.get("close_time", _local.get("close_time", "")),
                "_hrs"        : _hrs,
            })
        # Separate truly open from expired-awaiting-settlement
        open_kalshi    = [p for p in _all_api if p["_hrs"] is None or p["_hrs"] >= 0]
        settling_kalshi = [p for p in _all_api if p["_hrs"] is not None and p["_hrs"] < 0]
    except Exception as e:
        st.warning(f"Could not fetch live positions from Kalshi: {e}")
        open_kalshi     = []
        settling_kalshi = []
else:
    open_kalshi     = []
    settling_kalshi = []

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

        bet_dollars = entry * contracts / 100
        rows.append({
            "Ticker"   : ticker,
            "Asset"    : asset,
            "Strike"   : (f"${float(strike):,.0f}" if strike and strike.replace(".", "").isdigit() else strike),
            "Hrs Left" : (f"{int(hrs * 60)}m" if hrs is not None and hrs < 1
                          else f"{hrs:.0f}h" if hrs is not None else "—"),
            "Contracts": contracts,
            "Entry ¢"  : entry,
            "Bet $"    : f"${bet_dollars:.2f}",
            "Stop ¢"   : stop,
            "Live Bid" : f"{current}¢" if current is not None else "—",
            "P&L"      : (f"+{pnl_pct:.1f}%" if pnl_pct is not None and pnl_pct >= 0
                          else f"{pnl_pct:.1f}%" if pnl_pct is not None else "—"),
        })

    df_open = pd.DataFrame(rows).reset_index(drop=True)

    edited = st.data_editor(
        df_open.drop(columns=["Ticker"]),
        column_config={
            "Contracts": st.column_config.NumberColumn("Contracts", min_value=1, step=1),
            "Entry ¢"  : st.column_config.NumberColumn("Entry ¢", min_value=0, max_value=99, step=1),
            "Stop ¢"   : st.column_config.NumberColumn("Stop ¢",  min_value=0, max_value=99, step=1),
        },
        disabled=["Asset", "Strike", "Hrs Left", "Bet $", "Live Bid", "P&L"],
        hide_index=True,
        use_container_width=True,
    )

    if st.button("💾 Save contracts, entry & stop"):
        for i, row in edited.iterrows():
            tkr = df_open.iloc[i]["Ticker"]
            _local_by_ticker.setdefault(tkr, {"ticker": tkr})
            _local_by_ticker[tkr]["contracts"]   = int(row["Contracts"])
            _local_by_ticker[tkr]["entry_cents"] = int(row["Entry ¢"])
            _local_by_ticker[tkr]["stop_cents"]  = int(row["Stop ¢"])
        db.save_position_overrides(_local_by_ticker)
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

# Awaiting settlement (expired but not yet settled by Kalshi)
if settling_kalshi:
    with st.expander(f"⏳ Awaiting Settlement ({len(settling_kalshi)})"):
        st.caption("These contracts have expired but Kalshi hasn't settled them yet.")
        settle_rows = []
        for p in settling_kalshi:
            asset, expiry, strike = parse_ticker(p["ticker"])
            settle_rows.append({
                "Asset"    : asset,
                "Strike"   : (f"${float(strike):,.0f}" if strike and strike.replace(".", "").isdigit() else strike),
                "Expiry"   : expiry,
                "Contracts": p["contracts"],
                "Entry"    : f"{p['entry_cents']}¢",
            })
        st.dataframe(pd.DataFrame(settle_rows), hide_index=True, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# KALSHI — CLOSED POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
from collections import defaultdict

_fills      = []
_fills_err  = None
_raw_sample = None
if not _client.dry_run:
    try:
        _fills      = _client.get_fills(limit=2000)
        _raw_sample = _fills[0] if _fills else None
    except Exception as e:
        _fills_err = str(e)

# Group fills by ticker → reconstruct closed positions
_open_tickers = {p["ticker"] for p in open_kalshi} | {p["ticker"] for p in settling_kalshi}
_by_ticker: dict[str, list] = defaultdict(list)
for _f in _fills:
    _tkr = _f.get("market_ticker") or _f.get("ticker", "")
    if (_tkr
            and _tkr not in _open_tickers
            and (_tkr.startswith("KXBTC") or _tkr.startswith("KXETH"))):
        _by_ticker[_tkr].append(_f)

def _fill_count(_f: dict) -> float:
    """Contracts traded (always positive)."""
    for field in ("count_fp", "count"):
        v = _f.get(field)
        if v is not None:
            try:
                return abs(float(v))
            except (ValueError, TypeError):
                pass
    return 0.0

def _fill_action(_f: dict) -> str:
    """Returns 'buy' or 'sell'. Falls back to sign of count_fp if action absent."""
    action = _f.get("action", "")
    if action in ("buy", "sell"):
        return action
    for field in ("count_fp", "count"):
        v = _f.get(field)
        if v is not None:
            try:
                if float(v) < 0:
                    return "sell"
            except (ValueError, TypeError):
                pass
    return "buy"

def _fill_price_dollars(_f: dict) -> float:
    """Price in dollars (0–1). Tries dollar fields first, then fixed-point."""
    side = _f.get("side", "yes")
    candidates = [f"{side}_price_dollars", f"{side}_price_fixed", f"{side}_price",
                  "yes_price_dollars", "no_price_dollars", "yes_price_fixed", "price"]
    for field in candidates:
        v = _f.get(field)
        if v is not None:
            try:
                fv = float(v)
                return fv / 100 if fv > 1 else fv
            except (ValueError, TypeError):
                pass
    return 0.0

api_closed = []
for _tkr, _tkr_fills in _by_ticker.items():
    _buy_fills  = [f for f in _tkr_fills if _fill_action(f) == "buy"]
    _sell_fills = [f for f in _tkr_fills if _fill_action(f) == "sell"]
    _total_bought  = sum(_fill_count(f) for f in _buy_fills)
    _total_sold    = sum(_fill_count(f) for f in _sell_fills)
    if _total_bought == 0:
        continue
    _buy_cost      = sum(_fill_count(f) * _fill_price_dollars(f) for f in _buy_fills)
    _sell_proceeds = sum(_fill_count(f) * _fill_price_dollars(f) for f in _sell_fills)
    _remaining     = max(0.0, _total_bought - _total_sold)
    api_closed.append({
        "ticker"        : _tkr,
        "contracts"     : int(_total_bought),
        "remaining_ctr" : int(round(_remaining)),
        "buy_cost"      : _buy_cost,
        "sell_proceeds" : _sell_proceeds,
        "entry_cents"   : round(_buy_cost / _total_bought * 100),
        "side"          : _tkr_fills[0].get("side", "yes"),
        "status"        : "settled",
    })

closed_kalshi = api_closed

if _fills_err:
    st.warning(f"Fills API error: {_fills_err}")

if closed_kalshi:
    with st.expander(f"Closed / Settled Positions ({len(closed_kalshi)})"):
        need_settlement = tuple(p["ticker"] for p in closed_kalshi)
        settlement_map  = fetch_settlements(need_settlement) if need_settlement else {}

        rows = []
        for p in closed_kalshi:
            asset, expiry, strike = parse_ticker(p["ticker"])
            side        = p.get("side", "yes")
            ctrs        = p.get("contracts", 1)
            remaining   = p.get("remaining_ctr", ctrs)
            buy_cost    = p.get("buy_cost", 0)
            sell_proc   = p.get("sell_proceeds", 0)
            entry       = p.get("entry_cents", 0)
            result      = settlement_map.get(p["ticker"])

            # Settlement value: $1/contract if result matches side, else $0
            if result is not None:
                settle_val  = remaining * 1.00 if result == side else 0.0
                settle_label = ("won ✓" if result == side else "lost ✗")
                exit_label  = f"settled ({settle_label})"
            else:
                settle_val  = None
                exit_label  = "pending"

            if sell_proc > 0:
                exit_label = "sold early" if settle_val == 0.0 or result is None else exit_label + " + sold"

            if settle_val is not None:
                pnl = sell_proc - buy_cost + settle_val
            elif sell_proc > 0 and remaining == 0:
                # Fully sold before settlement; no settlement data yet
                pnl = sell_proc - buy_cost
                exit_label = "sold early"
            else:
                pnl = None

            # Compute exit price in cents for display
            ctrs_sold = ctrs - remaining
            if ctrs_sold > 0 and sell_proc > 0:
                exit_cents = round(sell_proc / ctrs_sold * 100)
            elif result is not None:
                exit_cents = 100 if result == side else 0
            else:
                exit_cents = None

            rows.append({
                "Asset"    : asset,
                "Strike"   : (f"${float(strike):,.0f}" if strike and strike.replace(".", "").isdigit() else strike),
                "Expiry"   : expiry,
                "Side"     : side.upper(),
                "Contracts": ctrs,
                "Entry ¢"  : entry,
                "Exit ¢"   : exit_cents if exit_cents is not None else "—",
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

        with st.expander("🔍 Debug: fill cash flows per position"):
            debug_rows = []
            for p in closed_kalshi:
                result = settlement_map.get(p["ticker"])
                side   = p.get("side", "yes")
                rem    = p.get("remaining_ctr", 0)
                settle = (rem * 1.00 if result == side else 0.0) if result else None
                debug_rows.append({
                    "Ticker"        : p["ticker"],
                    "Side"          : side,
                    "Bought"        : p.get("contracts", 0),
                    "Remaining"     : rem,
                    "Buy Cost $"    : f"${p.get('buy_cost', 0):.4f}",
                    "Sell Proc $"   : f"${p.get('sell_proceeds', 0):.4f}",
                    "Settle $"      : f"${settle:.2f}" if settle is not None else "—",
                    "Result"        : result or "—",
                })
            st.dataframe(pd.DataFrame(debug_rows), hide_index=True, use_container_width=True)
            if _raw_sample:
                st.caption("Most recent raw fill:")
                st.json(_raw_sample)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# KALSHI — SCAN OPPORTUNITIES
# ══════════════════════════════════════════════════════════════════════════════
st.header("Kalshi — Scan Opportunities")
st.caption("Top 5 by EV across both YES and NO sides, split by time to expiry.")

DAILY_BUDGET  = 50.0
WEEKLY_BUDGET = 200.0


def build_bucket(results: list, budget: float) -> list:
    from kalshi_crypto import MIN_EV, MIN_EDGE
    valid = [r for r in results if r["ev"] >= MIN_EV and r["edge"] >= MIN_EDGE]
    valid.sort(key=lambda x: x["ev"], reverse=True)
    groups: dict[tuple, list] = {}
    for r in valid:
        groups.setdefault((r["asset"], r["expiry"]), []).append(r)
    picks = []
    for group in sorted(groups.values(), key=lambda g: g[0]["ev"], reverse=True):
        if len(picks) >= 4:
            break
        picks.append((group[0], group[0]["kelly_pct"]))
    if not picks:
        return []
    total_weight = sum(w for _, w in picks)
    portfolio = []
    for r, weight in picks:
        dollars   = round(budget * (weight / total_weight), 2)
        contracts = max(1, int(dollars / (r["price"] / 100)))
        portfolio.append({**r,
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


def save_paper_trades(portfolio: list, bucket: str) -> None:
    existing      = db.load_paper_trades()
    existing_keys = {(p["ticker"], p["side"]) for p in existing}
    added = 0
    skipped = 0
    for p in portfolio:
        key = (p["ticker"], p["side"])
        if key in existing_keys:
            skipped += 1
            continue
        db.add_paper_trade({
            "ticker"       : p["ticker"],
            "side"         : p["side"],
            "price_cents"  : p["price"],
            "contracts"    : p["contracts_suggested"],
            "bet_dollars"  : p["kelly_dollars"],
            "model_prob"   : p["model_prob"],
            "cal_prob"     : p["calibrated_prob"],
            "ev"           : p["ev"],
            "hours_to_exp" : p["hours_to_expiry"],
            "close_time"   : p.get("close_time", ""),
            "bucket"       : bucket,
            "placed_at"    : datetime.now(timezone.utc).isoformat(),
            "status"       : "open",
            "result"       : None,
            "pnl_dollars"  : None,
        })
        added += 1
    if added:
        st.success(f"Recorded {added} new paper trade(s)." +
                   (f" ({skipped} already tracked.)" if skipped else ""))
    else:
        st.info(f"All {skipped} trade(s) already tracked — no duplicates added.")


if st.button("Run Kalshi Scan", type="primary", key="scan_kalshi"):
    try:
        from kalshi_crypto import (
            load_crypto_models, score_contract,
            download_crypto, download_crypto_hourly,
            CRYPTO_ASSETS, KALSHI_SERIES, INFERENCE_PERIOD,
        )

        with st.spinner("Loading model..."):
            models = load_crypto_models()
            _db_cal = db.load_calibration_db()
            if _db_cal:
                models["calibration"] = _db_cal

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

            under24 = [r for r in all_results if r["hours_to_expiry"] <= 24]
            over24  = [r for r in all_results if r["hours_to_expiry"] > 24]

            st.session_state["scan_daily_port"]  = build_bucket(under24, DAILY_BUDGET)
            st.session_state["scan_weekly_port"] = build_bucket(over24,  WEEKLY_BUDGET)
            st.session_state["scan_under24"]     = under24
            st.session_state["scan_over24"]      = over24
            st.session_state["scan_debug"]       = (
                " · ".join(scan_debug)
                + f" · {len(all_results)//2} contracts · {len(all_results)} sides scored"
            )

    except Exception as e:
        st.error(f"Scan error: {e}")
        import traceback; st.code(traceback.format_exc())

# ── Render scan results (persists across reruns via session_state) ─────────────
if "scan_daily_port" in st.session_state:
    daily_port  = st.session_state["scan_daily_port"]
    weekly_port = st.session_state["scan_weekly_port"]
    under24     = st.session_state["scan_under24"]
    over24      = st.session_state["scan_over24"]

    st.caption(st.session_state.get("scan_debug", ""))

    # ── Daily plays ──
    st.subheader(f"Daily Plays — ${DAILY_BUDGET:.0f} budget (≤24h)")
    if daily_port:
        st.dataframe(make_portfolio_table(daily_port), use_container_width=True, hide_index=True)
        if st.button("📝 Paper Trade Daily Plays", key="paper_daily"):
            save_paper_trades(daily_port, "daily")
    else:
        st.info("No daily contracts meet the thresholds right now.")

    # ── Weekly plays ──
    st.subheader(f"Weekly Plays — ${WEEKLY_BUDGET:.0f} budget (>24h)")
    if weekly_port:
        st.dataframe(make_portfolio_table(weekly_port), use_container_width=True, hide_index=True)
        if st.button("📝 Paper Trade Weekly Plays", key="paper_weekly"):
            save_paper_trades(weekly_port, "weekly")
    else:
        st.info("No weekly contracts available right now.")

    # ── Full scan results ──
    with st.expander("All scored contracts"):
        if under24:
            st.markdown("**≤ 24h to expiry**")
            st.dataframe(make_scan_table(
                sorted(under24, key=lambda x: x["ev"], reverse=True)[:10]),
                use_container_width=True, hide_index=True)
        if over24:
            st.markdown("**> 24h to expiry**")
            st.dataframe(make_scan_table(
                sorted(over24, key=lambda x: x["ev"], reverse=True)[:10]),
                use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAPER TRADES — TRACKING
# ══════════════════════════════════════════════════════════════════════════════
_pt_header_col, _pt_btn_col = st.columns([5, 2])
with _pt_header_col:
    st.header("Paper Trades — Tracking")
    st.caption("Model-suggested trades recorded without real money. Used to track accuracy and recalibrate.")

# ── Calibration status + recalibrate button ───────────────────────────────────
_cal_data = db.load_calibration_db()

with _pt_btn_col:
    st.write("")  # vertical spacing
    if st.button("🔁 Recalibrate from Paper Trades", use_container_width=True):
        try:
            from kalshi_crypto import recalibrate_from_paper_trades
            _settled = [t for t in db.load_paper_trades() if t.get("status") == "settled"]
            if not _settled:
                st.warning("No settled paper trades yet.")
            else:
                import tempfile, json as _j
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as _tf:
                    _j.dump(_settled + [t for t in db.load_paper_trades() if t.get("status") == "open"], _tf)
                    _tmp_path = _tf.name
                with st.spinner("Fitting calibration..."):
                    _new_cal = recalibrate_from_paper_trades(_tmp_path)
                if "error" in _new_cal:
                    st.warning(_new_cal["error"])
                else:
                    db.save_calibration_db(_new_cal)
                    _buckets = _new_cal.get("buckets", {})
                    for _bname, _bdata in _buckets.items():
                        if _bdata.get("skipped"):
                            st.info(f"{_bname}: {_bdata['reason']}")
                        else:
                            st.success(
                                f"{_bname} calibrated on {_bdata['n_trades']} trades — "
                                f"actual {_bdata['win_rate']:.0%} vs predicted {_bdata['pred_rate']:.0%}"
                            )
                    _cal_data = _new_cal
        except Exception as _ce:
            st.error(f"Calibration error: {_ce}")

if _cal_data:
    _bkts = _cal_data.get("buckets", {})
    _status_parts = []
    for _bn, _bd in _bkts.items():
        if not _bd.get("skipped"):
            _status_parts.append(f"**{_bn}** {_bd['n_trades']} trades (actual {_bd['win_rate']:.0%} vs pred {_bd['pred_rate']:.0%})")
    if _status_parts:
        _updated = _cal_data.get("updated_at", "")[:10]
        st.caption(f"Active calibration (updated {_updated}): " + " · ".join(_status_parts))
else:
    st.caption("No calibration active — click Recalibrate after accumulating 5+ settled paper trades per bucket.")

_paper = db.load_paper_trades()

if not _paper:
    st.info("No paper trades recorded yet. Run the Kalshi Scan and click '📝 Paper Trade' to start tracking.")
else:
    # ── Auto-settle expired trades ────────────────────────────────────────────
    _now_utc       = datetime.now(timezone.utc)
    _newly_settled = 0
    _open_paper    = []
    _settled_paper = []

    for _pt in _paper:
        if _pt.get("status") == "open":
            _close_str = _pt.get("close_time", "")
            _closed    = False
            if _close_str:
                try:
                    _close_dt = datetime.fromisoformat(_close_str.replace("Z", "+00:00"))
                    _closed   = _close_dt <= _now_utc
                except Exception:
                    # Non-ISO close_time (old records stored date string like "26MAR2801").
                    # Fall through to API check — if the contract settled, result != None.
                    _closed = True
            else:
                _closed = True  # no close_time at all → check API
            if _closed and not _client.dry_run:
                try:
                    _mkt    = _client._request("GET", f"/markets/{_pt['ticker']}").get("market", {})
                    _result = _mkt.get("result")
                    if _result is not None:
                        _side  = _pt.get("side", "yes")
                        _entry = _pt.get("price_cents", 50)
                        _ctrs  = _pt.get("contracts", 1)
                        _pnl   = (round((100 - _entry) * _ctrs / 100, 2)
                                  if _result == _side
                                  else round(-_entry * _ctrs / 100, 2))
                        db.settle_paper_trade(_pt["id"], _result, _pnl)
                        _pt["status"]      = "settled"
                        _pt["result"]      = _result
                        _pt["pnl_dollars"] = _pnl
                        _newly_settled    += 1
                except Exception:
                    pass
        if _pt.get("status") == "open":
            _open_paper.append(_pt)
        else:
            _settled_paper.append(_pt)

    if _newly_settled:
        st.success(f"Auto-settled {_newly_settled} paper trade(s).")

    # ── Open paper trades ─────────────────────────────────────────────────────
    if _open_paper:
        st.subheader(f"Open Paper Trades ({len(_open_paper)})")
        _open_tickers_pt = tuple(p["ticker"] for p in _open_paper)
        _pt_live = fetch_live_prices(_open_tickers_pt) if not _client.dry_run else {}

        _pt_rows = []
        for _pt in _open_paper:
            _entry = _pt.get("price_cents", 50)
            _ctrs  = _pt.get("contracts", 1)
            _bid   = _pt_live.get(_pt["ticker"])
            _hrs   = hours_left(_pt.get("close_time", ""))
            _side  = _pt.get("side", "yes")
            # Use settlement value (100¢) if market expired and bid dropped to 0
            # (bid=0 on a closed winning contract would show a false loss otherwise)
            if _bid == 0 and _hrs is not None and _hrs < 0:
                _bid = None  # can't use live bid for expired markets
            _unreal = round((_bid - _entry) * _ctrs / 100, 2) if _bid is not None else None
            _rec_h = _pt.get("hours_to_exp")
            _pt_rows.append({
                "Ticker"    : _pt["ticker"],
                "Side"      : _pt.get("side", "yes"),
                "Bucket"    : _pt.get("bucket", ""),
                "Entry ¢"   : _entry,
                "Live Bid"  : f"{_bid}¢" if _bid is not None else "—",
                "Contracts" : _ctrs,
                "Bet $"     : f"${_pt.get('bet_dollars', 0):.0f}",
                "Cal Prob"  : f"{_pt.get('cal_prob', 0)*100:.1f}%",
                "At Rec"    : (f"{int(_rec_h*60)}m" if _rec_h is not None and _rec_h < 1
                               else f"{_rec_h:.0f}h" if _rec_h is not None else "—"),
                "Hrs Left"  : (f"{int(_hrs*60)}m" if _hrs is not None and _hrs < 1
                               else f"{_hrs:.0f}h" if _hrs is not None else "—"),
                "Unreal P&L": (f"${_unreal:+.2f}" if _unreal is not None else "—"),
            })
        st.dataframe(
            pd.DataFrame(_pt_rows).style.map(color_pnl, subset=["Unreal P&L"]),
            hide_index=True, use_container_width=True,
        )

    # ── Settled paper trades ──────────────────────────────────────────────────
    if _settled_paper:
        with st.expander(f"Settled Paper Trades ({len(_settled_paper)})"):
            _s_rows = []
            for _pt in sorted(_settled_paper, key=lambda x: x.get("placed_at", ""), reverse=True):
                _pnl = _pt.get("pnl_dollars")
                _rec_h = _pt.get("hours_to_exp")
                _s_rows.append({
                    "Ticker"   : _pt["ticker"],
                    "Side"     : _pt.get("side", "yes"),
                    "Bucket"   : _pt.get("bucket", ""),
                    "At Rec"   : (f"{int(_rec_h*60)}m" if _rec_h is not None and _rec_h < 1
                                  else f"{_rec_h:.0f}h" if _rec_h is not None else "—"),
                    "Entry ¢"  : _pt.get("price_cents", 0),
                    "Cal Prob" : f"{_pt.get('cal_prob', 0)*100:.1f}%",
                    "Result"   : _pt.get("result", "—"),
                    "P&L $"    : (f"${_pnl:+.2f}" if _pnl is not None else "—"),
                    "Placed"   : _pt.get("placed_at", "")[:10],
                })
            st.dataframe(
                pd.DataFrame(_s_rows).style.map(color_pnl, subset=["P&L $"]),
                hide_index=True, use_container_width=True,
            )

    # ── Calibration summary ───────────────────────────────────────────────────
    _resolved = [p for p in _settled_paper if p.get("pnl_dollars") is not None]
    if _resolved:
        st.subheader("Calibration")
        _wins       = sum(1 for p in _resolved
                          if p.get("pnl_dollars", 0) > 0)
        _actual_wr  = _wins / len(_resolved)
        _pred_wr    = sum(p.get("cal_prob", 0.5) for p in _resolved) / len(_resolved)
        _total_pnl  = sum(p.get("pnl_dollars", 0) for p in _resolved)

        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("Settled Trades", len(_resolved))
        _c2.metric("Actual Win Rate", f"{_actual_wr:.0%}")
        _c3.metric("Predicted Win Rate", f"{_pred_wr:.0%}",
                   delta=f"{(_actual_wr - _pred_wr)*100:+.1f}pp")
        _c4.metric("Total P&L", f"${_total_pnl:+.2f}")

        # Bucket breakdown
        _buckets = {}
        for _p in _resolved:
            _b = _p.get("bucket", "other")
            _buckets.setdefault(_b, []).append(_p)
        if len(_buckets) > 1:
            _bk_rows = []
            for _b, _bps in sorted(_buckets.items()):
                _bwins = sum(1 for p in _bps if p.get("pnl_dollars", 0) > 0)
                _bpnl  = sum(p.get("pnl_dollars", 0) for p in _bps)
                _bpred = sum(p.get("cal_prob", 0.5) for p in _bps) / len(_bps)
                _bk_rows.append({
                    "Bucket"      : _b,
                    "Trades"      : len(_bps),
                    "Win Rate"    : f"{_bwins/len(_bps):.0%}",
                    "Pred Win Rate": f"{_bpred:.0%}",
                    "Total P&L"   : f"${_bpnl:+.2f}",
                })
            st.dataframe(pd.DataFrame(_bk_rows), hide_index=True, use_container_width=True)

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
