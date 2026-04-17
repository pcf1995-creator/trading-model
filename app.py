"""
app.py — Trading Dashboard
Run: streamlit run app.py
"""

import importlib.util
import json
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
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

tab_dash, tab_lt, tab_conviction, tab_perf = st.tabs(["📊 Dashboard", "📈 Long-Term L/S", "🎯 Conviction L/S", "📉 Performance"])

with tab_dash:

    # ── Helpers ────────────────────────────────────────────────────────────────────
    def load_json(path: Path) -> list:
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)


    def hours_left(close_time_str: str, ticker: str = "") -> float | None:
        if close_time_str:
            try:
                dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                return (dt - datetime.now(timezone.utc)).total_seconds() / 3600
            except Exception:
                pass
        # Fallback: parse close time from ticker e.g. KXBTCD-26APR0317-T70000
        if ticker:
            try:
                _MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
                code = ticker.split("-")[1]           # e.g. "26APR0317"
                yr   = 2000 + int(code[:2])
                mon  = _MONTHS[code[2:5]]
                day  = int(code[5:7])
                hr   = int(code[7:9])                 # ET hour (17 = 5pm ET)
                from zoneinfo import ZoneInfo
                et   = ZoneInfo("America/New_York")
                dt   = datetime(yr, mon, day, hr, 0, 0, tzinfo=et)
                return (dt - datetime.now(timezone.utc)).total_seconds() / 3600
            except Exception:
                pass
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


    def get_bid_cents(market: dict, side: str = "yes") -> int | None:
        """Return the current bid price in cents for the held side.
        YES bid = what you receive selling YES.
        NO bid  = 100 - yes_ask (what you receive selling NO).
        """
        if side.lower() == "no":
            # Prefer explicit no_bid fields, fall back to 100 - yes_ask
            for key in ("no_bid_dollars", ):
                v = market.get(key)
                if v is not None:
                    return round(float(v) * 100)
            no_bid_fp = market.get("no_bid_fp")
            if no_bid_fp is not None:
                return round(no_bid_fp / 100)
            no_bid = market.get("no_bid")
            if no_bid is not None:
                return int(no_bid)
            # Compute from yes_ask
            for key, scale in (("yes_ask_dollars", 100), ("yes_ask_fp", 0.01), ("yes_ask", 1)):
                v = market.get(key)
                if v is not None:
                    return max(0, 100 - round(float(v) * scale))
            return None
        # YES side
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
    def fetch_live_prices(tickers: tuple, sides: tuple | None = None) -> dict[str, int | None]:
        """Live bid for each ticker, using the correct side (YES bid or NO bid).
        sides: tuple of 'yes'/'no' matching tickers order. Defaults to all 'yes'.
        """
        client = make_kalshi_client()
        if client.dry_run:
            return {}
        side_map = dict(zip(tickers, sides)) if sides else {}
        prices = {}
        for ticker in tickers:
            try:
                mkt = client.get_market(ticker)
                prices[ticker] = get_bid_cents(mkt, side_map.get(ticker, "yes"))
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

    # Pre-fetch fills to compute avg entry prices from actual buy fills
    _fills_index: dict[str, dict] = {}   # ticker -> {yes: cents, no: cents}
    if not _client.dry_run:
        try:
            _prefetch_fills = _client.get_fills(limit=2000)
            _ticker_buy_fills: dict[str, list] = {}
            for _pf in _prefetch_fills:
                _pt = _pf.get("market_ticker") or _pf.get("ticker", "")
                if _pt and (_pt.startswith("KXBTC") or _pt.startswith("KXETH")):
                    _ticker_buy_fills.setdefault(_pt, []).append(_pf)
            for _pt, _pf_list in _ticker_buy_fills.items():
                _yes_cost = _yes_cnt = _no_cost = _no_cnt = 0.0
                for _pf in _pf_list:
                    if _pf.get("action", "buy") != "buy":
                        continue
                    _cnt = abs(float(_pf.get("count") or _pf.get("count_fp") or 0))
                    _fside = _pf.get("side", "yes")
                    _yp = float(_pf.get("yes_price_dollars") or _pf.get("yes_price") or 0)
                    if _yp > 1: _yp /= 100
                    _np = max(0.0, 1.0 - _yp) if _yp > 0 else 0.0
                    if _fside == "yes":
                        _yes_cost += _cnt * _yp; _yes_cnt += _cnt
                    else:
                        _no_cost += _cnt * _np; _no_cnt += _cnt
                _fills_index[_pt] = {
                    "yes": round(_yes_cost / _yes_cnt * 100) if _yes_cnt else None,
                    "no":  round(_no_cost  / _no_cnt  * 100) if _no_cnt  else None,
                }
        except Exception:
            pass

    if not _client.dry_run:
        try:
            _api_positions = _client.get_positions()
            _all_api = []
            for _pos in _api_positions:
                _tkr = _pos.get("ticker", "")
                # Only show crypto positions in this dashboard
                if not _tkr or not (_tkr.startswith("KXBTC") or _tkr.startswith("KXETH")):
                    continue
                # Match monitor.py: position_fp is same scale as position (no /100 needed)
                _pos_val = _pos.get("position")
                _net_pos = _pos_val if _pos_val is not None else round(float(_pos.get("position_fp", 0) or 0))
                if _net_pos == 0:
                    continue  # fully closed position, skip
                _side  = "yes" if _net_pos > 0 else "no"
                _local = _local_by_ticker.get(_tkr, {})
                _mkt   = _client.get_market(_tkr)
                _hrs   = hours_left(_mkt.get("close_time", ""))
                # Priority: saved entry → fills-based avg → live bid proxy
                _saved_entry = _local.get("entry_cents")
                _fills_entry = _fills_index.get(_tkr, {}).get(_side)
                _proxy_entry = get_bid_cents(_mkt, _side) or 0
                _entry = _saved_entry if _saved_entry else (_fills_entry if _fills_entry else _proxy_entry)
                # Use saved stop if present and non-zero; fall back to 50% of entry
                _saved_stop = _local.get("stop_cents")
                _stop = _saved_stop if _saved_stop else round(_entry * 0.5)
                _api_contracts = abs(_net_pos) if _net_pos != 0 else 1
                _all_api.append({
                    "ticker"      : _tkr,
                    "status"      : "open",
                    "side"        : _side,
                    # Prefer manually saved contracts; fall back to API position count
                    "contracts"   : _local.get("contracts") or _api_contracts,
                    "entry_cents" : _entry,
                    "stop_cents"  : _stop,
                    "_entry_proxy": not bool(_saved_entry or _fills_entry),
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
        open_sides   = tuple(p.get("side", "yes") for p in open_kalshi)
        with st.spinner("Loading live prices..."):
            live = fetch_live_prices(open_tickers, open_sides)

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

            bet_dollars  = entry * contracts / 100
            stop_dollars = stop * contracts / 100
            entry_proxy  = p.get("_entry_proxy", False)
            rows.append({
                "Ticker"   : ticker,
                "Asset"    : asset,
                "Strike"   : (f"${float(strike):,.0f}" if strike and strike.replace(".", "").isdigit() else strike),
                "Hrs Left" : (f"{int(hrs * 60)}m" if hrs is not None and hrs < 1
                              else f"{hrs:.0f}h" if hrs is not None else "—"),
                "Contracts": contracts,
                "Entry ¢"  : entry,
                "Bet $"    : f"~${bet_dollars:.2f}" if entry_proxy else f"${bet_dollars:.2f}",
                "Stop ¢"   : stop,
                "Stop $"   : f"${stop_dollars:.2f}",
                "Live Bid" : f"{current}¢ ({'NO' if p.get('side','yes').lower()=='no' else 'YES'})" if current is not None else "—",
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
            disabled=["Asset", "Strike", "Hrs Left", "Bet $", "Stop $", "Live Bid", "P&L"],
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
            st.cache_data.clear()
            st.rerun()

        # Compute metrics from edited table so they reflect current edits before saving
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Open Positions", len(open_kalshi))
        expiring_soon = sum(1 for p in open_kalshi
                            if (h := hours_left(p.get("close_time", ""))) is not None and h < 12)
        m2.metric("Expiring < 12h", expiring_soon)
        total_exp = sum(int(row["Entry ¢"]) * int(row["Contracts"])
                        for _, row in edited.iterrows())
        m3.metric("Total Exposure", f"${total_exp/100:.2f}")
        stops_at_risk = sum(
            1 for i, row in edited.iterrows()
            if int(row["Stop ¢"]) > 0
            and live.get(df_open.iloc[i]["Ticker"], 999) <= int(row["Stop ¢"])
        )
        m4.metric("At Stop", stops_at_risk, delta=None)

        # ── Stop-loss execution ────────────────────────────────────────────────────
        if st.button("🛑 Execute Stop-Losses", type="primary" if stops_at_risk else "secondary",
                     disabled=_client.dry_run):
            _stops_executed = 0
            _stop_errors    = []
            for p in open_kalshi:
                _bid  = live.get(p["ticker"])
                _stop = p.get("stop_cents", 0)
                if _stop == 0 or _bid is None:
                    continue
                if _bid <= _stop:
                    _side  = p.get("side", "yes")
                    _count = p["contracts"]
                    try:
                        _result = _client.sell_position(p["ticker"], _side, _count, _bid)
                        st.success(f"Sold {_count} {_side.upper()} {p['ticker']} @ {_bid}¢")
                        _stops_executed += 1
                    except Exception as _e:
                        _stop_errors.append(f"{p['ticker']}: {_e}")
            if _stops_executed == 0 and not _stop_errors:
                st.info("No positions currently at or below stop-loss.")
            for _err in _stop_errors:
                st.error(f"Order failed — {_err}")
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

    def _price_dollars(_f: dict, field_prefix: str) -> float:
        """Extract YES or NO price in dollars (0–1) from a fill."""
        for suffix in ("_dollars", "_fixed", ""):
            v = _f.get(f"{field_prefix}{suffix}")
            if v is not None:
                try:
                    fv = float(v)
                    return fv / 100 if fv > 1 else fv
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _no_price_dollars(_f: dict) -> float:
        """NO price in dollars. Kalshi fills only include yes_price; derive NO price as 1 - yes_price."""
        np = _price_dollars(_f, "no_price")
        if np > 0:
            return np
        yp = _price_dollars(_f, "yes_price")
        if yp > 0:
            return max(0.0, 1.0 - yp)
        return 0.0

    # Process fills per ticker, tracking YES and NO positions separately.
    # Kalshi fill convention: "sell yes" = closing a NO position (proceeds = no_price);
    #                         "sell no"  = closing a YES position (proceeds = yes_price).
    # Fills with identical (ts, action, side, count, price) are deduplicated — Kalshi
    # sometimes emits duplicate records for the same fill.
    api_closed = []
    for _tkr, _tkr_fills in _by_ticker.items():
        # Deduplicate fills
        _seen_keys: set = set()
        _deduped: list  = []
        for _f in _tkr_fills:
            _dk = (
                _f.get("ts") or _f.get("created_time", ""),
                _f.get("action", ""),
                _f.get("side", ""),
                _fill_count(_f),
                round(_price_dollars(_f, "yes_price") * 10000),
            )
            if _dk not in _seen_keys:
                _seen_keys.add(_dk)
                _deduped.append(_f)

        _sorted_fills = sorted(
            _deduped,
            key=lambda f: f.get("ts") or f.get("created_time", "")
        )

        # YES side: buy yes + sell no (YES exits)
        _yes_buy_cost      = 0.0
        _yes_sell_proceeds = 0.0
        _yes_bought        = 0.0
        _yes_pos           = 0.0

        # NO side: buy no + sell yes (NO exits)
        _no_buy_cost       = 0.0
        _no_sell_proceeds  = 0.0
        _no_bought         = 0.0
        _no_pos            = 0.0

        for _f in _sorted_fills:
            _cnt   = _fill_count(_f)
            _act   = _fill_action(_f)
            _fside = _f.get("side", "yes")
            _yp    = _price_dollars(_f, "yes_price")
            _np    = _no_price_dollars(_f)

            if _act == "buy" and _fside == "yes":
                _yes_buy_cost += _cnt * _yp
                _yes_pos      += _cnt
                _yes_bought   += _cnt
            elif _act == "sell" and _fside == "no":
                # Closing a YES position — Kalshi records as sell/no, price = yes_price
                _yes_sell_proceeds += _cnt * _yp
                _yes_pos           -= _cnt
            elif _act == "buy" and _fside == "no":
                _no_buy_cost += _cnt * _np
                _no_pos      += _cnt
                _no_bought   += _cnt
            elif _act == "sell" and _fside == "yes":
                # Closing a NO position — Kalshi records as sell/yes, price = no_price
                _no_sell_proceeds += _cnt * _np
                _no_pos           -= _cnt

        if _yes_bought > 0:
            api_closed.append({
                "ticker"        : _tkr,
                "contracts"     : int(_yes_bought),
                "rem_yes"       : int(round(max(0.0, _yes_pos))),
                "rem_no"        : 0,
                "buy_cost"      : _yes_buy_cost,
                "sell_proceeds" : _yes_sell_proceeds,
                "entry_cents"   : round(_yes_buy_cost / _yes_bought * 100),
                "side"          : "yes",
                "status"        : "settled",
            })
        if _no_bought > 0:
            api_closed.append({
                "ticker"        : _tkr,
                "contracts"     : int(_no_bought),
                "rem_yes"       : 0,
                "rem_no"        : int(round(max(0.0, _no_pos))),
                "buy_cost"      : _no_buy_cost,
                "sell_proceeds" : _no_sell_proceeds,
                "entry_cents"   : round(_no_buy_cost / _no_bought * 100),
                "side"          : "no",
                "status"        : "settled",
            })

    closed_kalshi  = api_closed
    settlement_map = {}

    if _fills_err:
        st.warning(f"Fills API error: {_fills_err}")

    if closed_kalshi:
        need_settlement = tuple(p["ticker"] for p in closed_kalshi)
        settlement_map  = fetch_settlements(need_settlement) if need_settlement else {}

        # ── Auto-settle matching paper trades ─────────────────────────────────
        _open_pts    = [t for t in db.load_paper_trades() if t.get("status") == "open"]
        _open_pt_map = {(t["ticker"], t.get("side", "yes").lower()): t for t in _open_pts}
        _auto_settled = 0
        for _cp in closed_kalshi:
            _tkr  = _cp["ticker"]
            _pt   = _open_pt_map.get((_tkr, _cp.get("side", "yes").lower()))
            if not _pt:
                continue
            _buy_cost  = _cp.get("buy_cost", 0)
            _sell_proc = _cp.get("sell_proceeds", 0)
            _remaining = _cp.get("rem_yes", 0) + _cp.get("rem_no", 0)
            _total     = _cp.get("contracts", 1)
            _result    = settlement_map.get(_tkr)  # "yes" / "no" / None
            # Determine settlement value for any remaining contracts
            if _result is not None and _remaining > 0:
                _side = _cp.get("side", "yes")
                _settle_val = _remaining * 1.0 if (
                    (_side == "yes" and _result == "yes") or
                    (_side == "no"  and _result == "no")
                ) else 0.0
                _pnl = round(_sell_proc - _buy_cost + _settle_val, 2)
            elif _remaining == 0:
                _pnl = round(_sell_proc - _buy_cost, 2)
                _result = _result  # keep if known, else None
            else:
                continue  # still open contracts, can't settle yet
            # Use market result if known; otherwise infer from P&L
            _pt_result = _result if _result in ("yes", "no") else ("yes" if _pnl > 0 else "no")
            db.settle_paper_trade(_pt["id"], _pt_result, _pnl)
            _auto_settled += 1
        if _auto_settled:
            st.toast(f"Auto-settled {_auto_settled} paper trade(s) from fills.", icon="✅")

        with st.expander(f"Closed / Settled Positions ({len(closed_kalshi)})"):

            rows = []
            for p in closed_kalshi:
                asset, expiry, strike = parse_ticker(p["ticker"])
                side      = p.get("side", "yes")
                ctrs      = p.get("contracts", 1)
                rem_yes   = p.get("rem_yes", 0)
                rem_no    = p.get("rem_no", 0)
                remaining = rem_yes + rem_no
                buy_cost  = p.get("buy_cost", 0)
                sell_proc = p.get("sell_proceeds", 0)
                fees      = p.get("total_fees", 0)
                entry     = p.get("entry_cents", 0)
                result    = settlement_map.get(p["ticker"])

                # Settlement value: YES contracts win on result="yes", NO contracts win on result="no"
                if result is not None:
                    settle_val = rem_yes * 1.00 if result == "yes" else 0.0
                    settle_val += rem_no * 1.00 if result == "no" else 0.0
                    won = (rem_yes > 0 and result == "yes") or (rem_no > 0 and result == "no")
                    exit_label = f"settled ({'won ✓' if won else 'lost ✗'})"
                else:
                    settle_val = None
                    exit_label = "pending"

                if sell_proc > 0 and remaining == 0:
                    exit_label = "sold" if settle_val is None else exit_label + " + sold"
                elif sell_proc > 0:
                    exit_label = "partial sell + " + (exit_label if settle_val is not None else "pending")

                if settle_val is not None:
                    pnl = sell_proc - buy_cost + settle_val - fees
                elif remaining == 0:
                    pnl = sell_proc - buy_cost - fees
                    exit_label = "sold"
                else:
                    pnl = None

                # Exit ¢: avg price received per contract sold (in the dominant side's terms)
                ctrs_sold = ctrs - remaining
                if ctrs_sold > 0 and sell_proc > 0:
                    exit_cents = round(sell_proc / ctrs_sold * 100)
                elif result is not None:
                    exit_cents = 100 if won else 0
                else:
                    exit_cents = None

                pnl_pct = (pnl / buy_cost * 100) if (pnl is not None and buy_cost > 0) else None
                rows.append({
                    "Asset"    : asset,
                    "Strike"   : (f"${float(strike):,.0f}" if strike and strike.replace(".", "").isdigit() else strike),
                    "Expiry"   : expiry,
                    "Side"     : side.upper(),
                    "Contracts": ctrs,
                    "Entry ¢"  : entry,
                    "Entry $"  : f"${buy_cost:.2f}",
                    "Exit ¢"   : exit_cents if exit_cents is not None else "—",
                    "Exit"     : exit_label,
                    "P&L $"    : (f"${pnl:+.2f}" if pnl is not None else "—"),
                    "P&L %"    : (f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—"),
                })

            df_closed = pd.DataFrame(rows)
            st.dataframe(df_closed.style.map(color_pnl, subset=["P&L $", "P&L %"]),
                         hide_index=True, use_container_width=True)

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
                    result  = settlement_map.get(p["ticker"])
                    ry, rn  = p.get("rem_yes", 0), p.get("rem_no", 0)
                    settle  = None
                    if result is not None:
                        settle = (ry if result == "yes" else 0) + (rn if result == "no" else 0)
                    debug_rows.append({
                        "Ticker"        : p["ticker"],
                        "Side"          : p.get("side", "yes"),
                        "Bought"        : p.get("contracts", 0),
                        "Rem YES"       : ry,
                        "Rem NO"        : rn,
                        "Buy Cost $"    : f"${p.get('buy_cost', 0):.4f}",
                        "Sell Proc $"   : f"${p.get('sell_proceeds', 0):.4f}",
                        "Settle $"      : f"${settle:.2f}" if settle is not None else "—",
                        "Result"        : result or "—",
                    })
                st.dataframe(pd.DataFrame(debug_rows), hide_index=True, use_container_width=True)

                st.caption("Raw fills per ticker (expand to diagnose wrong P&L):")
                for _tkr, _tkr_fills in _by_ticker.items():
                    with st.expander(_tkr):
                        st.json([{
                            "ts"           : f.get("ts") or f.get("created_time"),
                            "action"       : f.get("action"),
                            "side"         : f.get("side"),
                            "count"        : _fill_count(f),
                            "yes_price"    : _price_dollars(f, "yes_price"),
                            "no_price"     : _price_dollars(f, "no_price"),
                            "fee"          : f.get("fee"),
                            "fees"         : f.get("fees"),
                            "fee_dollars"  : f.get("fee_dollars"),
                            "fee_fp"       : f.get("fee_fp"),
                        } for f in sorted(_tkr_fills,
                                          key=lambda f: f.get("ts") or f.get("created_time", ""))])

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # KALSHI — SCAN OPPORTUNITIES
    # ══════════════════════════════════════════════════════════════════════════════
    st.header("Kalshi — Scan Opportunities")
    st.caption("Top 5 by EV across both YES and NO sides, split by time to expiry.")

    DAILY_BUDGET  = 50.0
    WEEKLY_BUDGET = 200.0

    # Correlation constants for portfolio construction.
    # Each same-direction pick's Kelly allocation is multiplied by (1 - corr)
    # relative to the most correlated earlier pick.
    # Tune as paper trade data accumulates.
    SAME_ASSET_SAME_SIDE_CORR = 0.90  # BTC YES + BTC YES — nearly identical bets
    BTC_ETH_CORR               = 0.80  # BTC YES + ETH YES — correlated direction


    def build_bucket(results: list, budget: float) -> list:
        from kalshi_crypto import MIN_EV, MIN_EDGE
        valid = [r for r in results if r["ev"] >= MIN_EV and r["edge"] >= MIN_EDGE]
        valid.sort(key=lambda x: x["ev"], reverse=True)

        # Pick top contracts by EV globally (no per-asset grouping).
        # Deduplicate by ticker so the same contract can't appear twice.
        seen: set[str] = set()
        picks: list[dict] = []
        for r in valid:
            if len(picks) >= 4:
                break
            if r["ticker"] in seen:
                continue
            seen.add(r["ticker"])
            picks.append(r)

        if not picks:
            return []

        # For each pick, compute a correlation discount vs the picks before it.
        # Two picks are "correlated" when they are same-direction BTC+ETH bets
        # (e.g. both NO, meaning both win only if crypto drops).
        def _corr_discount(r: dict, earlier: list[dict]) -> float:
            """Return the allocation multiplier based on highest correlation with any earlier pick."""
            max_corr = 0.0
            for e in earlier:
                if e["side"].lower() == r["side"].lower():
                    if e["asset"] == r["asset"]:
                        max_corr = max(max_corr, SAME_ASSET_SAME_SIDE_CORR)
                    else:
                        max_corr = max(max_corr, BTC_ETH_CORR)
            return 1.0 - max_corr

        discounts = []
        for i, r in enumerate(picks):
            discounts.append(_corr_discount(r, picks[:i]))

        # Size each pick as kelly_pct% of budget, discounted for correlation.
        # Only normalize down if the total exceeds the budget — weak single picks
        # get a proportionally small allocation, leaving room for better plays later.
        raw = [r["kelly_pct"] / 100 * budget * d for r, d in zip(picks, discounts)]
        total_raw = sum(raw) or 1.0
        scale = min(1.0, budget / total_raw)

        portfolio = []
        for r, raw_dollars, d in zip(picks, raw, discounts):
            dollars   = round(raw_dollars * scale, 2)
            if dollars < 1.0:
                continue   # correlation-discounted to nothing — skip
            contracts = max(1, int(dollars / (r["price"] / 100)))
            portfolio.append({**r,
                "kelly_dollars"      : dollars,
                "contracts_suggested": contracts,
                "correlated"         : d < 1.0,
            })
        return portfolio


    def make_portfolio_table(portfolio: list) -> pd.DataFrame:
        rows = []
        for p in portfolio:
            sd = p.get("strike_distance")  # raw: (strike/current - 1) * 100
            # Make side-aware: + = in the money, - = need a move to win
            if sd is not None:
                itm = -sd if p["side"].upper() == "YES" else sd
                pct_str = f"{itm:+.1f}%"
            else:
                pct_str = "—"
            rows.append({
                "Ticker"      : p["ticker"],
                "Side"        : p["side"],
                "Hrs Left"    : (f"{int(p['hours_to_expiry'] * 60)}m"
                                  if p['hours_to_expiry'] < 1
                                  else f"{p['hours_to_expiry']:.0f}h"),
                "% to Strike" : pct_str,
                "Price"       : f"{p['price']}¢",
                "Cal Prob"    : f"{(1-p['calibrated_prob'] if p['side'].lower()=='no' else p['calibrated_prob'])*100:.1f}%",
                "EV"          : f"{p['ev']:+.3f}",
                "Bet $"       : f"${p['kelly_dollars']:.0f}",
                "Contracts"   : p["contracts_suggested"],
                "Model"       : p.get("model_type", "daily"),
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
                "Cal Prob" : f"{(1-r['calibrated_prob'] if r['side'].lower()=='no' else r['calibrated_prob'])*100:.1f}%",
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
            try:
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
                    "bucket"       : (
                        "vol"             if p["hours_to_expiry"] < 1
                        else "intraday_short" if p["hours_to_expiry"] < 8
                        else "intraday_long"  if p["hours_to_expiry"] <= 24
                        else "weekly"
                    ),
                    "placed_at"    : datetime.now(timezone.utc).isoformat(),
                    "status"       : "open",
                    "result"       : None,
                    "pnl_dollars"  : None,
                })
                added += 1
            except Exception as _db_err:
                st.warning(f"Failed to save paper trade {p['ticker']} {p['side']}: {_db_err}")
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
                fetch_binance_minute_closes, BINANCE_SYMBOLS,
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
                        daily_df      = download_crypto(symbol, INFERENCE_PERIOD)
                        hourly_df     = download_crypto_hourly(symbol) if has_intraday else None
                        binance_sym   = BINANCE_SYMBOLS.get(symbol)
                        minute_closes = fetch_binance_minute_closes(binance_sym) if binance_sym else []
                        asset_dfs_by_symbol[symbol] = {"daily": daily_df, "hourly": hourly_df, "minute": minute_closes}

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

                sub1hr  = [r for r in all_results if r["hours_to_expiry"] < 1]
                under24 = [r for r in all_results if 1 <= r["hours_to_expiry"] <= 24]
                over24  = [r for r in all_results if r["hours_to_expiry"] > 24]

                _vol_port    = build_bucket(sub1hr,  20)
                _daily_port  = build_bucket(under24, DAILY_BUDGET)
                _weekly_port = build_bucket(over24,  WEEKLY_BUDGET)
                st.session_state["scan_vol_port"]    = _vol_port
                st.session_state["scan_daily_port"]  = _daily_port
                st.session_state["scan_weekly_port"] = _weekly_port
                st.session_state["scan_under24"]     = under24
                st.session_state["scan_over24"]      = over24
                st.session_state["scan_debug"]       = (
                    " · ".join(scan_debug)
                    + f" · {len(all_results)//2} contracts · {len(all_results)} sides scored"
                )
                # Auto-record all plays as paper trades, bucketed by model type
                if _vol_port:
                    save_paper_trades(_vol_port, "vol")
                if _daily_port:
                    save_paper_trades(_daily_port, "daily")
                if _weekly_port:
                    save_paper_trades(_weekly_port, "weekly")

        except Exception as e:
            st.error(f"Scan error: {e}")
            import traceback; st.code(traceback.format_exc())

    # ── Render scan results (persists across reruns via session_state) ─────────────
    if "scan_daily_port" in st.session_state:
        vol_port    = st.session_state.get("scan_vol_port", [])
        daily_port  = st.session_state["scan_daily_port"]
        weekly_port = st.session_state["scan_weekly_port"]
        under24     = st.session_state["scan_under24"]
        over24      = st.session_state["scan_over24"]

        st.caption(st.session_state.get("scan_debug", ""))

        # ── Vol model plays (<1hr) ──
        st.subheader("< 1hr Plays — $20 budget (vol model)")
        st.caption("Priced via realized-vol binary option model (Binance 1m data). Recorded as paper trades in the \"vol\" bucket.")
        if vol_port:
            st.dataframe(make_portfolio_table(vol_port), use_container_width=True, hide_index=True)
        else:
            st.info("No <1hr vol model plays meet the thresholds right now.")

        # ── Daily plays ──
        st.subheader(f"Daily Plays — ${DAILY_BUDGET:.0f} budget (1–24h)")
        if daily_port:
            st.dataframe(make_portfolio_table(daily_port), use_container_width=True, hide_index=True)
        else:
            st.info("No daily contracts meet the thresholds right now.")

        # ── Weekly plays ──
        st.subheader(f"Weekly Plays — ${WEEKLY_BUDGET:.0f} budget (>24h)")
        if weekly_port:
            st.dataframe(make_portfolio_table(weekly_port), use_container_width=True, hide_index=True)
        else:
            st.info("No weekly contracts available right now.")

    # ── Strike Explorer ───────────────────────────────────────────────────────
    with st.expander("🔍 Strike Explorer — browse all strikes for an expiry"):
        # Reuses scored results already computed during the last scan
        _sx_all_results = (
            st.session_state.get("scan_under24", []) +
            st.session_state.get("scan_over24",  [])
        )
        if not _sx_all_results:
            st.info("Run the Kalshi Scan first — Strike Explorer filters from those results.")
        else:
            _sx_col1, _sx_col2, _sx_col3 = st.columns(3)
            _sx_series = _sx_col1.selectbox("Series", ["KXBTCD", "KXETHD"], key="sx_series")
            _sx_expiry = _sx_col2.text_input("Expiry code (e.g. 26APR1717)", value="26APR1717", key="sx_expiry")
            _sx_col3a, _sx_col3b = _sx_col3.columns(2)
            _sx_lo = float(_sx_col3a.number_input("Strike min $", value=71500, step=500, key="sx_lo"))
            _sx_hi = float(_sx_col3b.number_input("Strike max $", value=77000, step=500, key="sx_hi"))

            _sx_prefix = f"{_sx_series}-{_sx_expiry}-T"
            _sx_filtered = [
                r for r in _sx_all_results
                if r.get("ticker", "").startswith(_sx_prefix)
            ]

            # Group by strike — one row per strike showing both YES and NO
            _sx_by_strike = {}
            for _r in _sx_filtered:
                try:
                    _strike = float(_r["ticker"].split("-T")[-1])
                except Exception:
                    continue
                if not (_sx_lo <= _strike <= _sx_hi):
                    continue
                _sx_by_strike.setdefault(_strike, {})["_hrs"]  = _r.get("hours_to_expiry")
                _sx_by_strike[_strike]["_pct"] = _r.get("pct_to_strike")
                if _r.get("side", "").upper() == "YES":
                    _sx_by_strike[_strike]["yes_ask"]  = _r.get("yes_ask_cents")
                    _sx_by_strike[_strike]["cal_yes"]  = _r.get("calibrated_prob")
                    _sx_by_strike[_strike]["ev_yes"]   = _r.get("ev")
                else:
                    _sx_by_strike[_strike]["no_ask"]   = _r.get("no_ask_cents")
                    _sx_by_strike[_strike]["cal_no"]   = _r.get("calibrated_prob")
                    _sx_by_strike[_strike]["ev_no"]    = _r.get("ev")

            _sx_rows = []
            for _strike in sorted(_sx_by_strike):
                _d = _sx_by_strike[_strike]
                _cal_yes = _d.get("cal_yes")
                _cal_no  = _d.get("cal_no")
                _sx_rows.append({
                    "Strike"     : f"${_strike:,.2f}",
                    "Hrs Left"   : f"{_d['_hrs']:.0f}h" if _d.get("_hrs") else "—",
                    "% to Strike": f"{_d['_pct']:+.1f}%" if _d.get("_pct") is not None else "—",
                    "YES Ask"    : f"{_d.get('yes_ask', '—')}¢" if _d.get("yes_ask") else "—",
                    "Cal P(YES)" : f"{_cal_yes*100:.1f}%" if _cal_yes is not None else "—",
                    "EV (YES)"   : f"{_d['ev_yes']:+.3f}" if _d.get("ev_yes") is not None else "—",
                    "NO Ask"     : f"{_d.get('no_ask', '—')}¢" if _d.get("no_ask") else "—",
                    "Cal P(NO)"  : f"{_cal_no*100:.1f}%"  if _cal_no  is not None else "—",
                    "EV (NO)"    : f"{_d['ev_no']:+.3f}"  if _d.get("ev_no")  is not None else "—",
                })

            if _sx_rows:
                st.dataframe(pd.DataFrame(_sx_rows), hide_index=True, use_container_width=True)
                st.caption(f"{len(_sx_rows)} strikes · {_sx_series}-{_sx_expiry} · ${_sx_lo:,.0f}–${_sx_hi:,.0f}")
            else:
                st.warning(f"No scored results found for {_sx_prefix} in that range. Check expiry code.")

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
        _cal_labels = {
            "vol"            : "<1hr (vol)",
            "intraday_short" : "1-8hr",
            "intraday_long"  : "8-24hr",
            "weekly"         : ">24hr",
            # legacy keys
            "intraday": "1-24hr (legacy)",
            "daily"   : ">24hr (legacy)",
        }
        _status_parts = []
        for _bn, _bd in _bkts.items():
            if not _bd.get("skipped"):
                _label = _cal_labels.get(_bn, _bn)
                _status_parts.append(f"**{_label}** {_bd['n_trades']} trades (actual {_bd['win_rate']:.0%} vs pred {_bd['pred_rate']:.0%})")
        if _status_parts:
            _updated = _cal_data.get("updated_at", "")[:10]
            st.caption(f"Active calibration (updated {_updated}): " + " · ".join(_status_parts))
    else:
        st.caption("No calibration active — click Recalibrate after accumulating 5+ settled paper trades per bucket.")

    _paper = db.load_paper_trades()

    with st.expander("🔍 Debug: Raw DB (last 10 trades)", expanded=False):
        _debug_rows = sorted(_paper, key=lambda x: x.get("placed_at", ""), reverse=True)[:10]
        st.json([{k: v for k, v in t.items()
                  if k in ("ticker","side","status","price_cents","close_time","placed_at","result","pnl_dollars")}
                 for t in _debug_rows])

    if not _paper:
        st.info("No paper trades recorded yet. Run the Kalshi Scan and click '📝 Paper Trade' to start tracking.")
    else:
        # ── Auto-settle expired trades ────────────────────────────────────────────
        _now_utc       = datetime.now(timezone.utc)
        _newly_settled = 0
        _open_paper    = []
        _settled_paper = []

        # Re-open any trades that were incorrectly settled with empty result
        # (Kalshi API returns result="" for open markets; "" is not None so the old
        #  guard passed, treating every live market as a loss.)
        for _pt in _paper:
            if _pt.get("status") == "settled" and not _pt.get("result"):
                db.reopen_paper_trade(_pt["id"])
                _pt["status"] = "open"
                _pt.pop("result", None)
                _pt.pop("pnl_dollars", None)

        # Correct P&L for settled trades where result was stored with wrong case
        # and P&L was computed using the loss formula instead of the win formula.
        for _pt in _paper:
            if _pt.get("status") == "settled" and _pt.get("result"):
                _result  = _pt["result"]
                _side    = _pt.get("side", "yes")
                _entry   = _pt.get("price_cents", 50)
                _ctrs    = _pt.get("contracts", 1)
                _correct = (round((100 - _entry) * _ctrs / 100, 2)
                            if _result.lower() == _side.lower()
                            else round(-_entry * _ctrs / 100, 2))
                if _pt.get("pnl_dollars") != _correct:
                    db.settle_paper_trade(_pt["id"], _result, _correct)
                    _pt["pnl_dollars"] = _correct

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
                        if _result:  # must be non-empty string ("yes"/"no"); "" means still open
                            _side  = _pt.get("side", "yes")
                            _entry = _pt.get("price_cents", 50)
                            _ctrs  = _pt.get("contracts", 1)
                            _pnl   = (round((100 - _entry) * _ctrs / 100, 2)
                                      if _result.lower() == _side.lower()
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

        def _time_bucket(hours):
            if hours is None: return "unknown"
            if hours < 1:  return "<1hr"
            if hours < 3:  return "1-3hr"
            if hours < 8:  return "3-8hr"
            if hours < 24: return "8-24hr"
            return ">24hr"

        # ── Open paper trades ─────────────────────────────────────────────────────
        if _open_paper:
            st.subheader(f"Open Paper Trades ({len(_open_paper)})")
            _open_paper = sorted(_open_paper, key=lambda x: x.get("placed_at", ""), reverse=True)
            _open_tickers_pt = tuple(p["ticker"] for p in _open_paper)
            _open_sides_pt   = tuple(p.get("side", "yes").lower() for p in _open_paper)
            _pt_live = fetch_live_prices(_open_tickers_pt, _open_sides_pt) if not _client.dry_run else {}

            _pt_rows = []
            for _pt in _open_paper:
                _entry = _pt.get("price_cents", 50)
                _ctrs  = _pt.get("contracts", 1)
                _bid   = _pt_live.get(_pt["ticker"])
                _hrs   = hours_left(_pt.get("close_time", ""), _pt.get("ticker", ""))
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
                    "Bucket"    : _time_bucket(_pt.get("hours_to_exp")),
                    "Entry ¢"   : _entry,
                    "Live Bid"  : f"{_bid}¢" if _bid is not None else "—",
                    "Contracts" : _ctrs,
                    "Bet $"     : f"${_pt.get('bet_dollars', 0):.0f}",
                    "Cal Prob"  : f"{(1-_pt.get('cal_prob',0) if _side.lower()=='no' else _pt.get('cal_prob',0))*100:.1f}%",
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
                        "Bucket"   : _time_bucket(_pt.get("hours_to_exp")),
                        "At Rec"   : (f"{int(_rec_h*60)}m" if _rec_h is not None and _rec_h < 1
                                      else f"{_rec_h:.0f}h" if _rec_h is not None else "—"),
                        "Contracts": _pt.get("contracts", 1),
                        "Entry ¢"  : _pt.get("price_cents", 0),
                        "Bet $"    : f"${_pt.get('bet_dollars', _pt.get('contracts', 1) * _pt.get('price_cents', 0) / 100):.0f}",
                        "Cal Prob" : f"{(1-_pt.get('cal_prob',0) if _pt.get('side','yes').lower()=='no' else _pt.get('cal_prob',0))*100:.1f}%",
                        "Result"   : _pt.get("result", "—"),
                        "P&L $"    : (f"${_pnl:+.2f}" if _pnl is not None else "—"),
                        "Placed"   : _pt.get("placed_at", "")[:10],
                    })
                st.dataframe(
                    pd.DataFrame(_s_rows).style.map(color_pnl, subset=["P&L $"]),
                    hide_index=True, use_container_width=True,
                )

        # ── Performance summary ───────────────────────────────────────────────────
        _resolved = [p for p in _settled_paper if p.get("pnl_dollars") is not None]

        def _win_prob(p):
            """Model's predicted probability of the side we BET ON winning."""
            cp = p.get("cal_prob", 0.5)
            return (1 - cp) if p.get("side", "yes").lower() == "no" else cp

        # Open trades summary by bucket
        if _open_paper:
            st.subheader("Open — By Bucket")
            _open_buckets = {}
            for _p in _open_paper:
                _b = _time_bucket(_p.get("hours_to_exp"))
                _open_buckets.setdefault(_b, []).append(_p)
            _ob_rows = []
            _bucket_order = ["<1hr", "1-3hr", "3-8hr", "8-24hr", ">24hr", "unknown"]
            for _b in _bucket_order:
                _bps = _open_buckets.get(_b)
                if not _bps:
                    continue
                _proj_wr = sum(_win_prob(p) for p in _bps) / len(_bps)
                _bet_tot = sum(p.get("bet_dollars", 0) for p in _bps)
                _ob_rows.append({
                    "Bucket"        : _b,
                    "Trades"        : len(_bps),
                    "Avg Win Prob"  : f"{_proj_wr:.0%}",
                    "Total Bet $"   : f"${_bet_tot:.0f}",
                })
            st.dataframe(pd.DataFrame(_ob_rows), hide_index=True, use_container_width=True)

        def _bet(p):
            return p.get("bet_dollars") or (p.get("contracts", 1) * p.get("price_cents", 0) / 100)

        def _bet_range(dollars):
            if dollars is None or dollars < 0:
                return "unknown"
            if dollars < 3:
                return "$0–3"
            if dollars < 8:
                return "$3–8"
            if dollars < 15:
                return "$8–15"
            return "$15+"

        def _bucket_stats(group_label, items):
            _wins_g  = [p for p in items if (p.get("pnl_dollars") or 0) > 0]
            _loss_g  = [p for p in items if (p.get("pnl_dollars") or 0) <= 0]
            _gpnl    = sum(p.get("pnl_dollars", 0) for p in items)
            _gbet    = sum(_bet(p) for p in items)
            _pnl_pct = (_gpnl / _gbet * 100) if _gbet > 0 else 0
            _avg_wp  = sum(_win_prob(p) for p in items) / len(items)
            _avg_w   = sum(p.get("pnl_dollars", 0) for p in _wins_g) / len(_wins_g) if _wins_g else 0
            _avg_l   = sum(p.get("pnl_dollars", 0) for p in _loss_g) / len(_loss_g) if _loss_g else 0
            return {
                group_label    : None,  # placeholder; caller overwrites key
                "Trades"       : len(items),
                "Total Bet $"  : f"${_gbet:.0f}",
                "Win Rate"     : f"{len(_wins_g)/len(items):.0%}",
                "Avg Win Prob" : f"{_avg_wp:.0%}",
                "Avg Win $"    : f"${_avg_w:+.2f}" if _wins_g else "—",
                "Avg Loss $"   : f"${_avg_l:+.2f}" if _loss_g else "—",
                "P&L %"        : f"{_pnl_pct:+.1f}%",
                "Total P&L"    : f"${_gpnl:+.2f}",
            }

        if _resolved:
            _color_cols = ["Avg Win $", "Avg Loss $", "P&L %", "Total P&L"]

            # ── By time bucket ────────────────────────────────────────────────
            st.subheader("Settled — By Time Bucket")
            _buckets = {}
            for _p in _resolved:
                _b = _time_bucket(_p.get("hours_to_exp"))
                _buckets.setdefault(_b, []).append(_p)

            _bk_rows = []
            _bucket_order = ["<1hr", "1-3hr", "3-8hr", "8-24hr", ">24hr", "unknown"]
            for _b in _bucket_order:
                _bps = _buckets.get(_b)
                if not _bps:
                    continue
                _row = _bucket_stats("Bucket", _bps)
                _row["Bucket"] = _b
                _bk_rows.append(_row)
            st.dataframe(
                pd.DataFrame(_bk_rows).style.map(color_pnl, subset=_color_cols),
                hide_index=True, use_container_width=True,
            )

            # ── By bet range ──────────────────────────────────────────────────
            st.subheader("Settled — By Bet Range (Kelly Size)")
            _ranges = {}
            for _p in _resolved:
                _r = _bet_range(_bet(_p))
                _ranges.setdefault(_r, []).append(_p)

            _br_rows = []
            _range_order = ["$0–3", "$3–8", "$8–15", "$15+", "unknown"]
            for _r in _range_order:
                _rps = _ranges.get(_r)
                if not _rps:
                    continue
                _row = _bucket_stats("Bet Range", _rps)
                _row["Bet Range"] = _r
                _br_rows.append(_row)
            st.dataframe(
                pd.DataFrame(_br_rows).style.map(color_pnl, subset=_color_cols),
                hide_index=True, use_container_width=True,
            )

            # Overall totals
            _wins      = sum(1 for p in _resolved if (p.get("pnl_dollars") or 0) > 0)
            _total_pnl = sum(p.get("pnl_dollars", 0) for p in _resolved)
            _total_bet = sum(_bet(p) for p in _resolved)
            _pnl_pct   = (_total_pnl / _total_bet * 100) if _total_bet > 0 else 0
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Total Settled", len(_resolved))
            _c2.metric("Win Rate", f"{_wins/len(_resolved):.0%}")
            _c3.metric("Return on Bets", f"{_pnl_pct:+.1f}%")
            _c4.metric("Total P&L", f"${_total_pnl:+.2f}")

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
    st.caption("Top long/short signals from the Random Forest model. "
               "Long: stock closes ≥2% higher in 5 days. Short: stock closes ≥2% lower.")

    # ── Upload local models to Supabase (one-time, after retraining) ──────────
    with st.expander("⬆️ Upload local models to cloud (run after retraining)"):
        st.caption("Use this after running features.py locally to push new/updated models to Supabase Storage so the hosted app can use them.")
        if st.button("Upload all local model files", key="upload_models"):
            _local_root = ROOT
            _files_to_upload = (
                list(_local_root.glob("model_*.joblib")) +
                list(_local_root.glob("features_*.csv")) +
                [_local_root / "ticker_summary.csv"]
            )
            _upload_prog = st.progress(0)
            _results = []
            for _i, _fp in enumerate(_files_to_upload):
                if _fp.exists():
                    _ok = db.upload_stock_file(_fp.name, _fp)
                    _results.append(("✅" if _ok else "❌", _fp.name))
                _upload_prog.progress((_i + 1) / len(_files_to_upload))
            _upload_prog.empty()
            _fails = [n for s, n in _results if s == "❌"]
            _ok_ct = sum(1 for s, _ in _results if s == "✅")
            if _fails:
                st.warning(f"Uploaded {_ok_ct} files. Failed: {', '.join(_fails)}")
            else:
                st.success(f"Uploaded {_ok_ct} files successfully. Refresh and run the scan.")

    stock_budget = st.number_input("Weekly stock budget ($)", min_value=100, max_value=100_000,
                                   value=2_000, step=100, key="stock_budget")

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

            summary_path = db.get_stock_file("ticker_summary.csv", ROOT)
            if summary_path is None:
                st.error("No ticker_summary.csv found.")
            else:
                summary    = pd.read_csv(summary_path)
                thresh_map = dict(zip(summary["Ticker"], summary["CV_Threshold"]))
                roc_map    = dict(zip(summary["Ticker"], summary["CV_ROC_AUC"]))
                _short_thresh_col  = summary["CV_Threshold_short"]  if "CV_Threshold_short"  in summary.columns else pd.Series([0.30] * len(summary))
                _short_roc_col     = summary["CV_ROC_AUC_short"]    if "CV_ROC_AUC_short"    in summary.columns else pd.Series([0.00] * len(summary))
                short_thresh_map   = dict(zip(summary["Ticker"], _short_thresh_col))
                short_roc_map      = dict(zip(summary["Ticker"], _short_roc_col))
                eligible   = [t for t in summary["Ticker"]
                              if roc_map.get(t, 0) >= pm.MIN_ROC_AUC]

                signals = []
                prog = st.progress(0, text="Scanning...")
                for i, ticker in enumerate(eligible):
                    prog.progress((i + 1) / len(eligible), text=f"Scanning {ticker}...")
                    # Long model
                    model_path    = db.get_stock_file(f"model_{ticker}.joblib", ROOT)
                    features_path = db.get_stock_file(f"features_{ticker}.csv", ROOT)
                    if model_path and features_path:
                        model         = joblib.load(model_path)
                        feature_names = pd.read_csv(features_path, header=None)[0].tolist()
                        threshold     = thresh_map.get(ticker, 0.30)
                        result        = pm.get_latest_signal(ticker, model, feature_names, threshold, "long")
                        if result:
                            signals.append(result)
                    # Short model
                    if short_roc_map.get(ticker, 0) >= pm.MIN_ROC_AUC:
                        short_model_path    = db.get_stock_file(f"model_{ticker}_short.joblib", ROOT)
                        short_features_path = db.get_stock_file(f"features_{ticker}_short.csv", ROOT)
                        if short_model_path and short_features_path:
                            short_model         = joblib.load(short_model_path)
                            short_feature_names = pd.read_csv(short_features_path, header=None)[0].tolist()
                            short_threshold     = short_thresh_map.get(ticker, 0.30)
                            short_result        = pm.get_latest_signal(ticker, short_model, short_feature_names, short_threshold, "short")
                            if short_result:
                                signals.append(short_result)
                prog.empty()

                long_signals  = sorted(
                    [s for s in signals if s.get("side", "long") == "long"  and s["signal"] and s["prob"] >= pm.MIN_PROB],
                    key=lambda x: x["prob"], reverse=True,
                )[:5]
                short_signals = sorted(
                    [s for s in signals if s.get("side") == "short" and s["signal"] and s["prob"] >= pm.MIN_PROB_SHORT],
                    key=lambda x: x["prob"], reverse=True,
                )[:5]
                buy_signals = sorted(long_signals + short_signals, key=lambda x: x["prob"], reverse=True)

                models_found = len(signals)
                if models_found == 0:
                    st.warning("No stock model files found on this server. "
                               "Stock models are trained locally and not deployed. "
                               "Run the scan locally with `streamlit run app.py`.")
                elif not buy_signals:
                    st.info(f"No signals today. ({models_found}/{len(eligible)*2} models scanned)")
                    st.session_state["stock_scan_signals"] = []
                else:
                    def _kelly_alloc(prob, budget):
                        f = max(0.0, 2 * prob - 1.0)
                        return round(f * budget, 0)

                    total_kelly = sum(_kelly_alloc(s["prob"], stock_budget) for s in buy_signals)
                    scale = min(1.0, stock_budget / total_kelly) if total_kelly > 0 else 1.0

                    # Save to session state so paper trade buttons survive the rerun
                    st.session_state["stock_scan_signals"] = [
                        {**s, "alloc": _kelly_alloc(s["prob"], stock_budget) * scale,
                         "shares": int(_kelly_alloc(s["prob"], stock_budget) * scale / s["close"])
                                   if s["close"] > 0 else 0}
                        for s in buy_signals
                    ]
                    st.session_state["stock_scan_eligible"] = len(eligible)
                    st.session_state["stock_scan_scored"]   = models_found

        except Exception as e:
            st.error(f"Stock scan error: {e}")
            import traceback; st.code(traceback.format_exc())

    # Render scan results + paper trade buttons from session state (persists across reruns)
    if "stock_scan_signals" in st.session_state and st.session_state["stock_scan_signals"]:
        _ss = st.session_state["stock_scan_signals"]
        _any_intraday = any(s.get("intraday") for s in _ss)
        scan_rows = []
        for s in _ss:
            _cur = s.get("current_price")
            _chg = ((_cur - s["close"]) / s["close"] * 100) if _cur and not s.get("intraday") else None
            _price_label = "Today's Price" if s.get("intraday") else "Signal Close"
            _side = s.get("side", "long")
            row = {
                "Ticker"      : s["ticker"],
                "Side"        : "SHORT" if _side == "short" else "LONG",
                "Suggested $" : f"${s['alloc']:,.0f}",
                "~Shares"     : s["shares"] if s["shares"] > 0 else "<1",
                _price_label  : f"${s['close']:.2f}",
                "Prob"        : f"{s['prob']*100:.1f}%",
                "Edge"        : f"+{(s['prob'] - s['threshold'])*100:.1f}pp",
            }
            if not s.get("intraday"):
                row["Now"]  = f"${_cur:.2f}" if _cur else "—"
                row["Move"] = f"{_chg:+.1f}%" if _chg is not None else "—"
            scan_rows.append(row)
        st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)
        if _any_intraday:
            st.info("Signal computed from today's intraday bar (after 3:30 PM ET) — place MOC order before close.")
        st.caption(
            f"Scanned {st.session_state.get('stock_scan_eligible','?')} tickers · "
            f"{st.session_state.get('stock_scan_scored','?')} models scored · "
            f"Prob = chance of ±2% move in 5 days · "
            f"Edge = prob minus per-ticker signal threshold · "
            f"Suggested $ uses fractional Kelly (±2% target/stop, min prob 60%)"
        )
        st.write("")
        for s in _ss:
            _side_label = "Short" if s.get("side") == "short" else "Long"
            if st.button(f"📝 Paper Trade {s['ticker']} ({_side_label})", key=f"pt_stock_{s['ticker']}_{s.get('side','long')}"):
                import uuid, traceback as _tb
                try:
                    _live = yf.Ticker(s["ticker"]).fast_info
                    _live_price = round(float(_live["lastPrice"]), 2)
                except Exception:
                    _live_price = s["close"]
                _trade = {
                    "id"         : str(uuid.uuid4()),
                    "ticker"     : s["ticker"],
                    "side"       : s.get("side", "long"),
                    "entry_price": _live_price,
                    "entry_date" : date.today().isoformat(),
                    "shares"     : s["shares"],
                    "dollars"    : round(s["shares"] * _live_price, 2),
                    "model_prob" : s["prob"],
                    "status"     : "open",
                    "exit_reason": None,
                    "exit_price" : None,
                    "exit_date"  : None,
                    "pnl_dollars": None,
                    "pnl_pct"    : None,
                    "placed_at"  : datetime.now(timezone.utc).isoformat(),
                }
                try:
                    db.add_stock_paper_trade(_trade)
                    st.success(f"Recorded {_side_label}: {s['ticker']} @ ${s['close']:.2f} × {s['shares']} shares (${s['alloc']:.0f})")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Failed to save paper trade: {_e}")
                    st.code(_tb.format_exc())

    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("Stocks — Paper Trade Tracker")
    st.caption("Tracks model-suggested entries. Auto-settles at +2% target, −2% stop, or after 5 trading days.")

    _stock_paper = db.load_stock_paper_trades()
    _open_sp   = [t for t in _stock_paper if t.get("status") == "open"]
    _closed_sp = [t for t in _stock_paper if t.get("status") == "closed"]
    if not _stock_paper:
        st.info("No stock paper trades yet. Run the scan and click '📝 Paper Trade' to start tracking.")
    else:
        # ── Auto-settle open trades ──────────────────────────────────────────
        if _open_sp:
            import yfinance as yf
            _tickers_to_check = list({t["ticker"] for t in _open_sp})
            try:
                _price_data = yf.download(_tickers_to_check, period="10d",
                                          auto_adjust=True, progress=False)
                if isinstance(_price_data.columns, pd.MultiIndex):
                    _closes = _price_data["Close"]
                else:
                    _closes = _price_data[["Close"]]
                    _closes.columns = _tickers_to_check
            except Exception:
                _closes = pd.DataFrame()

            _newly_closed = 0
            _today_str = datetime.now(timezone.utc).date().isoformat()

            for _sp in _open_sp:
                _tk    = _sp["ticker"]
                _ep    = float(_sp["entry_price"])
                _entry_dt = _sp.get("entry_date", "")
                try:
                    _entry_d = date.fromisoformat(_entry_dt)
                except Exception:
                    continue

                # Count trading days held
                _days_held = sum(
                    1 for _d in pd.bdate_range(_entry_d, date.today())
                    if _d.date() > _entry_d
                )

                # Get latest close
                try:
                    _cur_close = float(_closes[_tk].dropna().iloc[-1])
                except Exception:
                    continue

                _side = _sp.get("side", "long")
                if _side == "short":
                    _pnl_pct     = (_ep - _cur_close) / _ep
                    _pnl_dollars = round((_ep - _cur_close) * float(_sp.get("shares", 0)), 2)
                else:
                    _pnl_pct     = (_cur_close - _ep) / _ep
                    _pnl_dollars = round((_cur_close - _ep) * float(_sp.get("shares", 0)), 2)
                _exit_date = _closes[_tk].dropna().index[-1].date().isoformat()

                if _pnl_pct >= 0.02:
                    _reason = "target"
                elif _pnl_pct <= -0.02:
                    _reason = "stop"
                elif _days_held >= 5:
                    _reason = "time"
                else:
                    _reason = None

                if _reason:
                    db.close_stock_paper_trade(
                        _sp["id"], round(_cur_close, 4), _exit_date,
                        _reason, _pnl_dollars, round(_pnl_pct * 100, 2)
                    )
                    _sp.update({
                        "status": "closed", "exit_reason": _reason,
                        "exit_price": round(_cur_close, 4), "exit_date": _exit_date,
                        "pnl_dollars": _pnl_dollars, "pnl_pct": round(_pnl_pct * 100, 2),
                    })
                    _newly_closed += 1

            if _newly_closed:
                st.success(f"Auto-settled {_newly_closed} stock paper trade(s).")
                _open_sp   = [t for t in _stock_paper if t.get("status") == "open"]
                _closed_sp = [t for t in _stock_paper if t.get("status") == "closed"]

        # ── Open trades ──────────────────────────────────────────────────────
        if _open_sp:
            st.subheader(f"Open ({len(_open_sp)})")
            # Build rows keyed by id so we can match edits back to DB records
            _cur_prices = {}
            for _sp in _open_sp:
                try:
                    _cur_prices[_sp["ticker"]] = float(_closes[_sp["ticker"]].dropna().iloc[-1])
                except Exception:
                    _cur_prices[_sp["ticker"]] = None

            # Only rebuild from DB when trade list changes (new trade added / settled)
            _sp_ids = tuple(t["id"] for t in _open_sp)
            if st.session_state.get("sp_open_ids") != _sp_ids:
                st.session_state["sp_open_ids"] = _sp_ids
                _sp_open_rows = []
                for _sp in _open_sp:
                    _ep = float(_sp["entry_price"])
                    _cur = _cur_prices.get(_sp["ticker"])
                    try:
                        _entry_d = date.fromisoformat(_sp.get("entry_date", ""))
                        _days = sum(1 for _d in pd.bdate_range(_entry_d, date.today()) if _d.date() > _entry_d)
                    except Exception:
                        _days = 0
                    _placed = _sp.get("placed_at", "")
                    try:
                        _placed_dt = datetime.fromisoformat(_placed.replace("Z", "+00:00"))
                        from zoneinfo import ZoneInfo
                        _placed_et = _placed_dt.astimezone(ZoneInfo("America/New_York"))
                        _placed_str = _placed_et.strftime("%m/%d %I:%M %p ET")
                    except Exception:
                        _placed_str = _placed[:16] if _placed else "—"
                    _sp_open_rows.append({
                        "_id"       : _sp["id"],
                        "Side"      : "SHORT" if _sp.get("side") == "short" else "LONG",
                        "Ticker"    : _sp["ticker"],
                        "Entry $"   : _ep,
                        "Shares"    : float(_sp.get("shares") or 0),
                        "Entry Date": _sp.get("entry_date", ""),
                        "Entered"   : _placed_str,
                        "Cur Price" : round(_cur, 2) if _cur else None,
                        "Days Held" : _days,
                        "Prob"      : f"{_sp.get('model_prob', 0)*100:.1f}%",
                    })
                st.session_state["sp_open_rows"] = _sp_open_rows
            else:
                _sp_open_rows = st.session_state["sp_open_rows"]

            # Editable table: ONLY static fields (no live data — prevents resets)
            _edit_df = pd.DataFrame([{
                "_id"    : r["_id"],
                "Side"   : r["Side"],
                "Ticker" : r["Ticker"],
                "Entry $": r["Entry $"],
                "Shares" : r["Shares"],
            } for r in _sp_open_rows])
            _edited_sp = st.data_editor(
                _edit_df,
                column_config={
                    "_id"    : None,
                    "Side"   : st.column_config.TextColumn(disabled=True),
                    "Ticker" : st.column_config.TextColumn(disabled=True),
                    "Entry $": st.column_config.NumberColumn("Entry $", format="$%.2f", min_value=0.0, step=0.01),
                    "Shares" : st.column_config.NumberColumn("Shares", min_value=0.0, step=0.01, format="%.2f"),
                },
                hide_index=True, use_container_width=True, key="sp_open_editor",
            )
            if st.button("💾 Save Changes", key="sp_save_btn"):
                try:
                    for i, row in _edited_sp.iterrows():
                        _ep2 = float(row["Entry $"] or 0)
                        _sh2 = float(row["Shares"] or 0)
                        _row_id = _sp_open_rows[i]["_id"]
                        resp = db._get_client().table("stock_paper_trades").update({
                            "entry_price": _ep2,
                            "shares"     : _sh2,
                            "dollars"    : round(_ep2 * _sh2, 2),
                        }).eq("id", _row_id).execute()
                    st.session_state.pop("sp_open_ids", None)
                    st.success("Saved.")
                    st.rerun()
                except Exception as _save_err:
                    st.error(f"Save failed: {_save_err}")

            # Read-only summary with live prices and computed P&L
            _summary_rows = []
            for i, row in _edited_sp.iterrows():
                _ep2 = float(row["Entry $"] or 0)
                _sh2 = float(row["Shares"] or 0)
                _cur2 = _cur_prices.get(row["Ticker"])
                _inv  = round(_ep2 * _sh2, 2)
                _orig = _sp_open_rows[i]
                _is_short = _orig.get("Side") == "SHORT"
                if _cur2 and _ep2 > 0:
                    _pnl_pct = ((_ep2 - _cur2) / _ep2 * 100) if _is_short else ((_cur2 - _ep2) / _ep2 * 100)
                    _pnl_d   = round(((_ep2 - _cur2) if _is_short else (_cur2 - _ep2)) * _sh2, 2)
                else:
                    _pnl_pct = 0.0
                    _pnl_d   = 0.0
                _summary_rows.append({
                    "Side"      : _orig["Side"],
                    "Ticker"    : row["Ticker"],
                    "Entered"   : _orig["Entered"],
                    "Entry $"   : f"${_ep2:.2f}",
                    "Shares"    : f"{_sh2:.2f}",
                    "Invested"  : f"${_inv:.2f}",
                    "Cur Price" : f"${_cur2:.2f}" if _cur2 else "—",
                    "Days Held" : _orig["Days Held"],
                    "P&L %"     : f"{_pnl_pct:+.1f}%",
                    "P&L $"     : f"${_pnl_d:+.2f}",
                    "Prob"      : _orig["Prob"],
                })
            st.dataframe(
                pd.DataFrame(_summary_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                hide_index=True, use_container_width=True,
            )

        # ── Closed trades ────────────────────────────────────────────────────
        if _closed_sp:
            _reason_labels = {"target": "✓ Target +2%", "stop": "✗ Stop −2%", "time": "⏱ Time exit"}
            with st.expander(f"Closed ({len(_closed_sp)})"):
                _sp_closed_rows = []
                for _sp in sorted(_closed_sp, key=lambda x: x.get("exit_date", ""), reverse=True):
                    _pnl = _sp.get("pnl_dollars")
                    _sp_closed_rows.append({
                        "Side"      : "SHORT" if _sp.get("side") == "short" else "LONG",
                        "Ticker"    : _sp["ticker"],
                        "Entry $"   : f"${float(_sp['entry_price']):.2f}",
                        "Exit $"    : f"${float(_sp['exit_price']):.2f}" if _sp.get("exit_price") else "—",
                        "Shares"    : _sp.get("shares", 0),
                        "Exit"      : _reason_labels.get(_sp.get("exit_reason", ""), "—"),
                        "P&L %"     : f"{_sp.get('pnl_pct', 0):+.1f}%",
                        "P&L $"     : f"${_pnl:+.2f}" if _pnl is not None else "—",
                        "Entry Date": _sp.get("entry_date", ""),
                        "Exit Date" : _sp.get("exit_date", ""),
                        "Prob"      : f"{_sp.get('model_prob', 0)*100:.1f}%",
                    })
                st.dataframe(
                    pd.DataFrame(_sp_closed_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                    hide_index=True, use_container_width=True,
                )

            # ── Summary metrics ──────────────────────────────────────────────
            _with_pnl = [t for t in _closed_sp if t.get("pnl_dollars") is not None]
            if _with_pnl:
                _wins     = sum(1 for t in _with_pnl if t.get("pnl_pct", 0) > 0)
                _total_pnl = sum(t["pnl_dollars"] for t in _with_pnl)
                _avg_pnl  = sum(t.get("pnl_pct", 0) for t in _with_pnl) / len(_with_pnl)
                _by_reason = {}
                for t in _with_pnl:
                    r = t.get("exit_reason", "time")
                    _by_reason.setdefault(r, []).append(t.get("pnl_pct", 0))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", len(_with_pnl))
                c2.metric("Win Rate", f"{_wins/len(_with_pnl):.0%}")
                c3.metric("Avg P&L", f"{_avg_pnl:+.1f}%")
                c4.metric("Total P&L", f"${_total_pnl:+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM L/S TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_lt:
    st.header("Long-Term L/S Portfolio")
    st.caption(
        "Factor-scored (momentum · quality · value). "
        "Top 5 → LONG, Bottom 5 → SHORT. "
        "Portfolio split 50/50 long-short, equal-weighted per side. "
        "Reassessment-based exits — no fixed hold period, re-score monthly."
    )

    # ── Load long-term module lazily ─────────────────────────────────────────
    _lt_mod = None
    try:
        _lt_spec = importlib.util.spec_from_file_location(
            "longterm", ROOT / "stocks" / "longterm.py"
        )
        _lt_mod = importlib.util.module_from_spec(_lt_spec)
        _lt_spec.loader.exec_module(_lt_mod)
    except Exception as _lt_err:
        st.warning(f"Could not load longterm module: {_lt_err}")

    if _lt_mod:
        # ── Open long-term positions ──────────────────────────────────────────
        _lt_positions = _lt_mod.load_lt_positions()
        _lt_open   = [p for p in _lt_positions if p.get("status") == "open"]
        _lt_closed = [p for p in _lt_positions if p.get("status") == "closed"]

        if _lt_open:
            # Batch-fetch live prices for all open positions
            import yfinance as _yf
            _lt_tks = [p["ticker"] for p in _lt_open]
            try:
                _lt_live_df = _yf.download(
                    _lt_tks, period="5d", auto_adjust=True, progress=False, group_by="ticker"
                )
                def _lt_live_price(tk):
                    try:
                        if len(_lt_tks) == 1:
                            return round(float(_lt_live_df["Close"].dropna().iloc[-1]), 2)
                        return round(float(_lt_live_df[tk]["Close"].dropna().iloc[-1]), 2)
                    except Exception:
                        return None
            except Exception:
                def _lt_live_price(tk): return None

            _lt_rows = []
            _lt_pnl_dollars = []
            _lt_costs = []
            for _lp in _lt_open:
                _ep    = _lp["entry_price"]
                _cur   = _lt_live_price(_lp["ticker"]) or _ep
                _dir   = _lp["direction"]
                _pnl   = ((_cur - _ep) / _ep * 100) if _dir == "LONG" else ((_ep - _cur) / _ep * 100)
                _shares_held = _lp.get("shares", 0) or 0
                _cost  = _lp.get("cost", 0) or (_shares_held * _ep)
                # Fallback: if cost still 0 (old positions with int-truncated shares),
                # estimate from entry price × shares; if shares also 0, use $500 allocation
                if _cost == 0:
                    _cost = 500.0
                _pnl_d = _cost * _pnl / 100
                _score = _lp.get("current_score")
                _days  = (date.today() - date.fromisoformat(_lp["entry_date"])).days
                _flag  = ""
                if _lp.get("exit_signal") == "HARD_STOP":
                    _flag = "🚨 HARD STOP"
                elif _lp.get("reassess_signal"):
                    _flag = f"⚠️ {_lp['reassess_signal']}"
                _lt_rows.append({
                    "Ticker"     : _lp["ticker"],
                    "Dir"        : _dir,
                    "Shares"     : f"{_shares_held:.4f}" if _shares_held else "—",
                    "Invested $" : f"${_cost:,.2f}",
                    "Entry $"    : f"${_ep:.2f}",
                    "Cur $"      : f"${_cur:.2f}",
                    "Days"       : _days,
                    "P&L %"      : f"{_pnl:+.2f}%",
                    "P&L $"      : f"${_pnl_d:+.2f}",
                    "Score"      : f"{_score:.3f}" if _score is not None else "—",
                    "Status"     : _flag or "✓ Hold",
                })
                _lt_pnl_dollars.append(_pnl_d)
                _lt_costs.append(_cost)

            st.dataframe(
                pd.DataFrame(_lt_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                hide_index=True, use_container_width=True,
            )

            # ── Portfolio summary metrics ──────────────────────────────────────
            _total_cost    = sum(_lt_costs)
            _total_pnl_d   = sum(_lt_pnl_dollars)
            _weighted_pnl  = (_total_pnl_d / _total_cost * 100) if _total_cost else 0
            _longs  = [r for r in _lt_rows if r["Dir"] == "LONG"]
            _shorts = [r for r in _lt_rows if r["Dir"] == "SHORT"]
            _long_pnl  = sum(_lt_pnl_dollars[i] for i, r in enumerate(_lt_rows) if r["Dir"] == "LONG")
            _short_pnl = sum(_lt_pnl_dollars[i] for i, r in enumerate(_lt_rows) if r["Dir"] == "SHORT")

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric("Total P&L $",       f"${_total_pnl_d:+.2f}")
            _mc2.metric("Weighted P&L %",    f"{_weighted_pnl:+.2f}%")
            _mc3.metric("Long P&L $",        f"${_long_pnl:+.2f}")
            _mc4.metric("Short P&L $",       f"${_short_pnl:+.2f}")
        else:
            st.info("No open long-term positions. Run a scan to get recommendations.")

        # ── Scan controls ─────────────────────────────────────────────────────
        _lt_budget = st.number_input(
            "Portfolio size ($)", min_value=1_000, max_value=1_000_000,
            value=5_000, step=500, key="lt_budget",
        )

        if st.button("Run Long-Term Scan", type="primary", key="lt_scan"):
            _summary_path = db.get_stock_file("ticker_summary.csv", ROOT)
            if _summary_path is None and (ROOT / "stocks" / "ticker_summary.csv").exists():
                _summary_path = ROOT / "stocks" / "ticker_summary.csv"
            if _summary_path is None and (ROOT / "ticker_summary.csv").exists():
                _summary_path = ROOT / "ticker_summary.csv"

            if _summary_path is None:
                st.error("ticker_summary.csv not found — run the stock model training first.")
            else:
                _lt_tickers = pd.read_csv(_summary_path)["Ticker"].tolist()
                _lt_progress = st.progress(0, text="Fetching factor data…")

                def _lt_cb(i, n, t):
                    _lt_progress.progress((i + 1) / n, text=f"Scoring {t} ({i+1}/{n})")

                try:
                    _lt_df = _lt_mod.run_lt_scan(_lt_tickers, _lt_cb)
                    _lt_progress.empty()
                    st.session_state["lt_scan_result"] = _lt_df
                    st.session_state["lt_scan_budget"]  = _lt_budget
                except Exception as _lt_scan_err:
                    _lt_progress.empty()
                    st.error(f"Long-term scan error: {_lt_scan_err}")

        # ── Display scan results ──────────────────────────────────────────────
        if "lt_scan_result" in st.session_state:
            _lt_df = st.session_state["lt_scan_result"]
            _lt_budget_used = st.session_state.get("lt_scan_budget", _lt_budget)

            if not _lt_df.empty:
                # Show full ranked table in expander
                with st.expander("Full ranked universe", expanded=False):
                    _display_cols = ["rank", "ticker", "direction", "composite_score",
                                     "mom_12_1", "mom_1m", "roe", "gross_margin",
                                     "fwd_pe", "pb", "current_price"]
                    _display_cols = [c for c in _display_cols if c in _lt_df.columns]
                    _lt_display = _lt_df[_display_cols].copy()
                    for _pct_col in ["mom_12_1", "mom_1m", "roe", "gross_margin"]:
                        if _pct_col in _lt_display.columns:
                            _lt_display[_pct_col] = _lt_display[_pct_col].apply(
                                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—"
                            )
                    for _mul_col in ["fwd_pe", "pb"]:
                        if _mul_col in _lt_display.columns:
                            _lt_display[_mul_col] = _lt_display[_mul_col].apply(
                                lambda x: f"{x:.1f}x" if pd.notna(x) else "—"
                            )
                    st.dataframe(_lt_display, hide_index=True, use_container_width=True)

                # LONG recommendations
                _lt_longs  = _lt_df[_lt_df["direction"] == "LONG"]
                _lt_shorts = _lt_df[_lt_df["direction"] == "SHORT"]

                st.subheader("LONG Recommendations")
                _existing_lt_tickers = {p["ticker"] for p in _lt_open}
                _open_by_ticker      = {p["ticker"]: p for p in _lt_open}
                _alloc_per_side = _lt_budget_used / 2
                _per_long = _alloc_per_side / max(len(_lt_longs), 1)

                for _, _lr in _lt_longs.iterrows():
                    _tk = _lr["ticker"]
                    _already = _tk in _existing_lt_tickers
                    _shares = round(_per_long / _lr["current_price"], 4) if _lr["current_price"] else 0
                    _col1, _col2 = st.columns([3, 1])
                    with _col1:
                        _mom_str = f"{_lr['mom_12_1']*100:+.1f}%" if pd.notna(_lr.get("mom_12_1")) else "—"
                        _m1_str  = f"{_lr['mom_1m']*100:+.1f}%"  if pd.notna(_lr.get("mom_1m"))   else "—"
                        _roe_str = f"{_lr['roe']*100:.1f}%"        if pd.notna(_lr.get("roe"))      else "—"
                        st.markdown(
                            f"**{_tk}** · Score {_lr['composite_score']:.3f} · "
                            f"${_lr['current_price']} · "
                            f"**Invest: ${_per_long:,.2f}** ({_shares} shares) · "
                            f"Mom(12-1): {_mom_str} · 1m: {_m1_str} · ROE: {_roe_str}"
                        )
                    with _col2:
                        if _already:
                            _held = _open_by_ticker[_tk]
                            _entry_score   = _held.get("composite_score")
                            _current_score = _lr["composite_score"]
                            if _entry_score is not None:
                                _score_delta = _current_score - _entry_score
                                if _score_delta >= 0.10:
                                    st.success(f"Add more (score +{_score_delta:.2f} vs entry)")
                                elif _score_delta <= -0.10:
                                    st.warning(f"Hold / watch (score {_score_delta:.2f} vs entry)")
                                else:
                                    st.info(f"Hold — thesis intact (score {_score_delta:+.2f} vs entry)")
                            else:
                                st.caption("Already held")
                        elif st.button(f"📈 Record Long {_tk}", key=f"lt_long_{_tk}"):
                            _new_lt = {
                                "ticker"      : _tk,
                                "direction"   : "LONG",
                                "entry_price" : _lr["current_price"],
                                "entry_date"  : str(date.today()),
                                "shares"      : _shares,
                                "cost"        : round(_shares * _lr["current_price"], 2),
                                "status"      : "open",
                                "composite_score": float(_lr["composite_score"]),
                                "pnl_pct"     : 0.0,
                                "days_held"   : 0,
                                "exit_signal" : None,
                                "reassess_signal": None,
                            }
                            _lt_positions.append(_new_lt)
                            _lt_mod.save_lt_positions(_lt_positions)
                            st.success(f"Recorded LONG {_tk} @ ${_lr['current_price']}")
                            st.rerun()

                st.subheader("SHORT Recommendations")
                _per_short = _alloc_per_side / max(len(_lt_shorts), 1)
                for _, _sr in _lt_shorts.iterrows():
                    _tk = _sr["ticker"]
                    _already = _tk in _existing_lt_tickers
                    _shares = round(_per_short / _sr["current_price"], 4) if _sr["current_price"] else 0
                    _col1, _col2 = st.columns([3, 1])
                    with _col1:
                        _mom_str = f"{_sr['mom_12_1']*100:+.1f}%" if pd.notna(_sr.get("mom_12_1")) else "—"
                        _m1_str  = f"{_sr['mom_1m']*100:+.1f}%"  if pd.notna(_sr.get("mom_1m"))   else "—"
                        st.markdown(
                            f"**{_tk}** · Score {_sr['composite_score']:.3f} · "
                            f"${_sr['current_price']} · "
                            f"**Invest: ${_per_short:,.2f}** ({_shares} shares) · "
                            f"Mom(12-1): {_mom_str} · 1m: {_m1_str}"
                        )
                    with _col2:
                        if _already:
                            st.caption("Already held")
                        elif st.button(f"📉 Record Short {_tk}", key=f"lt_short_{_tk}"):
                            _new_lt = {
                                "ticker"      : _tk,
                                "direction"   : "SHORT",
                                "entry_price" : _sr["current_price"],
                                "entry_date"  : str(date.today()),
                                "shares"      : _shares,
                                "cost"        : round(_shares * _sr["current_price"], 2),
                                "status"      : "open",
                                "composite_score": float(_sr["composite_score"]),
                                "pnl_pct"     : 0.0,
                                "days_held"   : 0,
                                "exit_signal" : None,
                                "reassess_signal": None,
                            }
                            _lt_positions.append(_new_lt)
                            _lt_mod.save_lt_positions(_lt_positions)
                            st.success(f"Recorded SHORT {_tk} @ ${_sr['current_price']}")
                            st.rerun()

        # ── Reassess open positions ───────────────────────────────────────────
        if _lt_open:
            _reassess_col, _score_all_col = st.columns(2)

            with _reassess_col:
                if "lt_scan_result" in st.session_state:
                    if st.button("Reassess Open Positions", key="lt_reassess"):
                        _lt_updated = _lt_mod.assess_open_positions(
                            _lt_open, st.session_state["lt_scan_result"]
                        )
                        # Merge back into full positions list
                        _lt_by_ticker = {p["ticker"]: p for p in _lt_updated}
                        _lt_positions = [
                            _lt_by_ticker.get(p["ticker"], p) if p.get("status") == "open" else p
                            for p in _lt_positions
                        ]
                        _lt_mod.save_lt_positions(_lt_positions)
                        st.success("Positions reassessed.")
                        st.rerun()

            with _score_all_col:
                if st.button("Score All Open Positions", key="lt_score_all"):
                    _lt_open_tickers = [p["ticker"] for p in _lt_open]
                    try:
                        _lt_cb = lambda p: st.progress(p, text="Scoring positions...")
                        _lt_all_scored = _lt_mod.run_lt_scan(_lt_open_tickers, _lt_cb)
                        _lt_updated = _lt_mod.assess_open_positions(_lt_open, _lt_all_scored)
                        # Merge back into full positions list
                        _lt_by_ticker = {p["ticker"]: p for p in _lt_updated}
                        _lt_positions = [
                            _lt_by_ticker.get(p["ticker"], p) if p.get("status") == "open" else p
                            for p in _lt_positions
                        ]
                        _lt_mod.save_lt_positions(_lt_positions)
                        st.success(f"Scored {len(_lt_open_tickers)} open position(s).")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Score error: {_e}")

        # ── Close a position manually ─────────────────────────────────────────
        if _lt_open:
            with st.expander("Close a position"):
                _lt_close_ticker = st.selectbox(
                    "Ticker", [p["ticker"] for p in _lt_open], key="lt_close_sel"
                )
                _lt_close_price = st.number_input(
                    "Exit price ($)", min_value=0.01, value=100.0, key="lt_close_price"
                )
                _lt_close_reason = st.selectbox(
                    "Reason", ["reassessment", "hard_stop", "manual"], key="lt_close_reason"
                )
                if st.button("Close Position", key="lt_close_btn"):
                    for _lp in _lt_positions:
                        if _lp["ticker"] == _lt_close_ticker and _lp["status"] == "open":
                            _ep = _lp["entry_price"]
                            if _lp["direction"] == "LONG":
                                _pnl_pct = (_lt_close_price - _ep) / _ep * 100
                                _pnl_d   = (_lt_close_price - _ep) * _lp.get("shares", 0)
                            else:
                                _pnl_pct = (_ep - _lt_close_price) / _ep * 100
                                _pnl_d   = (_ep - _lt_close_price) * _lp.get("shares", 0)
                            _lp.update({
                                "status"      : "closed",
                                "exit_price"  : _lt_close_price,
                                "exit_date"   : str(date.today()),
                                "exit_reason" : _lt_close_reason,
                                "pnl_pct"     : round(_pnl_pct, 2),
                                "pnl_dollars" : round(_pnl_d, 2),
                            })
                            break
                    _lt_mod.save_lt_positions(_lt_positions)
                    st.success(f"Closed {_lt_close_ticker}.")
                    st.rerun()

        # ── Closed positions ──────────────────────────────────────────────────
        if _lt_closed:
            with st.expander(f"Closed long-term positions ({len(_lt_closed)})"):
                _lt_c_rows = []
                for _lp in sorted(_lt_closed, key=lambda x: x.get("exit_date", ""), reverse=True):
                    _pnl = _lp.get("pnl_pct", 0)
                    _lt_c_rows.append({
                        "Ticker"    : _lp["ticker"],
                        "Dir"       : _lp["direction"],
                        "Entry $"   : f"${_lp['entry_price']:.2f}",
                        "Exit $"    : f"${_lp.get('exit_price', 0):.2f}",
                        "Entry Date": _lp.get("entry_date", ""),
                        "Exit Date" : _lp.get("exit_date", ""),
                        "Days"      : _lp.get("days_held", "—"),
                        "Reason"    : _lp.get("exit_reason", "—"),
                        "P&L %"     : f"{_pnl:+.2f}%",
                        "P&L $"     : f"${_lp.get('pnl_dollars', 0):+.2f}",
                    })
                st.dataframe(
                    pd.DataFrame(_lt_c_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                    hide_index=True, use_container_width=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# CONVICTION L/S TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_conviction:
    st.header("Conviction L/S Portfolio")
    st.caption(
        "Four-pillar composite score: Technical (25%) · Fundamental (35%) · "
        "Earnings (25%) · Macro (15%). "
        "Expected hold 30–90 days. Positions only opened when score clears "
        "conviction threshold — fewer than 10 trades may be recommended. "
        "Hard stop 15%. Monthly reassessment."
    )

    # ── Load conviction module lazily ─────────────────────────────────────────
    _cv_mod = None
    try:
        _cv_spec = importlib.util.spec_from_file_location(
            "conviction", ROOT / "stocks" / "conviction.py"
        )
        _cv_mod = importlib.util.module_from_spec(_cv_spec)
        _cv_spec.loader.exec_module(_cv_mod)
    except Exception as _cv_err:
        st.warning(f"Could not load conviction module: {_cv_err}")

    if _cv_mod:
        # ── Open conviction positions ─────────────────────────────────────────
        _cv_positions = _cv_mod.load_conviction_positions()
        _cv_open      = [p for p in _cv_positions if p.get("status") == "open"]
        _cv_closed    = [p for p in _cv_positions if p.get("status") == "closed"]

        if _cv_open:
            import yfinance as _yf2
            _cv_tks = [p["ticker"] for p in _cv_open]
            try:
                _cv_live_df = _yf2.download(
                    _cv_tks, period="5d", auto_adjust=True, progress=False, group_by="ticker"
                )
                def _cv_live_price(tk):
                    try:
                        if len(_cv_tks) == 1:
                            return round(float(_cv_live_df["Close"].dropna().iloc[-1]), 2)
                        return round(float(_cv_live_df[tk]["Close"].dropna().iloc[-1]), 2)
                    except Exception:
                        return None
            except Exception:
                def _cv_live_price(tk): return None

            _cv_rows        = []
            _cv_pnl_dollars = []
            _cv_costs       = []
            for _cp in _cv_open:
                _ep    = _cp["entry_price"]
                _cur   = _cv_live_price(_cp["ticker"]) or _ep
                _dir   = _cp["direction"]
                _pnl   = ((_cur - _ep) / _ep * 100) if _dir == "LONG" else ((_ep - _cur) / _ep * 100)
                _shares_held = _cp.get("shares", 0) or 0
                _cost  = _cp.get("cost", _ep * _shares_held)
                _pnl_d = _cost * _pnl / 100
                _score = _cp.get("current_score")
                if _score is None:
                    _score = _cp.get("score_at_entry")
                _edays = _cp.get("earnings_days_out")
                _eflag = _cp.get("earnings_flag", False)
                # Build status — priority: hard stop > reassess > hold
                # Earnings note appended for all positions that have the data
                if _cp.get("exit_signal"):
                    _flag = "🚨 HARD STOP"
                elif _cp.get("reassess_signal"):
                    _flag = f"⚠️ {_cp['reassess_signal']}"
                else:
                    _flag = "✓ Hold"
                # Always append earnings countdown when we know the date
                if _edays is not None and _edays >= 0:
                    _earn_icon = "🚨" if _edays <= 7 else "📅"
                    _flag += f"  {_earn_icon} Earn {_edays}d"
                _cv_rows.append({
                    "Ticker"    : _cp["ticker"],
                    "Dir"       : _dir,
                    "Shares"    : f"{_shares_held:.4f}" if _shares_held else "—",
                    "Entry $"   : f"${_ep:.2f}",
                    "Current $" : f"${_cur:.2f}",
                    "P&L %"     : f"{_pnl:+.2f}%",
                    "P&L $"     : f"${_pnl_d:+.2f}",
                    "Days"      : (date.today() - date.fromisoformat(_cp["entry_date"])).days,
                    "Score"     : f"{_score:.3f}" if _score is not None else "—",
                    "Status"    : _flag,
                })
                _cv_pnl_dollars.append(_pnl_d)
                _cv_costs.append(_cost)

            st.dataframe(
                pd.DataFrame(_cv_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                hide_index=True, use_container_width=True,
            )

            _cv_total_cost  = sum(_cv_costs)
            _cv_total_pnl_d = sum(_cv_pnl_dollars)
            _cv_wtd_pnl     = (_cv_total_pnl_d / _cv_total_cost * 100) if _cv_total_cost else 0
            _cv_long_pnl    = sum(_cv_pnl_dollars[i] for i, r in enumerate(_cv_rows) if r["Dir"] == "LONG")
            _cv_short_pnl   = sum(_cv_pnl_dollars[i] for i, r in enumerate(_cv_rows) if r["Dir"] == "SHORT")

            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            _cc1.metric("Total P&L $",   f"${_cv_total_pnl_d:+.2f}")
            _cc2.metric("Weighted P&L %", f"{_cv_wtd_pnl:+.2f}%")
            _cc3.metric("Long P&L $",    f"${_cv_long_pnl:+.2f}")
            _cc4.metric("Short P&L $",   f"${_cv_short_pnl:+.2f}")
        else:
            st.info("No open conviction positions. Run a scan to get recommendations.")

        # ── Scan controls ─────────────────────────────────────────────────────
        _cv_budget = st.number_input(
            "Portfolio size ($)", min_value=1_000, max_value=1_000_000,
            value=5_000, step=500, key="cv_budget",
        )

        if st.button("Run Conviction Scan", type="primary", key="cv_scan"):
            _summary_path = db.get_stock_file("ticker_summary.csv", ROOT)
            if _summary_path is None and (ROOT / "stocks" / "ticker_summary.csv").exists():
                _summary_path = ROOT / "stocks" / "ticker_summary.csv"
            if _summary_path is None and (ROOT / "ticker_summary.csv").exists():
                _summary_path = ROOT / "ticker_summary.csv"

            if _summary_path is None:
                st.error("ticker_summary.csv not found — run stock model training first.")
            else:
                _cv_tickers  = pd.read_csv(_summary_path)["Ticker"].tolist()
                _cv_progress = st.progress(0, text="Fetching data…")

                def _cv_cb(i, n, t):
                    _cv_progress.progress((i + 1) / n, text=f"Scoring {t} ({i+1}/{n})")

                try:
                    _cv_df = _cv_mod.run_conviction_scan(_cv_tickers, _cv_cb)
                    _cv_progress.empty()
                    st.session_state["cv_scan_result"] = _cv_df
                    st.session_state["cv_scan_budget"] = _cv_budget
                except Exception as _cv_scan_err:
                    _cv_progress.empty()
                    st.error(f"Conviction scan error: {_cv_scan_err}")

        # ── Display scan results ──────────────────────────────────────────────
        if "cv_scan_result" in st.session_state:
            _cv_df          = st.session_state["cv_scan_result"]
            _cv_budget_used = st.session_state.get("cv_scan_budget", _cv_budget)

            if not _cv_df.empty:
                # Full ranked table
                with st.expander("Full ranked universe", expanded=False):
                    _cv_display_cols = [
                        "rank", "ticker", "direction", "composite_score",
                        "z_technical", "z_fundamental", "z_earnings", "z_macro",
                        "ret_3m", "ret_6m", "roe", "revenue_growth",
                        "beat_rate", "earnings_flag", "current_price",
                    ]
                    _cv_display_cols = [c for c in _cv_display_cols if c in _cv_df.columns]
                    _cv_disp = _cv_df[_cv_display_cols].copy()
                    for _pct_col in ["ret_3m", "ret_6m", "roe", "revenue_growth", "beat_rate"]:
                        if _pct_col in _cv_disp.columns:
                            _cv_disp[_pct_col] = _cv_disp[_pct_col].apply(
                                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—"
                            )
                    for _score_col in ["composite_score", "z_technical", "z_fundamental",
                                       "z_earnings", "z_macro"]:
                        if _score_col in _cv_disp.columns:
                            _cv_disp[_score_col] = _cv_disp[_score_col].apply(
                                lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
                            )
                    st.dataframe(_cv_disp, hide_index=True, use_container_width=True)

                _cv_longs  = _cv_df[_cv_df["direction"] == "LONG"]
                _cv_shorts = _cv_df[_cv_df["direction"] == "SHORT"]

                # Macro regime info bar
                _cv_regime    = _cv_df["regime"].iloc[0] if "regime" in _cv_df.columns else "—"
                _cv_max_long  = int(_cv_df["max_long"].iloc[0])  if "max_long"  in _cv_df.columns else 5
                _cv_max_short = int(_cv_df["max_short"].iloc[0]) if "max_short" in _cv_df.columns else 5
                _regime_colors = {"Expansion": "🟢", "Caution": "🟡", "Contraction": "🔴"}
                st.info(
                    f"{_regime_colors.get(_cv_regime, '⚪')} Macro regime: **{_cv_regime}** — "
                    f"slots available: {_cv_max_long}L / {_cv_max_short}S — "
                    f"{len(_cv_longs)} LONG + {len(_cv_shorts)} SHORT recommendations"
                )

                # LONG recommendations
                st.subheader("LONG Recommendations")
                if _cv_longs.empty:
                    st.caption("No tickers cleared the long conviction threshold today.")
                else:
                    _cv_existing = {p["ticker"] for p in _cv_open}
                    _alloc_long  = _cv_budget_used / 2
                    _per_long    = _alloc_long / max(len(_cv_longs), 1)

                    for _, _lr in _cv_longs.iterrows():
                        _tk      = _lr["ticker"]
                        _already = _tk in _cv_existing
                        _shares  = round(_per_long / _lr["current_price"], 4) if _lr["current_price"] else 0
                        _eflag   = _lr.get("earnings_flag", False)
                        _edays   = _lr.get("earnings_days_out")

                        _col1, _col2 = st.columns([3, 1])
                        with _col1:
                            _score_breakdown = (
                                f"Score **{_lr['composite_score']:+.3f}** | "
                                f"Tech {_lr.get('z_technical', float('nan')):+.2f} · "
                                f"Fund {_lr.get('z_fundamental', float('nan')):+.2f} · "
                                f"Earn {_lr.get('z_earnings', float('nan')):+.2f} · "
                                f"Macro {_lr.get('z_macro', float('nan')):+.2f}"
                            )
                            _earn_note = f" | 📅 Earnings in {_edays}d" if _eflag and _edays else ""
                            st.markdown(f"**{_tk}** @ ${_lr['current_price']}  —  {_score_breakdown}{_earn_note}")
                            st.caption(f"{_shares:.4f} shares · ${_per_long:,.0f} allocation")
                        with _col2:
                            if _already:
                                st.caption("Already held")
                            elif st.button(f"📈 Record Long {_tk}", key=f"cv_long_{_tk}"):
                                _new_cv = {
                                    "ticker"          : _tk,
                                    "direction"       : "LONG",
                                    "status"          : "open",
                                    "entry_price"     : _lr["current_price"],
                                    "entry_date"      : str(date.today()),
                                    "shares"          : _shares,
                                    "cost"            : round(_shares * _lr["current_price"], 2),
                                    "composite_score" : float(_lr["composite_score"]),
                                    "score_at_entry"  : float(_lr["composite_score"]),
                                    "exit_signal"     : None,
                                    "reassess_signal" : None,
                                    "earnings_flag"   : bool(_eflag),
                                    "earnings_days_out": _edays,
                                }
                                _cv_positions.append(_new_cv)
                                _cv_mod.save_conviction_positions(_cv_positions)
                                st.success(f"Recorded LONG {_tk} @ ${_lr['current_price']}")
                                st.rerun()

                # SHORT recommendations
                st.subheader("SHORT Recommendations")
                if _cv_shorts.empty:
                    st.caption("No tickers cleared the short conviction threshold today.")
                else:
                    _alloc_short = _cv_budget_used / 2
                    _per_short   = _alloc_short / max(len(_cv_shorts), 1)

                    for _, _sr in _cv_shorts.iterrows():
                        _tk      = _sr["ticker"]
                        _already = _tk in _cv_existing if "_cv_existing" in dir() else _tk in {p["ticker"] for p in _cv_open}
                        _shares  = round(_per_short / _sr["current_price"], 4) if _sr["current_price"] else 0
                        _eflag   = _sr.get("earnings_flag", False)
                        _edays   = _sr.get("earnings_days_out")

                        _col1, _col2 = st.columns([3, 1])
                        with _col1:
                            _score_breakdown = (
                                f"Score **{_sr['composite_score']:+.3f}** | "
                                f"Tech {_sr.get('z_technical', float('nan')):+.2f} · "
                                f"Fund {_sr.get('z_fundamental', float('nan')):+.2f} · "
                                f"Earn {_sr.get('z_earnings', float('nan')):+.2f} · "
                                f"Macro {_sr.get('z_macro', float('nan')):+.2f}"
                            )
                            _earn_note = f" | 📅 Earnings in {_edays}d" if _eflag and _edays else ""
                            st.markdown(f"**{_tk}** @ ${_sr['current_price']}  —  {_score_breakdown}{_earn_note}")
                            st.caption(f"{_shares:.4f} shares · ${_per_short:,.0f} allocation")
                        with _col2:
                            if _already:
                                st.caption("Already held")
                            elif st.button(f"📉 Record Short {_tk}", key=f"cv_short_{_tk}"):
                                _new_cv = {
                                    "ticker"          : _tk,
                                    "direction"       : "SHORT",
                                    "status"          : "open",
                                    "entry_price"     : _sr["current_price"],
                                    "entry_date"      : str(date.today()),
                                    "shares"          : _shares,
                                    "cost"            : round(_shares * _sr["current_price"], 2),
                                    "composite_score" : float(_sr["composite_score"]),
                                    "score_at_entry"  : float(_sr["composite_score"]),
                                    "exit_signal"     : None,
                                    "reassess_signal" : None,
                                    "earnings_flag"   : bool(_eflag),
                                    "earnings_days_out": _edays,
                                }
                                _cv_positions.append(_new_cv)
                                _cv_mod.save_conviction_positions(_cv_positions)
                                st.success(f"Recorded SHORT {_tk} @ ${_sr['current_price']}")
                                st.rerun()

        # ── Reassess open positions ───────────────────────────────────────────
        if _cv_open and "cv_scan_result" in st.session_state:
            if st.button("Reassess Open Positions", key="cv_reassess"):
                _cv_updated = _cv_mod.assess_open_positions(
                    _cv_open, st.session_state["cv_scan_result"]
                )
                _cv_by_ticker = {p["ticker"]: p for p in _cv_updated}
                _cv_positions = [
                    _cv_by_ticker.get(p["ticker"], p) if p.get("status") == "open" else p
                    for p in _cv_positions
                ]
                _cv_mod.save_conviction_positions(_cv_positions)
                st.success("Positions reassessed.")
                st.rerun()

        # ── Close a position manually ─────────────────────────────────────────
        if _cv_open:
            with st.expander("Close a position"):
                _cv_close_ticker = st.selectbox(
                    "Ticker", [p["ticker"] for p in _cv_open], key="cv_close_sel"
                )
                _cv_close_price = st.number_input(
                    "Exit price ($)", min_value=0.01, value=100.0, key="cv_close_price"
                )
                _cv_close_reason = st.selectbox(
                    "Reason", ["reassessment", "hard_stop", "manual"], key="cv_close_reason"
                )
                if st.button("Close Position", key="cv_close_btn"):
                    for _cp in _cv_positions:
                        if _cp["ticker"] == _cv_close_ticker and _cp["status"] == "open":
                            _ep = _cp["entry_price"]
                            if _cp["direction"] == "LONG":
                                _pnl_pct = (_cv_close_price - _ep) / _ep * 100
                                _pnl_d   = (_cv_close_price - _ep) * _cp.get("shares", 0)
                            else:
                                _pnl_pct = (_ep - _cv_close_price) / _ep * 100
                                _pnl_d   = (_ep - _cv_close_price) * _cp.get("shares", 0)
                            _cp.update({
                                "status"      : "closed",
                                "exit_price"  : _cv_close_price,
                                "exit_date"   : str(date.today()),
                                "exit_reason" : _cv_close_reason,
                                "pnl_pct"     : round(_pnl_pct, 2),
                                "pnl_dollars" : round(_pnl_d, 2),
                            })
                            break
                    _cv_mod.save_conviction_positions(_cv_positions)
                    st.success(f"Closed {_cv_close_ticker}.")
                    st.rerun()

        # ── Closed positions history ──────────────────────────────────────────
        if _cv_closed:
            with st.expander(f"Closed conviction positions ({len(_cv_closed)})"):
                _cv_c_rows = []
                for _cp in sorted(_cv_closed, key=lambda x: x.get("exit_date", ""), reverse=True):
                    _pnl = _cp.get("pnl_pct", 0)
                    _cv_c_rows.append({
                        "Ticker"    : _cp["ticker"],
                        "Dir"       : _cp["direction"],
                        "Entry $"   : f"${_cp['entry_price']:.2f}",
                        "Exit $"    : f"${_cp.get('exit_price', 0):.2f}",
                        "Entry Date": _cp.get("entry_date", ""),
                        "Exit Date" : _cp.get("exit_date", ""),
                        "Days"      : _cp.get("days_held", "—"),
                        "Reason"    : _cp.get("exit_reason", "—"),
                        "P&L %"     : f"{_pnl:+.2f}%",
                        "P&L $"     : f"${_cp.get('pnl_dollars', 0):+.2f}",
                    })
                st.dataframe(
                    pd.DataFrame(_cv_c_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                    hide_index=True, use_container_width=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TAB
# ══════════════════════════════════════════════════════════════════════════════
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_expiry(ticker: str):
    try:
        code = ticker.split("-")[1]          # e.g. "26MAR2717"
        return datetime(2000 + int(code[:2]), _MONTHS[code[2:5]], int(code[5:7]))
    except Exception:
        return None


def _week_label(dt) -> str:
    if dt is None:
        return "Unknown"
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%-d %b")         # e.g. "24 Mar"


with tab_perf:
    if not closed_kalshi:
        st.info("No closed positions yet.")
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── Build perf_data ────────────────────────────────────────────────────
        perf_data = []
        for p in closed_kalshi:
            result    = settlement_map.get(p["ticker"])
            rem_yes   = p.get("rem_yes", 0)
            rem_no    = p.get("rem_no", 0)
            remaining = rem_yes + rem_no
            buy_cost  = p.get("buy_cost", 0)
            sell_proc = p.get("sell_proceeds", 0)
            fees      = p.get("total_fees", 0)

            if result is not None:
                settle_val = (rem_yes if result == "yes" else 0) + (rem_no if result == "no" else 0)
                pnl = sell_proc - buy_cost + settle_val - fees
            elif remaining == 0:
                pnl = sell_proc - buy_cost - fees
            else:
                pnl = None

            asset, _expiry_str, strike = parse_ticker(p["ticker"])
            expiry_dt  = _parse_expiry(p["ticker"])
            week_label = _week_label(expiry_dt)

            # Exit ¢
            ctrs      = p.get("contracts", 1)
            entry     = p.get("entry_cents", 0)
            ctrs_sold = ctrs - remaining
            if ctrs_sold > 0 and sell_proc > 0:
                exit_cents = round(sell_proc / ctrs_sold * 100)
            elif result is not None:
                won = (rem_yes > 0 and result == "yes") or (rem_no > 0 and result == "no")
                exit_cents = 100 if won else 0
            else:
                exit_cents = None

            pnl_pct = (pnl / buy_cost * 100) if (pnl is not None and buy_cost and buy_cost > 0) else None
            perf_data.append({
                "ticker"     : p["ticker"],
                "asset"      : asset,
                "strike"     : strike,
                "expiry_dt"  : expiry_dt,
                "expiry_str" : _expiry_str,
                "week"       : week_label,
                "side"       : p.get("side", "yes").upper(),
                "contracts"  : ctrs,
                "entry_cents": entry,
                "exit_cents" : exit_cents,
                "pnl"        : pnl,
                "pnl_pct"    : pnl_pct,
                "buy_cost"   : buy_cost,
            })

        resolved = [d for d in perf_data if d["pnl"] is not None]

        # ── Summary metrics ────────────────────────────────────────────────────
        if resolved:
            total_pnl  = sum(d["pnl"] for d in resolved)
            wins       = sum(1 for d in resolved if d["pnl"] > 0)
            win_rate   = wins / len(resolved)
            best_trade = max(d["pnl"] for d in resolved)
            worst_trade = min(d["pnl"] for d in resolved)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Total P&L",   f"${total_pnl:+.2f}")
            mc2.metric("Win Rate",    f"{win_rate:.0%}")
            mc3.metric("Total Bets",  len(resolved))
            mc4.metric("Best Trade",  f"${best_trade:+.2f}")
            mc5.metric("Worst Trade", f"${worst_trade:+.2f}")

        st.divider()

        # ── P&L bar charts by week ────────────────────────────────────────────
        def _make_weekly_chart(data_subset, title, pct=False):
            from collections import defaultdict as _dd
            week_val  = _dd(float)
            week_cost = _dd(float)
            week_order = {}
            for d in data_subset:
                if d["pnl"] is None:
                    continue
                wk = d["week"]
                week_val[wk]  += d["pnl"]
                week_cost[wk] += d.get("buy_cost", 0)
                if wk not in week_order and d["expiry_dt"] is not None:
                    monday = d["expiry_dt"] - timedelta(days=d["expiry_dt"].weekday())
                    week_order[wk] = monday

            if not week_val:
                return None

            sorted_weeks = sorted(week_val.keys(),
                                  key=lambda w: week_order.get(w, datetime.min))
            if pct:
                values = [(week_val[w] / week_cost[w] * 100) if week_cost[w] > 0 else 0
                          for w in sorted_weeks]
                ylabel = "P&L (%)"
            else:
                values = [week_val[w] for w in sorted_weeks]
                ylabel = "P&L ($)"

            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(sorted_weeks, values, color=colors)
            ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_xlabel("Week of", color="white", fontsize=9)
            ax.set_ylabel(ylabel, color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            return fig

        btc_data = [d for d in perf_data if d["asset"] == "BTC"]
        eth_data = [d for d in perf_data if d["asset"] == "ETH"]

        # Row 1: $ charts
        col_btc, col_eth = st.columns(2)
        with col_btc:
            fig_btc = _make_weekly_chart(btc_data, "BTC — P&L by Week ($)")
            if fig_btc:
                st.pyplot(fig_btc)
                plt.close(fig_btc)
            else:
                st.caption("No BTC closed positions with P&L yet.")
        with col_eth:
            fig_eth = _make_weekly_chart(eth_data, "ETH — P&L by Week ($)")
            if fig_eth:
                st.pyplot(fig_eth)
                plt.close(fig_eth)
            else:
                st.caption("No ETH closed positions with P&L yet.")

        # Row 2: % charts
        col_btc_pct, col_eth_pct = st.columns(2)
        with col_btc_pct:
            fig_btc_pct = _make_weekly_chart(btc_data, "BTC — P&L by Week (%)", pct=True)
            if fig_btc_pct:
                st.pyplot(fig_btc_pct)
                plt.close(fig_btc_pct)
        with col_eth_pct:
            fig_eth_pct = _make_weekly_chart(eth_data, "ETH — P&L by Week (%)", pct=True)
            if fig_eth_pct:
                st.pyplot(fig_eth_pct)
                plt.close(fig_eth_pct)

        st.divider()

        # ── Side breakdown (YES vs NO P&L) ────────────────────────────────────
        yes_trades  = [d for d in resolved if d["side"] == "YES"]
        no_trades   = [d for d in resolved if d["side"] == "NO"]
        yes_pnl     = sum(d["pnl"] for d in yes_trades)
        no_pnl      = sum(d["pnl"] for d in no_trades)
        yes_cost    = sum(d.get("buy_cost", 0) for d in yes_trades)
        no_cost     = sum(d.get("buy_cost", 0) for d in no_trades)
        yes_pnl_pct = (yes_pnl / yes_cost * 100) if yes_cost > 0 else 0
        no_pnl_pct  = (no_pnl  / no_cost  * 100) if no_cost  > 0 else 0

        def _side_bar(labels, values, title, ylabel):
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.bar(labels, values, color=colors)
            ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_ylabel(ylabel, color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=9)
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            plt.tight_layout()
            return fig

        if resolved:
            _sc1, _sc2, _ = st.columns([1, 1, 1])
            with _sc1:
                fig_side = _side_bar(["YES", "NO"], [yes_pnl, no_pnl],
                                     "P&L by Side ($)", "P&L ($)")
                st.pyplot(fig_side)
                plt.close(fig_side)
            with _sc2:
                fig_side_pct = _side_bar(["YES", "NO"], [yes_pnl_pct, no_pnl_pct],
                                         "P&L by Side (%)", "P&L (%)")
                st.pyplot(fig_side_pct)
                plt.close(fig_side_pct)

        st.divider()

        # ── Detailed table ─────────────────────────────────────────────────────
        st.subheader("All Closed Positions")
        detail_rows = []
        for d in sorted(perf_data,
                        key=lambda x: x["expiry_dt"] or datetime.min,
                        reverse=True):
            strike_fmt = (f"${float(d['strike']):,.0f}"
                          if d["strike"] and d["strike"].replace(".", "").isdigit()
                          else d["strike"])
            detail_rows.append({
                "Asset"    : d["asset"],
                "Strike"   : strike_fmt,
                "Expiry"   : d["expiry_str"],
                "Side"     : d["side"],
                "Contracts": d["contracts"],
                "Entry ¢"  : d["entry_cents"],
                "Exit ¢"   : d["exit_cents"] if d["exit_cents"] is not None else "—",
                "Exit"     : ("sold" if d["pnl"] is not None and
                              settlement_map.get(d["ticker"]) is None
                              else settlement_map.get(d["ticker"], "pending") or "pending"),
                "P&L $"    : (f"${d['pnl']:+.2f}" if d["pnl"] is not None else "—"),
            })

        df_perf = pd.DataFrame(detail_rows)
        st.dataframe(
            df_perf.style.map(color_pnl, subset=["P&L $"]),
            hide_index=True,
            use_container_width=True,
        )
