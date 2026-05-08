"""
kalshi/app.py — Kalshi Streamlit dashboard

Two tabs:
  📊 Dashboard  — auto-place / scan Kalshi crypto contracts, settled-bet metrics
  📉 Performance — closed-trade backfill and exclude/include controls

Run: streamlit run kalshi/app.py
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
ROOT = Path(__file__).parent          # = kalshi/
sys.path.insert(0, str(ROOT))         # for sibling imports (kalshi_api, kalshi_crypto, db)
sys.path.insert(0, str(ROOT.parent))  # for `shared.db_common` package

from kalshi_api import KalshiClient, KALSHI_CONFIG  # noqa: E402
import db                                            # noqa: E402  → kalshi/db.py


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

POSITIONS_KALSHI = ROOT / "positions_kalshi.json"
PAPER_TRADES     = ROOT / "paper_trades.json"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kalshi Dashboard", layout="wide")
st.title("Kalshi Dashboard")

tab_dash, tab_perf = st.tabs(["📊 Dashboard", "📉 Performance"])

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

    c1, _c_rest = st.columns([1, 10])
    with c1:
        if st.button("↻ Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # ── Load positions: API is source of truth for open positions ─────────────────
    _client          = make_kalshi_client()
    _local_by_ticker = db.load_position_overrides()   # {ticker: {entry_cents, stop_cents, contracts}}

    # Pre-fetch fills to compute avg entry prices from actual buy fills
    @st.cache_data(ttl=60, show_spinner=False)
    def _fetch_fills_cached():
        """Cache fills for 60s since they rarely change during session."""
        return _client.get_fills(limit=1000) if not _client.dry_run else []

    def _extract_yes_price(f: dict) -> float:
        """Return yes price in dollars (0–1), trying all known Kalshi field variants."""
        for key in ("yes_price_dollars", "yes_price_fp", "yes_price"):
            v = f.get(key)
            if v is not None:
                try:
                    fv = float(v)
                    return fv / 100 if fv > 1 else fv
                except (ValueError, TypeError):
                    pass
        return 0.0

    _fills_index: dict[str, dict] = {}   # ticker -> {yes: cents, no: cents}
    if not _client.dry_run:
        try:
            _prefetch_fills = _fetch_fills_cached()
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
                    _yp = _extract_yes_price(_pf)
                    if _yp == 0:
                        continue  # skip fills with no usable price rather than corrupting avg
                    _np = max(0.0, 1.0 - _yp)
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

    @st.cache_data(ttl=60, show_spinner=False)
    def _fetch_positions_cached():
        """Cache positions for 60s since they rarely change during session."""
        return _client.get_positions() if not _client.dry_run else []

    if not _client.dry_run:
        try:
            _api_positions = _fetch_positions_cached()
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
                # For trades <1hr to expiration, use wider 25% stop to avoid whipsaws
                _saved_stop = _local.get("stop_cents")
                _stop_pct = 0.25 if (_hrs is not None and _hrs < 1) else 0.50
                _stop = _saved_stop if _saved_stop else round(_entry * _stop_pct)
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
            placed_at    = _local_by_ticker.get(ticker, {}).get("placed_at", "")[:16] if _local_by_ticker.get(ticker, {}).get("placed_at") else "—"
            rows.append({
                "Ticker"   : ticker,
                "Placed At" : placed_at,
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
            df_open.drop(columns=["Ticker", "Bet $", "Stop $"]),
            column_config={
                "Contracts": st.column_config.NumberColumn("Contracts", min_value=1, step=1),
                "Entry ¢"  : st.column_config.NumberColumn("Entry ¢", min_value=0, max_value=99, step=1),
                "Stop ¢"   : st.column_config.NumberColumn("Stop ¢",  min_value=0, max_value=99, step=1),
            },
            disabled=["Asset", "Strike", "Hrs Left", "Live Bid", "P&L", "Placed At"],
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
            try:
                db.save_position_overrides(_local_by_ticker)
                st.cache_data.clear()
                st.rerun()
            except Exception as _save_err:
                st.error(f"Save failed — {_save_err}. Check Supabase connection / position_overrides schema.")

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


    # Scan settings
    st.write("**Scan Settings**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        skip_intraday_short = st.checkbox(
            "Skip 1–8h trades",
            value=False,
            help="Don't save 1–8h (intraday_short) trades due to 0% win rate."
        )

    flags = db.load_feature_flags()

    with col2:
        auto_place_vol = st.checkbox(
            "Auto-place vol",
            value=flags.get("auto_place_vol", False),
            help="Auto-place vol (<1h) trades with -$25/day loss limit."
        )
        if auto_place_vol != st.session_state.get("_last_auto_place_vol", False):
            db.set_feature_flag("auto_place_vol", auto_place_vol)
            st.session_state["_last_auto_place_vol"] = auto_place_vol
            if auto_place_vol:
                st.success("✅ Vol auto-placement enabled")
            else:
                st.info("⏸ Vol auto-placement disabled")

    with col3:
        auto_place_intraday_long = st.checkbox(
            "Auto-place 8–24h",
            value=flags.get("auto_place_intraday_long", False),
            help="Auto-place 8–24h (intraday_long) trades. Requires recalibration first."
        )
        if auto_place_intraday_long != st.session_state.get("_last_auto_place_intraday_long", False):
            db.set_feature_flag("auto_place_intraday_long", auto_place_intraday_long)
            st.session_state["_last_auto_place_intraday_long"] = auto_place_intraday_long
            if auto_place_intraday_long:
                st.success("✅ 8–24h auto-placement enabled")
            else:
                st.info("⏸ 8–24h auto-placement disabled")

    with col4:
        auto_place_weekly = st.checkbox(
            "Auto-place weekly",
            value=flags.get("auto_place_weekly", False),
            help="Auto-place weekly (>24h) trades via cron."
        )
        if auto_place_weekly != st.session_state.get("_last_auto_place_weekly", False):
            db.set_feature_flag("auto_place_weekly", auto_place_weekly)
            st.session_state["_last_auto_place_weekly"] = auto_place_weekly
            if auto_place_weekly:
                st.success("✅ Weekly auto-placement enabled")
            else:
                st.info("⏸ Weekly auto-placement disabled")

    if st.button("Run Kalshi Scan", type="primary", key="scan_kalshi"):
        try:
            from kalshi_crypto import (
                load_crypto_models, score_contract,
                download_crypto, download_crypto_hourly,
                fetch_binance_minute_closes, BINANCE_SYMBOLS,
                CRYPTO_ASSETS, KALSHI_SERIES, INFERENCE_PERIOD,
            )

            @st.cache_data(ttl=3600, show_spinner=False)
            def _load_crypto_models_cached():
                """Cache crypto models (50MB+ joblib files) for 1 hour."""
                return load_crypto_models()

            @st.cache_data(ttl=3600, show_spinner=False)
            def _load_calibration_cached():
                """Cache calibration data (only changes on manual recalibration)."""
                return db.load_calibration_db()

            with st.spinner("Loading model..."):
                models = _load_crypto_models_cached()
                _db_cal = _load_calibration_cached()
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

                # Cache price data to avoid repeated API calls
                @st.cache_data(ttl=3600, show_spinner=False)
                def _download_crypto_cached(symbol):
                    return download_crypto(symbol, INFERENCE_PERIOD)

                @st.cache_data(ttl=1800, show_spinner=False)
                def _download_crypto_hourly_cached(symbol):
                    return download_crypto_hourly(symbol)

                @st.cache_data(ttl=300, show_spinner=False)
                def _fetch_binance_minute_closes_cached(symbol):
                    return fetch_binance_minute_closes(symbol) if symbol else []

                asset_dfs_by_symbol = {}
                with st.spinner("Fetching crypto prices..."):
                    for symbol in CRYPTO_ASSETS:
                        daily_df      = _download_crypto_cached(symbol)
                        hourly_df     = _download_crypto_hourly_cached(symbol) if has_intraday else None
                        binance_sym   = BINANCE_SYMBOLS.get(symbol)
                        minute_closes = _fetch_binance_minute_closes_cached(binance_sym) if has_intraday else []
                        asset_dfs_by_symbol[symbol] = {"daily": daily_df, "hourly": hourly_df, "minute": minute_closes}

                @st.cache_data(ttl=600, show_spinner=False)
                def _get_markets_cached(series):
                    """Cache market lists (only change when new expirations appear)."""
                    return client.get_markets(series_ticker=series, status="open")

                all_results = []
                scan_debug  = []
                with st.spinner("Scoring contracts..."):
                    for symbol, series in KALSHI_SERIES.items():
                        markets   = _get_markets_cached(series)
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
                    # Filter out intraday_short (1-8h) if toggled off
                    if skip_intraday_short:
                        _daily_port_filtered = [p for p in _daily_port if p["hours_to_expiry"] >= 8]
                    else:
                        _daily_port_filtered = _daily_port
                    if _daily_port_filtered:
                        save_paper_trades(_daily_port_filtered, "daily")
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
            # Test auto-placement for weekly trades
            if st.button("Place All Weekly Recommendations", key="place_weekly_manual"):
                try:
                    from kalshi_crypto import place_scheduled_orders
                    client = make_kalshi_client()
                    stats = place_scheduled_orders(client)
                    st.success(f"Placement attempt: {stats['weekly_placed']} weekly placed, "
                              f"{stats['vol_placed']} vol placed, {stats['skipped']} skipped.")
                except Exception as e:
                    st.error(f"Error placing trades: {e}")
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

    @st.cache_data(ttl=300, show_spinner=False)
    def _load_settled_trades():
        """Cache settled trades (they don't change). 5min TTL."""
        _all = db.load_paper_trades()
        return [t for t in _all if t.get("status") == "settled"]

    _settled_paper = _load_settled_trades()

    # Load only open trades (these may change on settlement checks)
    _all_paper = db.load_paper_trades()
    _open_paper = [t for t in _all_paper if t.get("status") == "open"]

    with st.expander("🔍 Debug: Raw DB (last 10 trades)", expanded=False):
        _debug_rows = sorted(_all_paper, key=lambda x: x.get("placed_at", ""), reverse=True)[:10]
        st.json([{k: v for k, v in t.items()
                  if k in ("ticker","side","status","price_cents","close_time","placed_at","result","pnl_dollars")}
                 for t in _debug_rows])

    if not _all_paper:
        st.info("No paper trades recorded yet. Run the Kalshi Scan and click '📝 Paper Trade' to start tracking.")
    else:
        # ── Auto-settle expired trades (only check open trades) ──────────────────────
        _now_utc       = datetime.now(timezone.utc)
        _newly_settled = 0

        # Re-open any trades that were incorrectly settled with empty result
        for _pt in _open_paper:
            if _pt.get("status") == "settled" and not _pt.get("result"):
                db.reopen_paper_trade(_pt["id"])
                _pt["status"] = "open"
                _pt.pop("result", None)
                _pt.pop("pnl_dollars", None)

        # Correct P&L for settled trades where result was stored with wrong case
        for _pt in _settled_paper:
            if _pt.get("result"):
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

        for _pt in _open_paper:
            if _pt.get("status") == "open":
                _close_str = _pt.get("close_time", "")
                _closed    = False
                if _close_str:
                    try:
                        _close_dt = datetime.fromisoformat(_close_str.replace("Z", "+00:00"))
                        _closed   = _close_dt <= _now_utc
                    except Exception:
                        _closed = True
                else:
                    _closed = True
                if _closed and not _client.dry_run:
                    try:
                        _mkt    = _client._request("GET", f"/markets/{_pt['ticker']}").get("market", {})
                        _result = _mkt.get("result")
                        if _result:
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
                _placed = _pt.get("placed_at", "")[:16] if _pt.get("placed_at") else "—"
                _pt_rows.append({
                    "Placed At"  : _placed,
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
                # Filter: show only ACTUAL trades (placement_status='placed')
                _actual_count = sum(1 for t in _settled_paper if t.get("placement_status") == "placed")
                _filter_actual = st.checkbox("Show ACTUAL trades only", value=True, key="filter_actual_trades")

                _trades_to_show = ([t for t in _settled_paper if t.get("placement_status") == "placed"]
                                   if _filter_actual else _settled_paper)

                _s_rows = []
                for _pt in sorted(_trades_to_show, key=lambda x: x.get("placed_at", ""), reverse=True):
                    _pnl = _pt.get("pnl_dollars")
                    _rec_h = _pt.get("hours_to_exp")
                    _type = "ACTUAL" if _pt.get("placement_status") == "placed" else "PAPER"
                    _s_rows.append({
                        "Type"     : _type,
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
                st.caption(f"Actual trades: {_actual_count}/{len(_settled_paper)}")

        # ── Performance summary ───────────────────────────────────────────────────
        # Paper-only: exclude rows that were either (a) placed real on Kalshi, or
        # (b) recorded directly as real positions via monitor.py. Real-trade
        # performance lives on the Performance tab.
        _resolved = [t for t in _settled_paper
                     if not t.get("real_trade")
                        and t.get("placement_status") != "placed"]

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
            # Group by mapped bucket label (consolidates "weekly" + "daily" → ">24hr")
            _bucket_labels = {
                "vol": "<1hr (vol)",
                "intraday_short": "1-8hr",
                "intraday_long": "8-24hr",
                "weekly": ">24hr",
                # legacy/fallback
                "daily": ">24hr",
                "intraday": "1-24hr",
            }
            _buckets_by_label = {}
            for _p in _resolved:
                # Use saved bucket field (matches calibration logic) instead of recalculating
                _b = _p.get("bucket")
                if not _b:
                    # Fallback for trades without bucket field: recalculate from hours_to_exp
                    _b = _time_bucket(_p.get("hours_to_exp"))
                _label = _bucket_labels.get(_b, _b)
                _buckets_by_label.setdefault(_label, []).append(_p)

            _bk_rows = []
            _bucket_order = ["<1hr (vol)", "1-8hr", "8-24hr", ">24hr", "1-24hr", "unknown"]
            for _label in _bucket_order:
                _bps = _buckets_by_label.get(_label)
                if not _bps:
                    continue
                _row = _bucket_stats("Bucket", _bps)
                _row["Bucket"] = _label
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


# ── Helpers used inside tab_perf (lifted from the original tab_conviction → tab_perf
#    seam so they're in scope when this app stands alone). ──────────────────────
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict as _dd_perf

    # ── Backfill button ────────────────────────────────────────────────────────
    if st.button("📥 Backfill Mar 1+", key="perf_backfill_btn"):
        _pbf_client = make_kalshi_client()
        if _pbf_client.dry_run:
            st.warning("No Kalshi credentials — backfill unavailable in dry-run mode.")
        else:
            with st.spinner("Fetching all historical fills since Mar 1..."):
                try:
                    _pbf_since_ts = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
                    _pbf_fills = _pbf_client.get_fills(limit=1000, min_ts=_pbf_since_ts)
                    st.write(f"**DEBUG:** API returned {len(_pbf_fills)} total fills")

                    if not _pbf_fills:
                        st.warning("No fills found in that date range.")
                    else:
                        # Filter out Kalshi bug entries ($0 price/cost) - keep only real trades
                        _pbf_real = [f for f in _pbf_fills
                                    if (_price_dollars(f, "yes_price_dollars") > 0 or
                                        _price_dollars(f, "no_price_dollars") > 0)]
                        st.write(f"**DEBUG:** {len(_pbf_real)} fills with valid prices")

                        from collections import defaultdict as _dd_bf
                        _pbf_by_tkr: dict = _dd_bf(list)
                        for _pbf in _pbf_real:
                            _pbtk = _pbf.get("market_ticker") or _pbf.get("ticker", "")
                            if _pbtk and (_pbtk.startswith("KXBTC") or _pbtk.startswith("KXETH")):
                                _pbf_by_tkr[_pbtk].append(_pbf)
                        st.write(f"**DEBUG:** {len(_pbf_by_tkr)} crypto tickers found: {list(_pbf_by_tkr.keys())[:5]}")
                        st.write(f"**DEBUG:** Fills per ticker: {dict((k, len(v)) for k, v in list(_pbf_by_tkr.items())[:5])}")

                        # Pre-fetch all market results upfront
                        _pbf_all_settle = {}
                        for _pbtk in _pbf_by_tkr.keys():
                            try:
                                _pbf_mkt = _pbf_client._request("GET", f"/markets/{_pbtk}").get("market", {})
                                _pbf_all_settle[_pbtk] = _pbf_mkt.get("result")
                            except Exception:
                                _pbf_all_settle[_pbtk] = None

                        _pbf_positions = []
                        for _pbtk, _pbf_tkr_fills in _pbf_by_tkr.items():
                            # Group fills by side (YES and NO tracks separately)
                            _pbf_by_side = {"yes": [], "no": []}
                            for _pbf in _pbf_tkr_fills:
                                _pbf_side = _pbf.get("side", "yes")
                                _pbf_by_side[_pbf_side].append(_pbf)

                            _pbf_asset, _pbf_exp_str, _pbf_strike = parse_ticker(_pbtk)
                            _pbf_exp_dt = _parse_expiry(_pbtk)
                            # Ensure timezone-aware comparison
                            _pbf_is_expired = False
                            if _pbf_exp_dt:
                                _pbf_exp_aware = _pbf_exp_dt.replace(tzinfo=timezone.utc) if _pbf_exp_dt.tzinfo is None else _pbf_exp_dt
                                _pbf_is_expired = _pbf_exp_aware < datetime.now(timezone.utc)

                            # Process each side: BUY YES + SELL YES, or BUY NO + SELL NO
                            for _pbf_side, _pbf_side_fills in _pbf_by_side.items():
                                if not _pbf_side_fills:
                                    continue

                                _pbf_total_bought = 0
                                _pbf_total_sold = 0
                                _pbf_buy_cost = 0.0
                                _pbf_sell_proceeds = 0.0
                                _pbf_latest_ts = 0
                                _pbf_buy_count = 0
                                _pbf_sell_count = 0

                                for _pbf in _pbf_side_fills:
                                    _pbf_cnt = _fill_count(_pbf)
                                    _pbf_act = _fill_action(_pbf)
                                    # Use the correct price field based on side
                                    _pbf_price_field = "no_price_dollars" if _pbf_side == "no" else "yes_price_dollars"
                                    _pbf_price = _price_dollars(_pbf, _pbf_price_field)
                                    _pbf_ts = _pbf.get("ts") or 0

                                    if _pbf_ts > _pbf_latest_ts:
                                        _pbf_latest_ts = _pbf_ts

                                    if _pbf_act == "buy":
                                        _pbf_total_bought += _pbf_cnt
                                        _pbf_buy_cost += _pbf_cnt * _pbf_price
                                        _pbf_buy_count += 1
                                    elif _pbf_act == "sell":
                                        _pbf_total_sold += _pbf_cnt
                                        _pbf_sell_proceeds += _pbf_cnt * _pbf_price
                                        _pbf_sell_count += 1

                                if _pbf_sell_count > 0:
                                    st.write(f"**DEBUG:** {_pbtk} {_pbf_side}: {_pbf_buy_count} buy fills, {_pbf_sell_count} sell fills")

                                # Check for closed positions (matching buy/sell)
                                _is_closed = _pbf_total_bought == _pbf_total_sold and _pbf_buy_cost > 0 and _pbf_sell_proceeds > 0

                                # Also check for expired open positions (auto-settled at expiry)
                                _is_auto_settled = _pbf_is_expired and _pbf_total_bought > 0 and _pbf_total_sold == 0 and _pbf_all_settle.get(_pbtk) is not None

                                if _is_closed:
                                    _pbf_positions.append({
                                        "ticker"    : _pbtk,
                                        "side"      : _pbf_side,
                                        "asset"     : _pbf_asset,
                                        "strike"    : _pbf_strike,
                                        "expiry"    : _pbf_exp_str,
                                        "week"      : _week_label(_pbf_exp_dt),
                                        "contracts" : int(_pbf_total_bought),
                                        "entry_cents": round(_pbf_buy_cost / _pbf_total_bought * 100) if _pbf_total_bought > 0 else 0,
                                        "buy_cost"  : round(_pbf_buy_cost, 2),
                                        "sell_proceeds": round(_pbf_sell_proceeds, 2),
                                        "_latest_ts": _pbf_latest_ts,
                                        "_settlement_result": _pbf_all_settle.get(_pbtk),
                                    })
                                elif _is_auto_settled:
                                    # Expired position: auto-settled. YES is worth 100 if result=yes else 0. NO is opposite.
                                    _pbf_result = _pbf_all_settle.get(_pbtk)
                                    _pbf_exit_price = 1.0 if (_pbf_side == "yes" and _pbf_result == "yes") or (_pbf_side == "no" and _pbf_result == "no") else 0.0
                                    _pbf_sell_proceeds = _pbf_total_bought * _pbf_exit_price
                                    _pbf_positions.append({
                                        "ticker"    : _pbtk,
                                        "side"      : _pbf_side,
                                        "asset"     : _pbf_asset,
                                        "strike"    : _pbf_strike,
                                        "expiry"    : _pbf_exp_str,
                                        "week"      : _week_label(_pbf_exp_dt),
                                        "contracts" : int(_pbf_total_bought),
                                        "entry_cents": round(_pbf_buy_cost / _pbf_total_bought * 100) if _pbf_total_bought > 0 else 0,
                                        "buy_cost"  : round(_pbf_buy_cost, 2),
                                        "sell_proceeds": round(_pbf_sell_proceeds, 2),
                                        "_latest_ts": _pbf_latest_ts,
                                        "_settlement_result": _pbf_result,
                                    })
                                    st.write(f"**DEBUG:** {_pbtk} {_pbf_side}: AUTO-SETTLED @ {_pbf_exit_price} ({_pbf_result})")
                                elif _pbf_side_fills:
                                    st.write(f"**DEBUG:** {_pbtk} {_pbf_side}: bought={_pbf_total_bought}, sold={_pbf_total_sold} (STILL OPEN)")

                        st.write(f"**DEBUG:** {len(_pbf_positions)} closed/settled positions found")

                        _pbf_final = []
                        for _pbf_row in _pbf_positions:
                            _pbtk2   = _pbf_row["ticker"]
                            _pbf_bc  = _pbf_row["buy_cost"]
                            _pbf_sp  = _pbf_row["sell_proceeds"]
                            _pbf_ctrs = _pbf_row["contracts"]
                            _pbf_res = _pbf_row.get("_settlement_result")

                            # Fully closed position: PnL = sell_proceeds - buy_cost
                            _pbf_pnl = round(_pbf_sp - _pbf_bc, 2)

                            # Exit price per contract
                            _pbf_exit_c = round(_pbf_sp / _pbf_ctrs * 100) if _pbf_ctrs > 0 else None

                            # Convert timestamp to ISO format for database
                            _pbf_settled_iso = None
                            if _pbf_row["_latest_ts"]:
                                try:
                                    _pbf_settled_iso = datetime.fromtimestamp(
                                        _pbf_row["_latest_ts"], tz=timezone.utc
                                    ).isoformat()
                                except:
                                    _pbf_settled_iso = None

                            _pbf_final.append({
                                k: v for k, v in _pbf_row.items()
                                if not k.startswith("_")
                            } | {
                                "exit_cents" : _pbf_exit_c,
                                "pnl"        : _pbf_pnl,
                                "result"     : _pbf_res,
                                "settled_at" : _pbf_settled_iso,
                            })

                        st.write(f"**DEBUG:** {len(_pbf_final)} final positions ready to store")
                        if _pbf_final:
                            st.write(f"Sample position: {_pbf_final[0]}")

                        # Clear old data before fresh backfill
                        db.clear_kalshi_trades()
                        _pbf_n = db.upsert_kalshi_trades(_pbf_final)
                        st.write(f"**DEBUG:** upsert_kalshi_trades returned: {_pbf_n}")
                        st.success(f"Backfilled {_pbf_n} historical positions from Mar 1+.")
                        st.cache_data.clear()
                        st.rerun()
                except Exception as _pbf_err:
                    st.error(f"Backfill failed: {_pbf_err}", icon="❌")

    # ── Load from Supabase ─────────────────────────────────────────────────────
    _kalshi_trades = db.load_kalshi_trades()

    if not _kalshi_trades:
        st.info("No closed positions in database yet. Click 'Sync from Kalshi' to fetch.")
    else:
        # Build perf_data from stored records
        perf_data = []
        for _pt in _kalshi_trades:
            _pt_pnl      = _pt.get("pnl")
            _pt_buy_cost = _pt.get("buy_cost") or 0
            _pt_pnl_pct  = ((_pt_pnl / _pt_buy_cost * 100)
                             if (_pt_pnl is not None and _pt_buy_cost > 0) else None)
            _pt_exp_dt   = _parse_expiry(_pt["ticker"])
            perf_data.append({
                "ticker"     : _pt["ticker"],
                "asset"      : _pt.get("asset"),
                "strike"     : _pt.get("strike"),
                "expiry_dt"  : _pt_exp_dt,
                "expiry_str" : _pt.get("expiry"),
                "week"       : _pt.get("week") or _week_label(_pt_exp_dt),
                "side"       : (_pt.get("side") or "yes").upper(),
                "contracts"  : _pt.get("contracts"),
                "entry_cents": _pt.get("entry_cents"),
                "exit_cents" : _pt.get("exit_cents"),
                "pnl"        : _pt_pnl,
                "pnl_pct"    : _pt_pnl_pct,
                "buy_cost"   : _pt_buy_cost,
                "result"     : _pt.get("result"),
            })

        resolved = [d for d in perf_data if d["pnl"] is not None]

        # ── Summary metrics ────────────────────────────────────────────────────
        if resolved:
            total_pnl   = sum(d["pnl"] for d in resolved)
            wins        = sum(1 for d in resolved if d["pnl"] > 0)
            win_rate    = wins / len(resolved)
            best_trade  = max(d["pnl"] for d in resolved)
            worst_trade = min(d["pnl"] for d in resolved)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Total P&L",   f"${total_pnl:+.2f}")
            mc2.metric("Win Rate",    f"{win_rate:.0%}")
            mc3.metric("Total Bets",  len(resolved))
            mc4.metric("Best Trade",  f"${best_trade:+.2f}")
            mc5.metric("Worst Trade", f"${worst_trade:+.2f}")

        st.divider()

        # ── P&L bar charts by week ─────────────────────────────────────────────
        def _make_weekly_chart(data_subset, title, pct=False):
            week_val   = _dd_perf(float)
            week_cost  = _dd_perf(float)
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

        # ── Detailed table with exclude controls ──────────────────────────────
        st.subheader("All Closed Positions")
        st.caption("Check rows to exclude them from performance metrics, then click Apply.")

        detail_rows = []
        for d in sorted(perf_data,
                        key=lambda x: x["expiry_dt"] or datetime.min,
                        reverse=True):
            strike_fmt = (f"${float(d['strike']):,.0f}"
                          if d["strike"] and str(d["strike"]).replace(".", "").isdigit()
                          else d["strike"])
            _exit_label = d["result"] if d["result"] else (
                "sold" if d["pnl"] is not None else "pending"
            )
            detail_rows.append({
                "Exclude"  : False,
                "Asset"    : d["asset"],
                "Strike"   : strike_fmt,
                "Expiry"   : d["expiry_str"],
                "Side"     : d["side"],
                "Contracts": d["contracts"],
                "Entry ¢"  : d["entry_cents"],
                "Exit ¢"   : d["exit_cents"] if d["exit_cents"] is not None else "—",
                "Exit"     : _exit_label,
                "P&L $"    : (f"${d['pnl']:+.2f}" if d["pnl"] is not None else "—"),
                "_ticker"  : d["ticker"],
                "_side"    : d["side"].lower(),
            })

        _perf_df = pd.DataFrame(detail_rows)
        _edited_perf = st.data_editor(
            _perf_df.drop(columns=["_ticker", "_side"]),
            column_config={
                "Exclude": st.column_config.CheckboxColumn("Exclude", default=False, width="small"),
                "P&L $"  : st.column_config.TextColumn("P&L $"),
            },
            hide_index=True,
            use_container_width=True,
            key="perf_table_editor",
        )

        _excl_mask = _edited_perf["Exclude"].fillna(False)
        _n_checked = int(_excl_mask.sum())
        if _n_checked > 0:
            if st.button(f"Exclude {_n_checked} selected trade(s)", type="primary",
                         key="apply_exclusions"):
                for _ei, _erow in _perf_df[_excl_mask].iterrows():
                    db.exclude_kalshi_trade(_erow["_ticker"], _erow["_side"])
                st.success(f"Excluded {_n_checked} trade(s). Reload to update metrics.")
                st.rerun()
