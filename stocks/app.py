"""
stocks/app.py — Stocks Streamlit dashboard

Three tabs:
  📊 Dashboard           — per-ticker probability-model paper/real trades + scan
  📈 Absolute Return L/S — momentum/quality/value factor portfolio
  🎯 S&P Benchmark L/S   — 4-pillar conviction composite portfolio

Run: streamlit run stocks/app.py
"""

import importlib.util
import json
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent          # = stocks/
sys.path.insert(0, str(ROOT))         # for sibling imports (longterm, conviction, db)
sys.path.insert(0, str(ROOT.parent))  # for `shared.db_common` package

import db  # noqa: E402  → stocks/db.py

POSITIONS_STOCKS = ROOT / "positions.json"

# ── Shared helpers (originally defined inside the monorepo tab_dash block) ────
def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def color_pnl(val: str) -> str:
    if isinstance(val, str) and val.startswith("+"):
        return "color: #2ecc71; font-weight: bold"
    if isinstance(val, str) and val.startswith("-"):
        return "color: #e74c3c; font-weight: bold"
    return ""


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stocks Dashboard", layout="wide")
st.title("Stocks Dashboard")


@st.fragment
def _real_trade_editor(edit_rows, trade_ids, side_col, editor_key, save_key):
    """Inline editor for real trades. Fragment isolates reruns to this widget only."""
    orig_df = pd.DataFrame(edit_rows)
    result = st.data_editor(
        orig_df,
        column_config={
            side_col     : st.column_config.SelectboxColumn(side_col,   options=["LONG", "SHORT"]),
            "Ticker"     : st.column_config.TextColumn("Ticker"),
            "Entry $"    : st.column_config.NumberColumn("Entry $",    format="$%.2f",  step=0.01),
            "Shares"     : st.column_config.NumberColumn("Shares",     format="%.4f",   step=0.0001),
            "Entry Date" : st.column_config.DateColumn("Entry Date"),
            "Source"     : st.column_config.TextColumn("Source",      disabled=True),
            "Cur $"      : st.column_config.NumberColumn("Cur $",      format="$%.2f",  disabled=True),
            "Days"       : st.column_config.NumberColumn("Days",                        disabled=True),
            "P&L %"      : st.column_config.NumberColumn("P&L %",     format="%.2f%%", disabled=True),
            "P&L $"      : st.column_config.NumberColumn("P&L $",     format="$%.2f",  disabled=True),
            "Invested $" : st.column_config.NumberColumn("Invested $", format="$%.2f", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )
    if st.button("💾 Save Changes", key=save_key):
        any_saved = False
        for ri, orig in enumerate(edit_rows):
            ed = result.iloc[ri]
            upd = {}
            if abs(float(ed["Entry $"] or 0) - float(orig["Entry $"] or 0)) > 0.001:
                upd["entry_price"] = float(ed["Entry $"])
            if abs(float(ed["Shares"] or 0) - float(orig["Shares"] or 0)) > 0.00001:
                upd["shares"] = float(ed["Shares"])
            if str(ed["Entry Date"]) != str(orig["Entry Date"]):
                upd["entry_date"] = str(ed["Entry Date"])
            if str(ed[side_col]) != str(orig[side_col]):
                upd["side"] = "short" if ed[side_col] == "SHORT" else "long"
            if str(ed["Ticker"]).upper().strip() != str(orig["Ticker"]).upper().strip():
                upd["ticker"] = str(ed["Ticker"]).upper().strip()
            if upd:
                try:
                    db.update_stock_real_trade(trade_ids[ri], upd)
                    any_saved = True
                except Exception as ue:
                    st.error(f"Save failed: {ue}")
        if any_saved:
            st.session_state.pop(editor_key, None)
            st.success("Changes saved.")
            st.rerun(scope="app")
        else:
            st.info("No changes to save.")


@st.fragment
def _closed_trade_editor(closed_rows, trade_ids, editor_key, save_key):
    """Editable table for closed real trades — lets user fix exit price/date/reason."""
    import math as _math
    orig_df = pd.DataFrame(closed_rows)
    result = st.data_editor(
        orig_df,
        column_config={
            "Dir"        : st.column_config.TextColumn("Dir",        disabled=True),
            "Ticker"     : st.column_config.TextColumn("Ticker",     disabled=True),
            "Source"     : st.column_config.TextColumn("Source",     disabled=True),
            "Entry $"    : st.column_config.NumberColumn("Entry $",  format="$%.2f", disabled=True),
            "Exit $"     : st.column_config.NumberColumn("Exit $",   format="$%.2f", step=0.01),
            "Shares"     : st.column_config.NumberColumn("Shares",   format="%.4f",  disabled=True),
            "Reason"     : st.column_config.TextColumn("Reason"),
            "P&L %"      : st.column_config.NumberColumn("P&L %",   format="%.2f%%", disabled=True),
            "P&L $"      : st.column_config.NumberColumn("P&L $",   format="$%.2f",  disabled=True),
            "Entry Date" : st.column_config.DateColumn("Entry Date", disabled=True),
            "Exit Date"  : st.column_config.DateColumn("Exit Date"),
        },
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )
    if st.button("💾 Save Changes", key=save_key):
        any_saved = False
        for ri, orig in enumerate(closed_rows):
            ed = result.iloc[ri]
            upd = {}
            orig_exit = orig.get("Exit $") or 0
            new_exit  = float(ed["Exit $"] or 0)
            if abs(new_exit - float(orig_exit)) > 0.001 and new_exit > 0:
                upd["exit_price"] = new_exit
                # Recompute P&L with new exit price
                ep    = float(orig.get("Entry $") or 0)
                sh    = float(orig.get("Shares") or 0)
                side  = "short" if orig.get("Dir") == "SHORT" else "long"
                if ep > 0:
                    pct = ((ep - new_exit) / ep if side == "short" else (new_exit - ep) / ep)
                    upd["pnl_pct"]    = round(pct * 100, 2)
                    upd["pnl_dollars"] = round((ep - new_exit if side == "short" else new_exit - ep) * sh, 2)
            if str(ed["Exit Date"]) != str(orig.get("Exit Date", "")):
                upd["exit_date"] = str(ed["Exit Date"])
            if str(ed["Reason"]).strip() != str(orig.get("Reason", "")).strip():
                upd["exit_reason"] = str(ed["Reason"]).strip()
            if upd:
                try:
                    db.update_stock_real_trade(trade_ids[ri], upd)
                    any_saved = True
                except Exception as ue:
                    st.error(f"Save failed for {orig.get('Ticker')}: {ue}")
        if any_saved:
            st.session_state.pop(editor_key, None)
            st.success("Changes saved.")
            st.rerun(scope="app")
        else:
            st.info("No changes to save.")


tab_dash, tab_lt, tab_conviction, tab_overnight, tab_sma = st.tabs([
    "📊 Dashboard",
    "📈 Absolute Return L/S",
    "🎯 S&P Benchmark L/S",
    "🌙 Overnight Drift",
    "📉 200d SMA",
])

with tab_dash:
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
                    "stock_predict", ROOT / "predict.py"
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
            _key_suffix = f"{s['ticker']}_{s.get('side','long')}"
            _btn_paper, _btn_real, _spacer = st.columns([2, 2, 3])
            _do_paper = _btn_paper.button(
                f"📝 Paper {s['ticker']} ({_side_label})",
                key=f"pt_stock_{_key_suffix}",
            )
            _do_real = _btn_real.button(
                f"💰 Real {s['ticker']} ({_side_label})",
                key=f"rt_stock_{_key_suffix}",
            )
            if _do_paper or _do_real:
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
                    if _do_paper:
                        db.add_stock_paper_trade(_trade)
                        st.success(f"Paper {_side_label}: {s['ticker']} @ ${_live_price:.2f} × {s['shares']} shares")
                    else:
                        db.add_stock_real_trade(_trade, source="scan")
                        st.success(f"💰 Real {_side_label}: {s['ticker']} @ ${_live_price:.2f} × {s['shares']} shares")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Failed to save trade: {_e}")
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

            # ── Promote to Real Trades ───────────────────────────────────────
            st.caption("🚀 Copy a paper trade into Real Trades (paper row stays untouched).")
            for _sp in _open_sp:
                _side_short = "SHORT" if _sp.get("side") == "short" else "LONG"
                if st.button(
                    f"🚀 Promote {_sp['ticker']} ({_side_short}) to Real",
                    key=f"promote_{_sp['id']}",
                ):
                    try:
                        db.promote_paper_to_real(_sp["id"])
                        st.success(f"Copied {_sp['ticker']} to Real Trades.")
                        st.rerun()
                    except Exception as _pe:
                        st.error(f"Promote failed: {_pe}")

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

    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("Stocks — Real Trade Tracker")
    st.caption("Real money trades. Manual entry/close — no auto-settlement. "
               "Populated by the 💰 Real button on a scan or 🚀 Promote on a paper row.")

    # Per-ticker probability model trades only — exclude conviction-model trades
    # which have their own tracker on the S&P Benchmark tab.
    _stock_real = [t for t in db.load_stock_real_trades()
                   if t.get("source") in ("scan",)]
    _open_rt   = [t for t in _stock_real if t.get("status") == "open"]
    _closed_rt = [t for t in _stock_real if t.get("status") == "closed"]

    if not _stock_real:
        st.info("No real trades yet. Use 💰 Real on a scan suggestion, or 🚀 Promote a paper trade.")
    else:
        # Live prices for open real trades
        _rt_cur_prices: dict[str, float | None] = {}
        if _open_rt:
            import yfinance as _yf_rt
            _rt_tks = list({t["ticker"] for t in _open_rt})
            try:
                _rt_px = _yf_rt.download(_rt_tks, period="5d", auto_adjust=True, progress=False)
                if isinstance(_rt_px.columns, pd.MultiIndex):
                    _rt_closes = _rt_px["Close"]
                else:
                    _rt_closes = _rt_px[["Close"]]
                    _rt_closes.columns = _rt_tks
                for _tk in _rt_tks:
                    try:
                        _rt_cur_prices[_tk] = float(_rt_closes[_tk].dropna().iloc[-1])
                    except Exception:
                        _rt_cur_prices[_tk] = None
            except Exception:
                _rt_cur_prices = {tk: None for tk in _rt_tks}

        # Open real trades
        if _open_rt:
            st.subheader(f"Open ({len(_open_rt)})")
            st.caption("Click any cell in the white columns to edit. Greyed columns are read-only.")
            _rt_trade_ids  = []
            _rt_edit_rows  = []
            for _rt in _open_rt:
                _ep = float(_rt["entry_price"])
                _sh = float(_rt.get("shares") or 0)
                _is_short = _rt.get("side") == "short"
                _cur = _rt_cur_prices.get(_rt["ticker"])
                if _cur and _ep > 0:
                    _pnl_pct = ((_ep - _cur) / _ep * 100) if _is_short else ((_cur - _ep) / _ep * 100)
                    _pnl_d   = round(((_ep - _cur) if _is_short else (_cur - _ep)) * _sh, 2)
                else:
                    _pnl_pct, _pnl_d = 0.0, 0.0
                try:
                    _entry_d = date.fromisoformat(_rt.get("entry_date", ""))
                    _days = sum(1 for _d in pd.bdate_range(_entry_d, date.today()) if _d.date() > _entry_d)
                    _entry_date_val = _entry_d
                except Exception:
                    _days = 0
                    _entry_date_val = date.today()
                _rt_trade_ids.append(_rt["id"])
                _rt_edit_rows.append({
                    "Side"       : "SHORT" if _is_short else "LONG",
                    "Ticker"     : _rt["ticker"],
                    "Entry $"    : _ep,
                    "Shares"     : _sh,
                    "Entry Date" : _entry_date_val,
                    "Source"     : _rt.get("source", "scan"),
                    "Cur $"      : _cur,
                    "Days"       : _days,
                    "P&L %"      : round(_pnl_pct, 1),
                    "P&L $"      : round(_pnl_d, 2),
                    "Invested $" : round(_ep * _sh, 2),
                })
            _real_trade_editor(_rt_edit_rows, _rt_trade_ids, "Side", "rt_open_editor", "rt_save_edits")

            # Per-row close form
            st.caption("✅ Close a real trade with the actual fill price:")
            for _rt in _open_rt:
                _side_short = "SHORT" if _rt.get("side") == "short" else "LONG"
                with st.expander(f"Close {_rt['ticker']} ({_side_short}) — entered ${float(_rt['entry_price']):.2f}"):
                    with st.form(f"close_rt_{_rt['id']}"):
                        _exit_price = st.number_input(
                            "Exit price ($)", min_value=0.0, step=0.01,
                            value=float(_rt_cur_prices.get(_rt["ticker"]) or _rt["entry_price"]),
                            key=f"rt_exit_px_{_rt['id']}",
                        )
                        _exit_date = st.date_input(
                            "Exit date", value=date.today(), key=f"rt_exit_dt_{_rt['id']}",
                        )
                        _exit_reason = st.selectbox(
                            "Reason", ["target", "stop", "time", "manual", "other"],
                            key=f"rt_exit_rs_{_rt['id']}",
                        )
                        _notes = st.text_input("Notes (optional)", key=f"rt_notes_{_rt['id']}")
                        if st.form_submit_button("Close trade"):
                            try:
                                db.close_stock_real_trade(
                                    _rt["id"], float(_exit_price),
                                    _exit_date.isoformat(), _exit_reason,
                                    notes=_notes or None,
                                )
                                st.success(f"Closed {_rt['ticker']} @ ${_exit_price:.2f}")
                                st.rerun()
                            except Exception as _ce:
                                st.error(f"Close failed: {_ce}")

        # Closed real trades
        if _closed_rt:
            with st.expander(f"Closed ({len(_closed_rt)})"):
                _rt_closed_rows = []
                for _rt in sorted(_closed_rt, key=lambda x: x.get("exit_date", ""), reverse=True):
                    _pnl = _rt.get("pnl_dollars")
                    _rt_closed_rows.append({
                        "Side"      : "SHORT" if _rt.get("side") == "short" else "LONG",
                        "Ticker"    : _rt["ticker"],
                        "Source"    : _rt.get("source", "scan"),
                        "Entry $"   : f"${float(_rt['entry_price']):.2f}",
                        "Exit $"    : f"${float(_rt['exit_price']):.2f}" if _rt.get("exit_price") else "—",
                        "Shares"    : _rt.get("shares", 0),
                        "Reason"    : _rt.get("exit_reason", "—"),
                        "P&L %"     : f"{_rt.get('pnl_pct', 0):+.1f}%",
                        "P&L $"     : f"${_pnl:+.2f}" if _pnl is not None else "—",
                        "Entry Date": _rt.get("entry_date", ""),
                        "Exit Date" : _rt.get("exit_date", ""),
                    })
                st.dataframe(
                    pd.DataFrame(_rt_closed_rows).style.map(color_pnl, subset=["P&L %", "P&L $"]),
                    hide_index=True, use_container_width=True,
                )

            _rt_with_pnl = [t for t in _closed_rt if t.get("pnl_dollars") is not None]
            if _rt_with_pnl:
                _rt_wins      = sum(1 for t in _rt_with_pnl if t.get("pnl_pct", 0) > 0)
                _rt_total_pnl = sum(t["pnl_dollars"] for t in _rt_with_pnl)
                _rt_avg_pnl   = sum(t.get("pnl_pct", 0) for t in _rt_with_pnl) / len(_rt_with_pnl)
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Real Trades", len(_rt_with_pnl))
                rc2.metric("Win Rate", f"{_rt_wins/len(_rt_with_pnl):.0%}")
                rc3.metric("Avg P&L", f"{_rt_avg_pnl:+.1f}%")
                rc4.metric("Total P&L", f"${_rt_total_pnl:+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM L/S TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_lt:
    st.header("Absolute Return L/S Portfolio")
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
            "longterm", ROOT / "longterm.py"
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
            if _summary_path is None and (ROOT / "ticker_summary.csv").exists():
                _summary_path = ROOT / "ticker_summary.csv"
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
    st.header("S&P Benchmark L/S Portfolio")
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
            "conviction", ROOT / "conviction.py"
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

        # Show auto-replace summary when the scan just fired one
        if "cv_auto_replace_msg" in st.session_state:
            st.success(f"🔄 {st.session_state.pop('cv_auto_replace_msg')}")


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

            # Auto-refresh scores from latest scan or do mini-scan
            _score_map = {}
            if "cv_scan_result" in st.session_state and not st.session_state["cv_scan_result"].empty:
                _scan_df = st.session_state["cv_scan_result"]
                _score_map = dict(zip(_scan_df["ticker"], _scan_df["composite_score"]))
            else:
                # No recent scan, do quick score fetch for open positions only
                try:
                    _mini_scan_df = _cv_mod.run_conviction_scan(_cv_tks)
                    _score_map = dict(zip(_mini_scan_df["ticker"], _mini_scan_df["composite_score"]))
                except Exception:
                    pass

            # Update current_score for each open position
            for _cp in _cv_open:
                if _cp["ticker"] in _score_map:
                    _cp["current_score"] = _score_map[_cp["ticker"]]

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
                _pos_size_entry = round(_shares_held * _ep, 2)
                _pos_size_current = round(_shares_held * _cur, 2)
                _cv_rows.append({
                    "Ticker"      : _cp["ticker"],
                    "Entry Date"  : _cp["entry_date"],
                    "Dir"         : _dir,
                    "Shares"      : f"{_shares_held:.4f}" if _shares_held else "—",
                    "Position Size": f"${_pos_size_entry:,.0f}",
                    "Entry $"     : f"${_ep:.2f}",
                    "Current $"   : f"${_cur:.2f}",
                    "P&L %"       : f"{_pnl:+.2f}%",
                    "P&L $"       : f"${_pnl_d:+.2f}",
                    "Days"        : (date.today() - date.fromisoformat(_cp["entry_date"])).days,
                    "Entry Score" : f"{_cp.get('score_at_entry'):+.3f}" if _cp.get("score_at_entry") is not None else "—",
                    "Score"       : f"{_cp.get('current_score'):+.3f}" if _cp.get("current_score") is not None else "—",
                    "Status"      : _flag,
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

            # Net exposure: (Long $ - Short $) / Total Capital × 100
            _cv_long_entry = sum(_shares_held * _ep for _cp, _shares_held, _ep in
                                [(p, p.get("shares", 0), p["entry_price"]) for p in _cv_open if p["direction"] == "LONG"])
            _cv_short_entry = sum(_shares_held * _ep for _cp, _shares_held, _ep in
                                 [(p, p.get("shares", 0), p["entry_price"]) for p in _cv_open if p["direction"] == "SHORT"])
            _cv_long_current = sum(_shares_held * (_cv_live_price(p["ticker"]) or p["entry_price"]) for p in _cv_open if p["direction"] == "LONG"
                                  for _shares_held in [p.get("shares", 0)])
            _cv_short_current = sum(_shares_held * (_cv_live_price(p["ticker"]) or p["entry_price"]) for p in _cv_open if p["direction"] == "SHORT"
                                   for _shares_held in [p.get("shares", 0)])
            _cv_total_capital = st.session_state.get("cv_budget", 5000)  # default to 5k
            _cv_net_exposure_entry = ((_cv_long_entry - _cv_short_entry) / _cv_total_capital * 100) if _cv_total_capital else 0
            _cv_net_exposure_current = ((_cv_long_current - _cv_short_current) / _cv_total_capital * 100) if _cv_total_capital else 0

            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            _cc1.metric("Total P&L $",   f"${_cv_total_pnl_d:+.2f}")
            _cc2.metric("Weighted P&L %", f"{_cv_wtd_pnl:+.2f}%")
            _cc3.metric("Long P&L $",    f"${_cv_long_pnl:+.2f}")
            _cc4.metric("Short P&L $",   f"${_cv_short_pnl:+.2f}")

            _ce1, _ce2 = st.columns(2)
            _ce1.metric("Net Exposure (Entry)",   f"{_cv_net_exposure_entry:+.1f}%")
            _ce2.metric("Net Exposure (Current)", f"{_cv_net_exposure_current:+.1f}%")

            # ── Capital allocation & position review ───────────────────────────
            # ── Promote open paper positions to Real Trades ───────────────────
            st.caption("🚀 Copy paper positions into Real Trades (paper positions stay untouched).")
            if st.button("🚀 Promote All to Real", key="cv_promote_all"):
                import uuid as _uuid
                _promoted = 0
                for _cp in _cv_open:
                    try:
                        db.add_stock_real_trade({
                            "id"         : str(_uuid.uuid4()),
                            "ticker"     : _cp["ticker"],
                            "side"       : "short" if _cp.get("direction") == "SHORT" else "long",
                            "entry_price": float(_cp["entry_price"]),
                            "entry_date" : _cp.get("entry_date", str(date.today())),
                            "shares"     : float(_cp.get("shares") or 0),
                            "dollars"    : float(_cp.get("cost") or 0),
                            "model_prob" : _cp.get("score_at_entry"),
                            "status"     : "open",
                            "placed_at"  : datetime.now(timezone.utc).isoformat(),
                        }, source="paper_promotion")
                        _promoted += 1
                    except Exception as _pe:
                        st.error(f"Failed to promote {_cp['ticker']}: {_pe}")
                if _promoted:
                    st.success(f"Promoted {_promoted} position(s) to Real Trades.")
                    st.rerun()
            _cv_promoted_tickers = {
                r["ticker"] for r in db.load_stock_real_trades()
                if r.get("status") == "open"
            }
            for _cp in _cv_open:
                _cp_dir = "SHORT" if _cp.get("direction") == "SHORT" else "LONG"
                if _cp["ticker"] in _cv_promoted_tickers:
                    st.caption(f"✅ {_cp['ticker']} already in Real Trades")
                elif st.button(
                    f"🚀 Promote {_cp['ticker']} ({_cp_dir}) to Real",
                    key=f"cv_promote_{_cp['ticker']}",
                ):
                    try:
                        import uuid as _uuid
                        _rt_row = {
                            "id"         : str(_uuid.uuid4()),
                            "ticker"     : _cp["ticker"],
                            "side"       : "short" if _cp.get("direction") == "SHORT" else "long",
                            "entry_price": float(_cp["entry_price"]),
                            "entry_date" : _cp.get("entry_date", str(date.today())),
                            "shares"     : float(_cp.get("shares") or 0),
                            "dollars"    : float(_cp.get("cost") or 0),
                            "model_prob" : _cp.get("score_at_entry"),
                            "status"     : "open",
                            "placed_at"  : datetime.now(timezone.utc).isoformat(),
                        }
                        db.add_stock_real_trade(_rt_row, source="paper_promotion")
                        st.success(f"Copied {_cp['ticker']} to Real Trades.")
                        st.rerun()
                    except Exception as _pe:
                        st.error(f"Promote failed: {_pe}")

            st.divider()
            st.subheader("Capital Allocation & Score Review")

            _cv_deployed = sum(_cv_costs)
            _cv_available = _cv_total_capital - _cv_deployed
            _cpa1, _cpa2, _cpa3 = st.columns(3)
            _cpa1.metric("Budget", f"${_cv_total_capital:,.0f}")
            _cpa2.metric("Deployed", f"${_cv_deployed:,.0f}", delta=f"{_cv_deployed/_cv_total_capital*100:.1f}%")
            _cpa3.metric("Available", f"${_cv_available:,.0f}", delta=f"{_cv_available/_cv_total_capital*100:.1f}%" if _cv_available >= 0 else f"{_cv_available/_cv_total_capital*100:.1f}%")

            # Positions ranked by score (worst first) — helps identify what to close/resize
            _cv_score_review = []
            for i, _cp in enumerate(_cv_open):
                _cur_price = _cv_live_price(_cp["ticker"]) or _cp["entry_price"]
                _current_size = round(_cp.get("shares", 0) * _cur_price, 0)
                _entry_score   = _cp.get("score_at_entry")
                _current_score = _cp.get("current_score")
                _pnl_val = _cv_pnl_dollars[i] if i < len(_cv_pnl_dollars) else 0
                # Status uses actual conviction thresholds so below-threshold
                # positions are immediately visible without running a scan.
                _score_status = "—"
                if _current_score is not None:
                    _dir = _cp["direction"]
                    if _dir == "LONG" and _current_score < _cv_mod.MIN_LONG_SCORE:
                        _score_status = "🚨 Below Threshold"
                    elif _dir == "SHORT" and _current_score > _cv_mod.MIN_SHORT_SCORE:
                        _score_status = "🚨 Below Threshold"
                    elif _dir == "LONG" and _current_score < 0.35:
                        _score_status = "⚠️ Weak"
                    elif _dir == "SHORT" and _current_score > -0.35:
                        _score_status = "⚠️ Weak"
                    else:
                        _score_status = "✓ OK"
                _cv_score_review.append({
                    "Ticker"      : _cp["ticker"],
                    "Dir"         : _cp["direction"],
                    "Size"        : f"${_current_size:,.0f}",
                    "Entry Score" : f"{_entry_score:+.3f}" if _entry_score is not None else "—",
                    "Score"       : f"{_current_score:+.3f}" if _current_score is not None else "—",
                    "P&L"         : f"${_pnl_val:+.0f}",
                    "Status"      : _score_status,
                })

            _cv_score_review = sorted(_cv_score_review, key=lambda x: float(x["Score"].replace("—", "999")) if x["Score"] != "—" else 999.0)
            st.caption("Positions ranked by score (weak scores = good candidates to close/resize)")
            st.dataframe(
                pd.DataFrame(_cv_score_review).style.map(color_pnl, subset=["P&L"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("No open conviction positions. Run a scan to get recommendations.")

        # ── Delete positions ─────────────────────────────────────────────────
        if _cv_open:
            with st.expander("Delete position(s)"):
                _cv_del_tickers = st.multiselect(
                    "Select tickers to delete", [p["ticker"] for p in _cv_open], key="cv_del_sel"
                )
                if _cv_del_tickers and st.button("Delete Selected", key="cv_del_btn"):
                    try:
                        # Delete from Supabase directly
                        _cv_client = db._get_client()
                        if _cv_client:
                            for _tk in _cv_del_tickers:
                                _cv_client.table("lt_positions").delete().filter("ticker", "eq", _tk).filter("id", "like", "cv_%").execute()
                        # Also remove from in-memory list
                        _cv_positions = [p for p in _cv_positions if p["ticker"] not in _cv_del_tickers or p["status"] == "closed"]
                        _cv_mod.save_conviction_positions(_cv_positions)
                        st.success(f"Deleted {', '.join(_cv_del_tickers)}")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Delete error: {_e}")

        # ── Scan controls ─────────────────────────────────────────────────────
        _cv_budget = st.number_input(
            "Portfolio size ($)", min_value=1_000, max_value=1_000_000,
            value=5_000, step=500, key="cv_budget",
        )

        def _cv_do_auto_replace(positions_all, open_positions, scored_df):
            """Assess open positions, auto-close any that fell below entry threshold,
            and paper-add the best available same-direction replacement.
            Returns (new_positions_all, summary_string)."""
            import uuid as _uuid2
            assessed    = _cv_mod.assess_open_positions(open_positions, scored_df)
            open_tix    = {p["ticker"] for p in assessed if p.get("status") == "open"}
            closed_list = []
            replaced_list = []

            for _p in assessed:
                if _p.get("status") != "open":
                    continue
                _sig = _p.get("reassess_signal", "") or ""
                if "threshold" not in _sig:
                    continue  # not a score-threshold exit — leave for manual review

                # ── Auto-close ──
                _cpx = _p.get("current_price") or float(_p["entry_price"])
                _ep  = float(_p["entry_price"])
                _sh  = float(_p.get("shares") or 0)
                _dir = _p["direction"]
                _pnl_pct = ((_cpx - _ep) / _ep * 100) if _dir == "LONG" else ((_ep - _cpx) / _ep * 100)
                _pnl_d   = ((_cpx - _ep) if _dir == "LONG" else (_ep - _cpx)) * _sh
                _p.update({
                    "status"      : "closed",
                    "exit_price"  : _cpx,
                    "exit_date"   : str(date.today()),
                    "exit_reason" : "score_below_threshold",
                    "pnl_pct"     : round(_pnl_pct, 2),
                    "pnl_dollars" : round(_pnl_d, 2),
                })
                closed_list.append(_p["ticker"])
                open_tix.discard(_p["ticker"])

                # ── Find best replacement (same direction, not already held) ──
                _freed = _p.get("cost") or round(_ep * _sh, 2)
                _open_now = sum(1 for _q in assessed if _q.get("status") == "open")
                if _open_now >= _cv_mod.PAPER_MAX_POSITIONS:
                    continue  # still full after this close (shouldn't happen)
                _gross_now = sum(_q.get("cost", 0) for _q in assessed if _q.get("status") == "open")
                if _gross_now + _freed > _cv_mod.PAPER_MAX_GROSS:
                    continue  # would breach gross cap

                if _dir == "LONG":
                    _cands = scored_df[
                        (scored_df["direction"] == "LONG") &
                        (~scored_df["ticker"].isin(open_tix))
                    ].sort_values("composite_score", ascending=False)
                else:
                    _cands = scored_df[
                        (scored_df["direction"] == "SHORT") &
                        (~scored_df["ticker"].isin(open_tix))
                    ].sort_values("composite_score", ascending=True)

                if _cands.empty:
                    continue  # no qualifying candidate available

                _best   = _cands.iloc[0]
                _rep_tk = _best["ticker"]
                _rep_px = float(_best["current_price"])
                _rep_sh = round(_freed / _rep_px, 4) if _rep_px else 0
                assessed.append({
                    "id"                     : f"cv_{_rep_tk}_{str(_uuid2.uuid4())[:8]}",
                    "ticker"                 : _rep_tk,
                    "direction"              : _dir,
                    "status"                 : "open",
                    "entry_price"            : _rep_px,
                    "entry_date"             : str(date.today()),
                    "shares"                 : _rep_sh,
                    "cost"                   : round(_rep_sh * _rep_px, 2),
                    "composite_score"        : float(_best["composite_score"]),
                    "score_at_entry"         : float(_best["composite_score"]),
                    "z_technical_at_entry"   : float(_best["z_technical"])   if pd.notna(_best.get("z_technical"))   else None,
                    "z_fundamental_at_entry" : float(_best["z_fundamental"]) if pd.notna(_best.get("z_fundamental")) else None,
                    "z_earnings_at_entry"    : float(_best["z_earnings"])    if pd.notna(_best.get("z_earnings"))    else None,
                    "z_macro_at_entry"       : float(_best["z_macro"])       if pd.notna(_best.get("z_macro"))       else None,
                    "regime_at_entry"        : str(_best.get("regime", "")),
                    "exit_signal"            : None,
                    "reassess_signal"        : None,
                    "earnings_flag"          : bool(_best.get("earnings_flag", False)),
                    "earnings_days_out"      : _best.get("earnings_days_out"),
                })
                open_tix.add(_rep_tk)
                replaced_list.append(f"{_rep_tk} ({_dir})")

            # For positions with non-threshold signals, still persist the reassess flag
            # (they remain open, just flagged in the UI for manual review)
            closed_hist = [p for p in positions_all if p.get("status") != "open"]
            new_all     = closed_hist + assessed

            parts = []
            if closed_list:
                parts.append(f"Auto-closed: {', '.join(closed_list)}")
            if replaced_list:
                parts.append(f"Replaced with: {', '.join(replaced_list)}")
            return new_all, " | ".join(parts)

        if st.button("Run Conviction Scan", type="primary", key="cv_scan"):
            _summary_path = db.get_stock_file("ticker_summary.csv", ROOT)
            if _summary_path is None and (ROOT / "ticker_summary.csv").exists():
                _summary_path = ROOT / "ticker_summary.csv"
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
                    # Auto-close below-threshold + replace immediately after scan
                    if _cv_open:
                        _cv_positions, _ar_msg = _cv_do_auto_replace(
                            _cv_positions, _cv_open, _cv_df
                        )
                        _cv_mod.save_conviction_positions(_cv_positions)
                        if _ar_msg:
                            st.session_state["cv_auto_replace_msg"] = _ar_msg
                    st.rerun()
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
                        "pct_from_close",
                    ]
                    _cv_display_cols = [c for c in _cv_display_cols if c in _cv_df.columns]
                    _cv_disp = _cv_df[_cv_display_cols].copy()
                    for _pct_col in ["ret_3m", "ret_6m", "roe", "revenue_growth", "beat_rate"]:
                        if _pct_col in _cv_disp.columns:
                            _cv_disp[_pct_col] = _cv_disp[_pct_col].apply(
                                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—"
                            )
                    if "pct_from_close" in _cv_disp.columns:
                        # Already in percent units (not 0–1 fraction).
                        _cv_disp["pct_from_close"] = _cv_disp["pct_from_close"].apply(
                            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
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

                # ── Select-all buttons ───────────────────────────────────────
                _sa_col1, _sa_col2, _sa_col3 = st.columns([2, 2, 3])
                _cv_paper_all = _sa_col1.button("📈 Paper All Recommendations", key="cv_paper_all")
                _cv_real_all  = _sa_col2.button("💰 Real All Recommendations",  key="cv_real_all")

                if _cv_paper_all or _cv_real_all:
                    import uuid as _uuid
                    # Paper open set: skip these on Paper All (idempotent paper add).
                    _cv_paper_open = {p["ticker"] for p in _cv_open}
                    # Real open set: skip these on Real All (idempotent real add — paper
                    # state is irrelevant; we want every recommendation in real, but only
                    # one open real position per ticker).
                    _cv_real_open = {
                        r["ticker"] for r in db.load_stock_real_trades()
                        if r.get("status") == "open"
                    }
                    _paper_added = 0
                    _real_added  = 0
                    # Running counters for paper cap enforcement across the bulk loop
                    _pa_bulk_count = len(_cv_open)
                    _pa_bulk_gross = sum(p.get("cost", 0) for p in _cv_open)
                    # Derive allocations independently (these vars aren't in scope yet)
                    _lcs = _cv_longs["composite_score"].sum() if not _cv_longs.empty else 0
                    _scs = abs(_cv_shorts["composite_score"].sum()) if not _cv_shorts.empty else 0
                    _tc  = _lcs + _scs
                    _alloc_long_all  = _cv_budget_used * (_lcs / _tc) if _tc > 0 else _cv_budget_used / 2
                    _alloc_short_all = _cv_budget_used * (_scs / _tc) if _tc > 0 else _cv_budget_used / 2
                    for _, _lr in _cv_longs.iterrows():
                        _tk = _lr["ticker"]
                        _pl = _alloc_long_all * (_lr["composite_score"] / _lcs) if _lcs > 0 else _alloc_long_all / max(len(_cv_longs), 1)
                        _sh = round(_pl / _lr["current_price"], 4) if _lr["current_price"] else 0
                        # ── Paper add (skip if already open or over caps) ──────
                        if _cv_paper_all and _tk not in _cv_paper_open:
                            _bulk_new_cost = round(_sh * _lr["current_price"], 2)
                            if (_pa_bulk_count < _cv_mod.PAPER_MAX_POSITIONS
                                    and _pa_bulk_gross + _bulk_new_cost <= _cv_mod.PAPER_MAX_GROSS):
                                _cv_positions.append({
                                    "ticker": _tk, "direction": "LONG", "status": "open",
                                    "entry_price": _lr["current_price"], "entry_date": str(date.today()),
                                    "shares": _sh, "cost": _bulk_new_cost,
                                    "composite_score": float(_lr["composite_score"]),
                                    "score_at_entry": float(_lr["composite_score"]),
                                    "z_technical_at_entry"   : float(_lr["z_technical"])   if pd.notna(_lr.get("z_technical"))   else None,
                                    "z_fundamental_at_entry" : float(_lr["z_fundamental"]) if pd.notna(_lr.get("z_fundamental")) else None,
                                    "z_earnings_at_entry"    : float(_lr["z_earnings"])    if pd.notna(_lr.get("z_earnings"))    else None,
                                    "z_macro_at_entry"       : float(_lr["z_macro"])       if pd.notna(_lr.get("z_macro"))       else None,
                                    "regime_at_entry"        : str(_lr.get("regime", "")),
                                    "exit_signal": None, "reassess_signal": None,
                                    "earnings_flag": bool(_lr.get("earnings_flag", False)),
                                    "earnings_days_out": _lr.get("earnings_days_out"),
                                })
                                _paper_added += 1
                                _pa_bulk_count += 1
                                _pa_bulk_gross += _bulk_new_cost
                        # ── Real add (skip if already open in real) ────────────
                        # Independent of paper state — paper rows stay untouched.
                        if _cv_real_all and _tk not in _cv_real_open:
                            db.add_stock_real_trade({
                                "id": str(_uuid.uuid4()), "ticker": _tk, "side": "long",
                                "entry_price": float(_lr["current_price"]), "entry_date": str(date.today()),
                                "shares": float(_sh), "dollars": round(_sh * _lr["current_price"], 2),
                                "model_prob": float(_lr["composite_score"]), "status": "open",
                                "placed_at": datetime.now(timezone.utc).isoformat(),
                            }, source="conviction")
                            _real_added += 1
                    for _, _sr in _cv_shorts.iterrows():
                        _tk = _sr["ticker"]
                        _ps = _alloc_short_all * (abs(_sr["composite_score"]) / _scs) if _scs > 0 else _alloc_short_all / max(len(_cv_shorts), 1)
                        _sh = round(_ps / _sr["current_price"], 4) if _sr["current_price"] else 0
                        if _cv_paper_all and _tk not in _cv_paper_open:
                            _bulk_new_cost_s = round(_sh * _sr["current_price"], 2)
                            if (_pa_bulk_count < _cv_mod.PAPER_MAX_POSITIONS
                                    and _pa_bulk_gross + _bulk_new_cost_s <= _cv_mod.PAPER_MAX_GROSS):
                                _cv_positions.append({
                                    "ticker": _tk, "direction": "SHORT", "status": "open",
                                    "entry_price": _sr["current_price"], "entry_date": str(date.today()),
                                    "shares": _sh, "cost": _bulk_new_cost_s,
                                    "composite_score": float(_sr["composite_score"]),
                                    "score_at_entry": float(_sr["composite_score"]),
                                    "z_technical_at_entry"   : float(_sr["z_technical"])   if pd.notna(_sr.get("z_technical"))   else None,
                                    "z_fundamental_at_entry" : float(_sr["z_fundamental"]) if pd.notna(_sr.get("z_fundamental")) else None,
                                    "z_earnings_at_entry"    : float(_sr["z_earnings"])    if pd.notna(_sr.get("z_earnings"))    else None,
                                    "z_macro_at_entry"       : float(_sr["z_macro"])       if pd.notna(_sr.get("z_macro"))       else None,
                                    "regime_at_entry"        : str(_sr.get("regime", "")),
                                    "exit_signal": None, "reassess_signal": None,
                                    "earnings_flag": bool(_sr.get("earnings_flag", False)),
                                    "earnings_days_out": _sr.get("earnings_days_out"),
                                })
                                _paper_added += 1
                                _pa_bulk_count += 1
                                _pa_bulk_gross += _bulk_new_cost_s
                        if _cv_real_all and _tk not in _cv_real_open:
                            db.add_stock_real_trade({
                                "id": str(_uuid.uuid4()), "ticker": _tk, "side": "short",
                                "entry_price": float(_sr["current_price"]), "entry_date": str(date.today()),
                                "shares": float(_sh), "dollars": round(_sh * _sr["current_price"], 2),
                                "model_prob": float(_sr["composite_score"]), "status": "open",
                                "placed_at": datetime.now(timezone.utc).isoformat(),
                            }, source="conviction")
                            _real_added += 1
                    if _paper_added:
                        _cv_mod.save_conviction_positions(_cv_positions)
                    _msg_parts = []
                    if _cv_paper_all:
                        _msg_parts.append(f"📈 Paper: {_paper_added} added")
                    if _cv_real_all:
                        _msg_parts.append(f"💰 Real: {_real_added} added")
                    st.success(" — ".join(_msg_parts) if _msg_parts else "Nothing to add.")
                    st.rerun()

                # LONG recommendations
                st.subheader("LONG Recommendations")
                if _cv_longs.empty:
                    st.caption("No tickers cleared the long conviction threshold today.")
                else:
                    _cv_existing_paper = {p["ticker"] for p in _cv_open}
                    _cv_existing_real = {
                        r["ticker"] for r in db.load_stock_real_trades()
                        if r.get("status") == "open"
                    }
                    # Backwards compat for code below that still references _cv_existing.
                    _cv_existing = _cv_existing_paper
                    # Conviction-weighted sizing: allocate based on score strength
                    _long_conviction_sum = _cv_longs["composite_score"].sum()
                    _short_conviction_sum = abs(_cv_shorts["composite_score"].sum()) if not _cv_shorts.empty else 0
                    _total_conviction = _long_conviction_sum + _short_conviction_sum
                    if _total_conviction > 0:
                        _alloc_long = _cv_budget_used * (_long_conviction_sum / _total_conviction)
                    else:
                        _alloc_long = _cv_budget_used / 2  # fallback to 50/50 if no conviction

                    for _, _lr in _cv_longs.iterrows():
                        _tk       = _lr["ticker"]
                        _in_paper = _tk in _cv_existing_paper
                        _in_real  = _tk in _cv_existing_real
                        # Conviction-weighted allocation for this position
                        _per_long = _alloc_long * (_lr["composite_score"] / _long_conviction_sum) if _long_conviction_sum > 0 else _alloc_long / max(len(_cv_longs), 1)
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
                            # % from prior daily close — flag with ⚠️ when the stock has
                            # already moved >2% intraday in the unfavorable direction
                            # (up for a long, down for a short).
                            _pfc_l      = _lr.get("pct_from_close", 0.0) or 0.0
                            _prior_cl_l = _lr.get("prior_close") or _lr.get("current_price")
                            _pfc_l_warn = "  ⚠️" if _pfc_l > 2.0 else ""
                            _pfc_l_str  = f" ({_pfc_l:+.2f}%{_pfc_l_warn})" if abs(_pfc_l) >= 0.05 else ""
                            _close_str_l = f" | close ${_prior_cl_l}" if _prior_cl_l else ""
                            st.markdown(f"**{_tk}** @ ${_lr['current_price']}{_pfc_l_str}{_close_str_l}  —  {_score_breakdown}{_earn_note}")
                            st.caption(f"{_shares:.4f} shares · ${_per_long:,.0f} allocation")
                        with _col2:
                            # Paper button: hidden if already held in paper.
                            if _in_paper:
                                st.caption("📈 Paper held")
                                _cv_do_paper = False
                            else:
                                _cv_do_paper = st.button(f"📈 Paper Long {_tk}", key=f"cv_long_{_tk}")
                            # Real button: hidden if already held in real (paper state irrelevant).
                            if _in_real:
                                st.caption("💰 Real held")
                                _cv_do_real = False
                            else:
                                _cv_do_real = st.button(f"💰 Real Long {_tk}", key=f"cv_real_long_{_tk}")

                            if _cv_do_paper:
                                _cv_paper_gross_l = sum(p.get("cost", 0) for p in _cv_open)
                                _new_cost_l = round(_shares * _lr["current_price"], 2)
                                if len(_cv_open) >= _cv_mod.PAPER_MAX_POSITIONS:
                                    st.warning(f"Paper portfolio at max {_cv_mod.PAPER_MAX_POSITIONS} positions — close one first.")
                                elif _cv_paper_gross_l + _new_cost_l > _cv_mod.PAPER_MAX_GROSS:
                                    st.warning(f"Adding {_tk} (${_new_cost_l:,.0f}) would exceed ${_cv_mod.PAPER_MAX_GROSS:,.0f} gross limit (current: ${_cv_paper_gross_l:,.0f}).")
                                else:
                                    _cv_positions.append({
                                        "ticker"                 : _tk,
                                        "direction"              : "LONG",
                                        "status"                 : "open",
                                        "entry_price"            : _lr["current_price"],
                                        "entry_date"             : str(date.today()),
                                        "shares"                 : _shares,
                                        "cost"                   : round(_shares * _lr["current_price"], 2),
                                        "composite_score"        : float(_lr["composite_score"]),
                                        "score_at_entry"         : float(_lr["composite_score"]),
                                        "z_technical_at_entry"   : float(_lr["z_technical"])   if pd.notna(_lr.get("z_technical"))   else None,
                                        "z_fundamental_at_entry" : float(_lr["z_fundamental"]) if pd.notna(_lr.get("z_fundamental")) else None,
                                        "z_earnings_at_entry"    : float(_lr["z_earnings"])    if pd.notna(_lr.get("z_earnings"))    else None,
                                        "z_macro_at_entry"       : float(_lr["z_macro"])       if pd.notna(_lr.get("z_macro"))       else None,
                                        "regime_at_entry"        : str(_lr.get("regime", "")),
                                        "exit_signal"            : None,
                                        "reassess_signal"        : None,
                                        "earnings_flag"          : bool(_eflag),
                                        "earnings_days_out"      : _edays,
                                    })
                                    _cv_mod.save_conviction_positions(_cv_positions)
                                    st.success(f"📈 Paper LONG {_tk} @ ${_lr['current_price']}")
                                    st.rerun()
                            elif _cv_do_real:
                                import uuid as _uuid
                                db.add_stock_real_trade({
                                    "id"         : str(_uuid.uuid4()),
                                    "ticker"     : _tk,
                                    "side"       : "long",
                                    "entry_price": float(_lr["current_price"]),
                                    "entry_date" : str(date.today()),
                                    "shares"     : float(_shares),
                                    "dollars"    : round(_shares * _lr["current_price"], 2),
                                    "model_prob" : float(_lr["composite_score"]),
                                    "status"     : "open",
                                    "placed_at"  : datetime.now(timezone.utc).isoformat(),
                                }, source="conviction")
                                st.success(f"💰 Real LONG {_tk} @ ${_lr['current_price']} (paper untouched)")
                                st.rerun()

                # SHORT recommendations
                st.subheader("SHORT Recommendations")
                if _cv_shorts.empty:
                    st.caption("No tickers cleared the short conviction threshold today.")
                else:
                    # Conviction-weighted sizing for shorts: use absolute value of scores
                    if _short_conviction_sum > 0:
                        _alloc_short = _cv_budget_used * (_short_conviction_sum / _total_conviction)
                    else:
                        _alloc_short = _cv_budget_used / 2  # fallback to 50/50 if no conviction

                    for _, _sr in _cv_shorts.iterrows():
                        _tk       = _sr["ticker"]
                        # Reuse the held-sets computed in the LONG block; if shorts are
                        # rendered without longs (longs empty) the sets won't exist yet.
                        if "_cv_existing_paper" not in dir():
                            _cv_existing_paper = {p["ticker"] for p in _cv_open}
                            _cv_existing_real = {
                                r["ticker"] for r in db.load_stock_real_trades()
                                if r.get("status") == "open"
                            }
                        _in_paper = _tk in _cv_existing_paper
                        _in_real  = _tk in _cv_existing_real
                        # Conviction-weighted allocation for this position
                        _per_short = _alloc_short * (abs(_sr["composite_score"]) / _short_conviction_sum) if _short_conviction_sum > 0 else _alloc_short / max(len(_cv_shorts), 1)
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
                            # % from prior daily close — for shorts, the unfavorable
                            # direction is DOWN (price already gave us alpha intraday).
                            _pfc_s      = _sr.get("pct_from_close", 0.0) or 0.0
                            _prior_cl_s = _sr.get("prior_close") or _sr.get("current_price")
                            _pfc_s_warn = "  ⚠️" if _pfc_s < -2.0 else ""
                            _pfc_s_str  = f" ({_pfc_s:+.2f}%{_pfc_s_warn})" if abs(_pfc_s) >= 0.05 else ""
                            _close_str_s = f" | close ${_prior_cl_s}" if _prior_cl_s else ""
                            st.markdown(f"**{_tk}** @ ${_sr['current_price']}{_pfc_s_str}{_close_str_s}  —  {_score_breakdown}{_earn_note}")
                            st.caption(f"{_shares:.4f} shares · ${_per_short:,.0f} allocation")
                        with _col2:
                            if _in_paper:
                                st.caption("📉 Paper held")
                                _cv_do_paper_s = False
                            else:
                                _cv_do_paper_s = st.button(f"📉 Paper Short {_tk}", key=f"cv_short_{_tk}")
                            if _in_real:
                                st.caption("💰 Real held")
                                _cv_do_real_s = False
                            else:
                                _cv_do_real_s = st.button(f"💰 Real Short {_tk}", key=f"cv_real_short_{_tk}")

                            if _cv_do_paper_s:
                                _cv_paper_gross_s = sum(p.get("cost", 0) for p in _cv_open)
                                _new_cost_s = round(_shares * _sr["current_price"], 2)
                                if len(_cv_open) >= _cv_mod.PAPER_MAX_POSITIONS:
                                    st.warning(f"Paper portfolio at max {_cv_mod.PAPER_MAX_POSITIONS} positions — close one first.")
                                elif _cv_paper_gross_s + _new_cost_s > _cv_mod.PAPER_MAX_GROSS:
                                    st.warning(f"Adding {_tk} (${_new_cost_s:,.0f}) would exceed ${_cv_mod.PAPER_MAX_GROSS:,.0f} gross limit (current: ${_cv_paper_gross_s:,.0f}).")
                                else:
                                    _cv_positions.append({
                                        "ticker"                 : _tk,
                                        "direction"              : "SHORT",
                                        "status"                 : "open",
                                        "entry_price"            : _sr["current_price"],
                                        "entry_date"             : str(date.today()),
                                        "shares"                 : _shares,
                                        "cost"                   : round(_shares * _sr["current_price"], 2),
                                        "composite_score"        : float(_sr["composite_score"]),
                                        "score_at_entry"         : float(_sr["composite_score"]),
                                        "z_technical_at_entry"   : float(_sr["z_technical"])   if pd.notna(_sr.get("z_technical"))   else None,
                                        "z_fundamental_at_entry" : float(_sr["z_fundamental"]) if pd.notna(_sr.get("z_fundamental")) else None,
                                        "z_earnings_at_entry"    : float(_sr["z_earnings"])    if pd.notna(_sr.get("z_earnings"))    else None,
                                        "z_macro_at_entry"       : float(_sr["z_macro"])       if pd.notna(_sr.get("z_macro"))       else None,
                                        "regime_at_entry"        : str(_sr.get("regime", "")),
                                        "exit_signal"            : None,
                                        "reassess_signal"        : None,
                                        "earnings_flag"          : bool(_eflag),
                                        "earnings_days_out"      : _edays,
                                    })
                                    _cv_mod.save_conviction_positions(_cv_positions)
                                    st.success(f"📉 Paper SHORT {_tk} @ ${_sr['current_price']}")
                                    st.rerun()
                            elif _cv_do_real_s:
                                import uuid as _uuid
                                db.add_stock_real_trade({
                                    "id"         : str(_uuid.uuid4()),
                                    "ticker"     : _tk,
                                    "side"       : "short",
                                    "entry_price": float(_sr["current_price"]),
                                    "entry_date" : str(date.today()),
                                    "shares"     : float(_shares),
                                    "dollars"    : round(_shares * _sr["current_price"], 2),
                                    "model_prob" : float(_sr["composite_score"]),
                                    "status"     : "open",
                                    "placed_at"  : datetime.now(timezone.utc).isoformat(),
                                }, source="conviction")
                                st.success(f"💰 Real SHORT {_tk} @ ${_sr['current_price']} (paper untouched)")
                                st.rerun()

                # Deployment summary
                st.divider()
                _total_deployed = _alloc_long + _alloc_short
                _available      = _cv_budget_used - _total_deployed
                _dc1, _dc2, _dc3, _dc4 = st.columns(4)
                _dc1.metric("Budget", f"${_cv_budget_used:,.0f}")
                _dc2.metric("Deployed (Long)", f"${_alloc_long:,.0f}")
                _dc3.metric("Deployed (Short)", f"${_alloc_short:,.0f}")
                _dc4.metric("Available", f"${_available:,.0f}", delta=f"{_available/_cv_budget_used*100:.1f}%" if _cv_budget_used else None)

                # ── Portfolio Actions ─────────────────────────────────────────
                # For each open paper position that fell off the scan
                # recommendations, surface a Replace or Resize action.
                if _cv_open:
                    _pa_score_map = (
                        dict(zip(_cv_df["ticker"], _cv_df["composite_score"]))
                        if "composite_score" in _cv_df.columns else {}
                    )
                    _pa_dir_map = (
                        dict(zip(_cv_df["ticker"], _cv_df["direction"].fillna("")))
                        if "direction" in _cv_df.columns else {}
                    )
                    _pa_price_map = (
                        dict(zip(_cv_df["ticker"], _cv_df["current_price"]))
                        if "current_price" in _cv_df.columns else {}
                    )

                    _pa_replace = []
                    _pa_resize  = []
                    for _p in _cv_open:
                        _ptk       = _p["ticker"]
                        _pdir      = _p["direction"]
                        _psc_entry = float(_p.get("score_at_entry") or _p.get("composite_score") or 0)
                        _new_score = _pa_score_map.get(_ptk)
                        _new_dir   = _pa_dir_map.get(_ptk, "")

                        if _new_score is None:
                            _pa_replace.append({
                                "pos": _p, "new_score": None,
                                "reason": "ticker absent from scan universe",
                            })
                        elif _pdir == "LONG" and _new_dir != "LONG":
                            _pa_replace.append({
                                "pos": _p, "new_score": _new_score,
                                "reason": f"score {_new_score:+.3f} dropped below LONG threshold",
                            })
                        elif _pdir == "SHORT" and _new_dir != "SHORT":
                            _pa_replace.append({
                                "pos": _p, "new_score": _new_score,
                                "reason": f"score {_new_score:+.3f} rose above SHORT threshold",
                            })
                        elif _psc_entry != 0 and _new_score is not None and abs(_new_score) < abs(_psc_entry) * 0.70:
                            _pa_resize.append({
                                "pos": _p, "new_score": _new_score, "entry_score": _psc_entry,
                                "reason": (
                                    f"score {_new_score:+.3f} vs entry {_psc_entry:+.3f} "
                                    f"(weakened {(1 - abs(_new_score) / abs(_psc_entry)) * 100:.0f}%)"
                                ),
                            })

                    if _pa_replace or _pa_resize:
                        st.divider()
                        st.subheader("Portfolio Actions")
                        _held_tickers = {p["ticker"] for p in _cv_open}

                        if _pa_replace:
                            st.markdown("**Replace** — score dropped below threshold or ticker left scan universe")
                            _repl_claimed = set()   # tickers already assigned as replacements this pass
                            for _pa_item in _pa_replace:
                                _pp    = _pa_item["pos"]
                                _ptk   = _pp["ticker"]
                                _pdir  = _pp["direction"]
                                # Exclude: currently-held conviction tickers + already-claimed replacements.
                                # Use head(10) so enough non-held options appear even when many are held.
                                _exclude = _held_tickers | _repl_claimed
                                _pool = (
                                    _cv_df[
                                        (_cv_df["direction"].fillna("") == _pdir)
                                        & (~_cv_df["ticker"].isin(_exclude))
                                    ].head(10)
                                    if "direction" in _cv_df.columns else pd.DataFrame()
                                )
                                _pool_tickers = _pool["ticker"].tolist() if not _pool.empty else []
                                # Resolve current price: scan map → live fetch → entry price fallback.
                                # Must guard NaN: float(NaN) is truthy in Python so plain `or` won't fall through.
                                _raw_p = _pa_price_map.get(_ptk)
                                if _raw_p is None or (isinstance(_raw_p, float) and (pd.isna(_raw_p) or _raw_p <= 0)):
                                    _raw_p = _cv_live_price(_ptk) or _pp.get("entry_price") or 0
                                _cur_p = float(_raw_p) if _raw_p else float(_pp.get("entry_price") or 0)
                                _ep    = float(_pp.get("entry_price") or _cur_p or 1)
                                _pnl_r_raw = ((_cur_p - _ep) / _ep if _pdir == "LONG" else (_ep - _cur_p) / _ep) if _ep else 0
                                _pnl_r = 0.0 if (pd.isna(_pnl_r_raw) if isinstance(_pnl_r_raw, float) else False) else _pnl_r_raw
                                _pnl_d = round(_pnl_r * float(_pp.get("cost") or 0), 2)
                                with st.container(border=True):
                                    _rc1, _rc2 = st.columns([3, 1])
                                    with _rc1:
                                        st.markdown(f"**{_ptk}** ({_pdir}) — {_pa_item['reason']}")
                                        if _pool_tickers:
                                            _repl_sel = st.selectbox(
                                                "Replace with", _pool_tickers,
                                                key=f"pa_repl_sel_{_ptk}",
                                            )
                                            if _repl_sel:
                                                _repl_claimed.add(_repl_sel)
                                        else:
                                            st.caption("No replacement candidates available.")
                                            _repl_sel = None
                                    with _rc2:
                                        _do_replace = bool(_repl_sel) and st.button(f"Replace {_ptk}", key=f"pa_repl_{_ptk}")
                                        _do_close   = (not _repl_sel) and st.button(f"Close {_ptk}", key=f"pa_close_{_ptk}")
                                    if _do_replace or _do_close:
                                        import uuid as _uuid
                                        for _i2, _ex2 in enumerate(_cv_positions):
                                            if _ex2.get("ticker") == _ptk and _ex2.get("status") == "open":
                                                _cv_positions[_i2] = {
                                                    **_ex2, "status": "closed",
                                                    "exit_price": _cur_p,
                                                    "exit_date": str(date.today()),
                                                    "exit_reason": "reassessment",
                                                    "pnl_pct": round(_pnl_r * 100, 2),
                                                    "pnl_dollars": _pnl_d,
                                                }
                                                break
                                        # Also close any open real trade for this ticker
                                        _real_closed = []
                                        for _rt in db.load_stock_real_trades():
                                            if _rt.get("ticker") == _ptk and _rt.get("status") == "open":
                                                try:
                                                    db.close_stock_real_trade(
                                                        _rt["id"], _cur_p,
                                                        str(date.today()), "reassessment",
                                                    )
                                                    _real_closed.append(_ptk)
                                                except Exception as _rce:
                                                    st.warning(f"Could not auto-close real {_ptk}: {_rce}")
                                        if _do_replace:
                                            _nr     = _pool[_pool["ticker"] == _repl_sel].iloc[0]
                                            _cost_r = float(_pp.get("cost") or 0)
                                            _sh_r   = round(_cost_r / float(_nr["current_price"]), 4) if _nr["current_price"] else 0
                                            _cv_positions.append({
                                                "ticker"          : _repl_sel,
                                                "direction"       : _pdir,
                                                "status"          : "open",
                                                "entry_price"     : float(_nr["current_price"]),
                                                "entry_date"      : str(date.today()),
                                                "shares"          : _sh_r,
                                                "cost"            : round(_sh_r * float(_nr["current_price"]), 2),
                                                "composite_score" : float(_nr["composite_score"]),
                                                "score_at_entry"  : float(_nr["composite_score"]),
                                                "exit_signal"     : None,
                                                "reassess_signal" : None,
                                                "earnings_flag"   : bool(_nr.get("earnings_flag", False)),
                                                "earnings_days_out": _nr.get("earnings_days_out"),
                                            })
                                        _cv_mod.save_conviction_positions(_cv_positions)
                                        _msg = f"Replaced {_ptk} → {_repl_sel}" if _do_replace else f"Closed {_ptk}"
                                        if _real_closed:
                                            _msg += f" (real trade also closed @ ${_cur_p:.2f})"
                                        st.success(_msg)
                                        st.rerun()

                        if _pa_resize:
                            st.markdown("**Resize** — conviction weakened >30% but still above threshold")
                            for _pa_item in _pa_resize:
                                _pp         = _pa_item["pos"]
                                _ptk        = _pp["ticker"]
                                _pdir       = _pp["direction"]
                                _entry_sc   = abs(_pa_item["entry_score"])
                                _new_sc     = abs(_pa_item["new_score"])
                                _trim_ratio = max(0.0, min(1.0 - (_new_sc / _entry_sc), 0.90)) if _entry_sc else 0.0
                                _new_shares = round(float(_pp.get("shares") or 0) * (1 - _trim_ratio), 4)
                                _new_cost   = round(_new_shares * float(_pp.get("entry_price") or 0), 2)
                                with st.container(border=True):
                                    _rz1, _rz2 = st.columns([3, 1])
                                    with _rz1:
                                        st.markdown(f"**{_ptk}** ({_pdir}) — {_pa_item['reason']}")
                                        st.caption(f"Trim ~{_trim_ratio*100:.0f}% → {_new_shares:.4f} shares (${_new_cost:,.0f})")
                                    with _rz2:
                                        if st.button(f"Resize {_ptk}", key=f"pa_resize_{_ptk}"):
                                            for _i3, _ex3 in enumerate(_cv_positions):
                                                if _ex3.get("ticker") == _ptk and _ex3.get("status") == "open":
                                                    _cv_positions[_i3] = {
                                                        **_ex3,
                                                        "shares"          : _new_shares,
                                                        "cost"            : _new_cost,
                                                        "composite_score" : float(_pa_item["new_score"]),
                                                    }
                                                    break
                                            _cv_mod.save_conviction_positions(_cv_positions)
                                            st.success(f"Resized {_ptk} — trimmed {_trim_ratio*100:.0f}%")
                                            st.rerun()

        # ── Reassess open positions ───────────────────────────────────────────
        if _cv_open and "cv_scan_result" in st.session_state:
            if st.button("Reassess Open Positions", key="cv_reassess"):
                _cv_positions, _ar_msg = _cv_do_auto_replace(
                    _cv_positions, _cv_open, st.session_state["cv_scan_result"]
                )
                _cv_mod.save_conviction_positions(_cv_positions)
                if _ar_msg:
                    st.success(_ar_msg)
                else:
                    st.success("Positions reassessed — all scores within thresholds.")
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

        # ── Resize a position ──────────────────────────────────────────────────
        if _cv_open:
            with st.expander("Resize position"):
                _cv_resize_ticker = st.selectbox(
                    "Ticker", [p["ticker"] for p in _cv_open], key="cv_resize_sel"
                )
                # Find current position size
                _cv_resize_current = next((p for p in _cv_open if p["ticker"] == _cv_resize_ticker), None)
                if _cv_resize_current:
                    _cv_current_size = _cv_resize_current.get("cost", _cv_resize_current["entry_price"] * _cv_resize_current.get("shares", 0))

                    _cv_resize_method = st.radio(
                        "Resize by",
                        ["Dollar amount", "Percentage reduction"],
                        key="cv_resize_method"
                    )

                    if _cv_resize_method == "Dollar amount":
                        _cv_new_size = st.number_input(
                            f"New position size (current: ${_cv_current_size:,.0f})",
                            min_value=0.0,
                            value=_cv_current_size * 0.75,
                            step=50.0,
                            key="cv_resize_dollars"
                        )
                    else:
                        _cv_reduce_pct = st.slider(
                            "Reduce position by (%)",
                            min_value=0,
                            max_value=100,
                            value=25,
                            step=5,
                            key="cv_resize_pct"
                        )
                        _cv_new_size = _cv_current_size * (1 - _cv_reduce_pct / 100)

                    st.caption(f"Current: ${_cv_current_size:,.0f} → New: ${_cv_new_size:,.0f}")

                    if st.button("Resize Position", key="cv_resize_btn"):
                        for _cp in _cv_positions:
                            if _cp["ticker"] == _cv_resize_ticker and _cp["status"] == "open":
                                _ep = _cp["entry_price"]
                                _old_shares = _cp.get("shares", 0)
                                _new_shares = round(_cv_new_size / _ep, 4) if _ep > 0 else 0
                                _cp.update({
                                    "shares": _new_shares,
                                    "cost": round(_cv_new_size, 2),
                                })
                                break
                        _cv_mod.save_conviction_positions(_cv_positions)
                        st.success(f"Resized {_cv_resize_ticker} to ${_cv_new_size:,.0f}")
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

        # ── Weekly performance tracking ───────────────────────────────────────────
        st.divider()
        st.subheader("Weekly Performance vs S&P 500")

        _cv_weekly_perf = _cv_mod.calculate_weekly_performance(_cv_positions)
        if _cv_weekly_perf is None:
            st.info("Need positions to calculate weekly performance.")
        else:
            # Build weekly performance table
            _cv_weekly_rows = []
            for i, week in enumerate(_cv_weekly_perf["weeks"]):
                port_ret = _cv_weekly_perf["portfolio_returns"][i]
                spy_ret = _cv_weekly_perf["spy_returns"][i]
                outperf = port_ret - spy_ret
                _cv_weekly_rows.append({
                    "Week": week,
                    "Portfolio %": f"{port_ret*100:+.2f}%",
                    "S&P 500 %": f"{spy_ret*100:+.2f}%",
                    "Outperformance %": f"{outperf*100:+.2f}%",
                })

            st.dataframe(
                pd.DataFrame(_cv_weekly_rows).style.map(
                    color_pnl,
                    subset=["Portfolio %", "S&P 500 %", "Outperformance %"]
                ),
                hide_index=True, use_container_width=True,
            )

            # Cumulative returns chart
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter

            fig, ax = plt.subplots(figsize=(10, 4))

            # Convert week dates to datetime for plotting
            week_dates = pd.to_datetime(_cv_weekly_perf["weeks"])
            portfolio_cum = [x * 100 for x in _cv_weekly_perf["cumulative_portfolio"]]
            spy_cum = [x * 100 for x in _cv_weekly_perf["cumulative_spy"]]

            ax.plot(week_dates, portfolio_cum, marker="o", linewidth=2.5, label="Portfolio",
                   color="#3498db", markersize=6)
            ax.plot(week_dates, spy_cum, marker="s", linewidth=2.5, label="S&P 500",
                   color="#95a5a6", markersize=6)

            ax.set_xlabel("Week", fontsize=11, color="white")
            ax.set_ylabel("Cumulative Return %", fontsize=11, color="white")
            ax.legend(loc="best", framealpha=0.9, fontsize=10)
            ax.grid(True, alpha=0.2, color="white")
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="white", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("white")

            st.pyplot(fig)
            plt.close(fig)

        # ── Real Trade Tracker (conviction) ───────────────────────────────────
        st.divider()
        st.header("S&P Benchmark — Real Trade Tracker")
        st.caption("Real money trades for the conviction model. Manual entry/close — no auto-settlement.")

        # ── Add a real trade manually ─────────────────────────────────────────
        with st.expander("➕ Add Real Trade"):
            with st.form("cvr_add_trade"):
                _add_c1, _add_c2 = st.columns(2)
                _add_ticker = _add_c1.text_input("Ticker").upper().strip()
                _add_side   = _add_c2.selectbox("Direction", ["Long", "Short"])
                _add_c3, _add_c4, _add_c5 = st.columns(3)
                _add_ep     = _add_c3.number_input("Entry price ($)", min_value=0.01, value=100.0, step=0.01)
                _add_shares = _add_c4.number_input("Shares", min_value=0.0001, value=1.0, step=0.0001, format="%.4f")
                _add_date   = _add_c5.date_input("Entry date", value=date.today())
                _add_notes  = st.text_input("Notes (optional)")
                if st.form_submit_button("Add Trade"):
                    if not _add_ticker:
                        st.error("Ticker is required.")
                    else:
                        import uuid as _uuid
                        db.add_stock_real_trade({
                            "id"         : str(_uuid.uuid4()),
                            "ticker"     : _add_ticker,
                            "side"       : _add_side.lower(),
                            "entry_price": float(_add_ep),
                            "entry_date" : _add_date.isoformat(),
                            "shares"     : float(_add_shares),
                            "dollars"    : round(float(_add_ep) * float(_add_shares), 2),
                            "model_prob" : None,
                            "status"     : "open",
                            "placed_at"  : datetime.now(timezone.utc).isoformat(),
                        }, source="manual")
                        st.success(f"Added {_add_side} {_add_ticker} @ ${_add_ep:.2f}")
                        st.rerun()

        # Conviction-model trades only — fresh adds (source='conviction') and
        # promotions from this tab's paper book (source='paper_promotion').
        # Per-ticker probability-model trades have source='scan' and live on
        # the Stocks Dashboard tab tracker.
        _cv_real = [t for t in db.load_stock_real_trades()
                    if t.get("source") in ("conviction", "paper_promotion", "manual")]
        _cv_real_open   = [t for t in _cv_real if t.get("status") == "open"]
        _cv_real_closed = [t for t in _cv_real if t.get("status") == "closed"]

        if not _cv_real:
            st.info("No real trades yet. Use 💰 Real Long/Short on a scan result, or 🚀 Promote a paper position.")
        else:
            # Live prices for open real trades
            _cvr_prices: dict[str, float | None] = {}
            if _cv_real_open:
                import yfinance as _yf_cvr
                _cvr_tks = list({t["ticker"] for t in _cv_real_open})
                try:
                    _cvr_px = _yf_cvr.download(_cvr_tks, period="5d", auto_adjust=True, progress=False)
                    if isinstance(_cvr_px.columns, pd.MultiIndex):
                        _cvr_cls = _cvr_px["Close"]
                    else:
                        _cvr_cls = _cvr_px[["Close"]]
                        _cvr_cls.columns = _cvr_tks
                    for _tk in _cvr_tks:
                        try:
                            _cvr_prices[_tk] = float(_cvr_cls[_tk].dropna().iloc[-1])
                        except Exception:
                            _cvr_prices[_tk] = None
                except Exception:
                    _cvr_prices = {tk: None for tk in _cvr_tks}

            if _cv_real_open:
                st.subheader(f"Open ({len(_cv_real_open)})")
                st.caption("Click any cell in the white columns to edit. Greyed columns are read-only.")
                _cvr_trade_ids   = []
                _cvr_edit_rows   = []
                _cvr_pnl_dollars = []
                _cvr_costs       = []
                _cvr_long_pnl    = 0.0
                _cvr_short_pnl   = 0.0
                _cvr_long_entry  = 0.0
                _cvr_short_entry = 0.0
                _cvr_long_curr   = 0.0
                _cvr_short_curr  = 0.0
                for _rt in _cv_real_open:
                    _ep  = float(_rt["entry_price"])
                    _sh  = float(_rt.get("shares") or 0)
                    _cur = _cvr_prices.get(_rt["ticker"])
                    _is_short = _rt.get("side") == "short"
                    if _cur and _ep > 0:
                        _pnl_pct = ((_ep - _cur) / _ep * 100) if _is_short else ((_cur - _ep) / _ep * 100)
                        _pnl_d   = round(((_ep - _cur) if _is_short else (_cur - _ep)) * _sh, 2)
                    else:
                        _pnl_pct, _pnl_d = 0.0, 0.0
                    try:
                        _ed  = date.fromisoformat(_rt.get("entry_date", ""))
                        _days = sum(1 for _d in pd.bdate_range(_ed, date.today()) if _d.date() > _ed)
                        _entry_date_val = _ed
                    except Exception:
                        _days = 0
                        _entry_date_val = date.today()
                    _cvr_trade_ids.append(_rt["id"])
                    _cvr_edit_rows.append({
                        "Dir"        : "SHORT" if _is_short else "LONG",
                        "Ticker"     : _rt["ticker"],
                        "Entry $"    : _ep,
                        "Shares"     : _sh,
                        "Entry Date" : _entry_date_val,
                        "Source"     : _rt.get("source", "—"),
                        "Cur $"      : _cur,
                        "Days"       : _days,
                        "P&L %"      : round(_pnl_pct, 2),
                        "P&L $"      : round(_pnl_d, 2),
                        "Invested $" : round(_ep * _sh, 2),
                    })
                    # Aggregates
                    _cost = _ep * _sh
                    _curv = (_cur or _ep) * _sh
                    _cvr_costs.append(_cost)
                    _cvr_pnl_dollars.append(_pnl_d)
                    if _is_short:
                        _cvr_short_pnl   += _pnl_d
                        _cvr_short_entry += _cost
                        _cvr_short_curr  += _curv
                    else:
                        _cvr_long_pnl    += _pnl_d
                        _cvr_long_entry  += _cost
                        _cvr_long_curr   += _curv

                _real_trade_editor(_cvr_edit_rows, _cvr_trade_ids, "Dir", "cvr_open_editor", "cvr_save_edits")

                # ── Stocks-style P&L summary (mirrors the paper-portfolio block) ──
                _cvr_total_pnl  = sum(_cvr_pnl_dollars)
                _cvr_total_cost = sum(_cvr_costs)
                _cvr_wtd_pnl    = (_cvr_total_pnl / _cvr_total_cost * 100) if _cvr_total_cost else 0
                _cvr_capital    = st.session_state.get("cv_budget", _cvr_total_cost or 5000)
                _cvr_net_entry  = ((_cvr_long_entry - _cvr_short_entry) / _cvr_capital * 100) if _cvr_capital else 0
                _cvr_net_curr   = ((_cvr_long_curr  - _cvr_short_curr)  / _cvr_capital * 100) if _cvr_capital else 0

                _crm1, _crm2, _crm3, _crm4 = st.columns(4)
                _crm1.metric("Total P&L $",   f"${_cvr_total_pnl:+.2f}")
                _crm2.metric("Weighted P&L %", f"{_cvr_wtd_pnl:+.2f}%")
                _crm3.metric("Long P&L $",    f"${_cvr_long_pnl:+.2f}")
                _crm4.metric("Short P&L $",   f"${_cvr_short_pnl:+.2f}")

                _cre1, _cre2 = st.columns(2)
                _cre1.metric("Net Exposure (Entry)",   f"{_cvr_net_entry:+.1f}%")
                _cre2.metric("Net Exposure (Current)", f"{_cvr_net_curr:+.1f}%")

                st.divider()
                st.caption("Close a real trade with the actual fill price:")
                for _rt in _cv_real_open:
                    _dir_lbl = "SHORT" if _rt.get("side") == "short" else "LONG"
                    _trade_label = f"{_rt['ticker']} ({_dir_lbl}) — entered ${float(_rt['entry_price']):.2f}"
                    with st.expander(f"✅ Close {_trade_label}"):
                        with st.form(f"close_cvr_{_rt['id']}"):
                            _cvr_exit_px = st.number_input(
                                "Exit price ($)", min_value=0.0, step=0.01,
                                value=float(_cvr_prices.get(_rt["ticker"]) or _rt["entry_price"]),
                            )
                            _cvr_exit_dt = st.date_input("Exit date", value=date.today())
                            _cvr_reason  = st.selectbox(
                                "Reason", ["target", "stop", "reassessment", "hard_stop", "manual", "other"]
                            )
                            _cvr_notes = st.text_input("Notes (optional)")
                            if st.form_submit_button("Close trade"):
                                try:
                                    db.close_stock_real_trade(
                                        _rt["id"], float(_cvr_exit_px),
                                        _cvr_exit_dt.isoformat(), _cvr_reason,
                                        notes=_cvr_notes or None,
                                    )
                                    st.success(f"Closed {_rt['ticker']} @ ${_cvr_exit_px:.2f}")
                                    st.rerun()
                                except Exception as _ce:
                                    st.error(f"Close failed: {_ce}")

            if _cv_real_closed:
                with st.expander(f"Closed ({len(_cv_real_closed)})"):
                    st.caption("Click any white cell to edit. Exit $, Exit Date, and Reason are editable.")
                    _cvrc_rows = []
                    _cvrc_ids  = []
                    for _rt in sorted(_cv_real_closed, key=lambda x: x.get("exit_date", ""), reverse=True):
                        _pnl   = _rt.get("pnl_dollars")
                        _pnlp  = _rt.get("pnl_pct", 0) or 0
                        _ep_v  = float(_rt["entry_price"]) if _rt.get("entry_price") else 0.0
                        _ex_v  = float(_rt["exit_price"])  if _rt.get("exit_price")  else 0.0
                        try:
                            _ex_date_v = date.fromisoformat(_rt.get("exit_date", "")) if _rt.get("exit_date") else date.today()
                            _en_date_v = date.fromisoformat(_rt.get("entry_date", "")) if _rt.get("entry_date") else date.today()
                        except Exception:
                            _ex_date_v = date.today()
                            _en_date_v = date.today()
                        _cvrc_ids.append(_rt["id"])
                        _cvrc_rows.append({
                            "Dir"        : "SHORT" if _rt.get("side") == "short" else "LONG",
                            "Ticker"     : _rt["ticker"],
                            "Source"     : _rt.get("source", "—"),
                            "Entry $"    : _ep_v,
                            "Exit $"     : _ex_v,
                            "Shares"     : float(_rt.get("shares") or 0),
                            "Reason"     : _rt.get("exit_reason", ""),
                            "P&L %"      : round(_pnlp, 2),
                            "P&L $"      : round(_pnl, 2) if _pnl is not None else 0.0,
                            "Entry Date" : _en_date_v,
                            "Exit Date"  : _ex_date_v,
                        })
                    _closed_trade_editor(_cvrc_rows, _cvrc_ids, "cvrc_closed_editor", "cvrc_closed_save")

                # Realized P&L from closed real trades (stocks-style, no win rate).
                _cvr_realized = [t for t in _cv_real_closed if t.get("pnl_dollars") is not None]
                if _cvr_realized:
                    _cvr_realized_total  = sum(t["pnl_dollars"] for t in _cvr_realized)
                    _cvr_realized_long   = sum(t["pnl_dollars"] for t in _cvr_realized
                                               if t.get("side") != "short")
                    _cvr_realized_short  = sum(t["pnl_dollars"] for t in _cvr_realized
                                               if t.get("side") == "short")
                    _crc1, _crc2, _crc3 = st.columns(3)
                    _crc1.metric("Realized Total P&L $", f"${_cvr_realized_total:+.2f}")
                    _crc2.metric("Realized Long P&L $",  f"${_cvr_realized_long:+.2f}")
                    _crc3.metric("Realized Short P&L $", f"${_cvr_realized_short:+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# OVERNIGHT DRIFT TAB
# ══════════════════════════════════════════════════════════════════════════════
import overnight as _on_mod  # noqa: E402

with tab_overnight:
    st.header("🌙 Overnight Drift Scanner")
    st.markdown(
        "Stocks often earn positive returns **close → open** while losing intraday "
        "(**open → close**). This scanner quantifies that edge per ticker so you can "
        "find names where the drift is statistically significant and survives transaction costs."
    )

    # ── Configuration ─────────────────────────────────────────────────────────
    _on_c1, _on_c2, _on_c3 = st.columns(3)
    with _on_c1:
        _on_period = st.selectbox("Lookback period", list(_on_mod.PERIOD_DAYS.keys()), index=3)
    with _on_c2:
        _on_cost = st.slider("Round-trip cost (bps)", min_value=0, max_value=20, value=6,
                             help="Total bps for one buy + one sell (e.g. spread + commission). "
                                  "6 bps ≈ $0.006/share on a $100 stock.")
    with _on_c3:
        _on_cost_side = _on_cost / 2
        st.metric("Cost per side", f"{_on_cost_side:.1f} bps")

    # Ticker universe — default to the same tickers used in the conviction scan
    _on_summary = ROOT / "ticker_summary.csv"
    _on_universe = (
        pd.read_csv(_on_summary)["Ticker"].tolist()
        if _on_summary.exists()
        else _on_mod.DEFAULT_TICKERS
    )
    _on_tickers_input = st.text_area(
        "Tickers to scan (one per line or comma-separated)",
        value="\n".join(_on_universe),
        height=120,
    )
    _on_tickers = [
        t.strip().upper() for t in _on_tickers_input.replace(",", "\n").splitlines()
        if t.strip()
    ]

    _on_period_days = _on_mod.PERIOD_DAYS[_on_period]

    if st.button("🔍 Run Overnight Scan", key="on_run_scan", type="primary"):
        _on_prog = st.progress(0, text="Starting…")

        def _on_cb(tk, i, n):
            _on_prog.progress((i + 1) / n, text=f"Scanning {tk} ({i+1}/{n})")

        with st.spinner("Fetching data…"):
            _on_result = _on_mod.scan_tickers(
                _on_tickers, _on_period_days, _on_cost / 2, _on_cb
            )
        _on_prog.empty()
        if not _on_result.empty:
            st.session_state["on_scan_result"] = _on_result
            st.session_state["on_scan_cost"]   = _on_cost
            st.session_state["on_scan_period"] = _on_period
            st.rerun()
        else:
            st.error("No results returned — check ticker list.")

    # ── Paper Trade History (always visible, above scan results) ──────────────
    def _on_last_close(ticker: str) -> float | None:
        """Last SETTLED close price — excludes today's intraday bar."""
        try:
            import yfinance as _yf
            _h = _yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if isinstance(_h.columns, pd.MultiIndex):
                _h.columns = _h.columns.get_level_values(0)
            _settled = _h[_h.index.date < date.today()]
            if not _settled.empty:
                return float(_settled["Close"].iloc[-1])
            return float(_h["Close"].iloc[-1]) if not _h.empty else None
        except Exception:
            return None

    def _on_sim_pnl(ticker: str, entry_date_str: str, dollars: float):
        """Compound actual overnight (open/prev-close − 1) returns from entry to today."""
        try:
            import yfinance as _yf
            _ed_date = pd.Timestamp(entry_date_str).date()
            _end_date = date.today() + timedelta(days=1)
            _h = _yf.Ticker(ticker).history(
                start=str(_ed_date), end=str(_end_date), auto_adjust=True
            )
            if isinstance(_h.columns, pd.MultiIndex):
                _h.columns = _h.columns.get_level_values(0)
            if _h.empty or "Open" not in _h.columns:
                return None, None, 0, None
            _h = _h[["Open", "Close"]].dropna()
            # Use .date() comparison to avoid tz-aware vs tz-naive mismatch
            _h = _h[_h.index.date >= _ed_date]
            if len(_h) < 2:
                return None, None, 0, None
            _on_ret = (_h["Open"] / _h["Close"].shift(1) - 1).dropna()
            if _on_ret.empty:
                return None, None, 0, None
            _cum = float((1 + _on_ret).prod() - 1) * 100
            _win = float((_on_ret > 0).mean() * 100)
            return round(_cum, 3), round((dollars or 0) * _cum / 100, 2), len(_on_ret), round(_win, 1)
        except Exception:
            return None, None, 0, None

    st.divider()
    st.subheader("📈 Paper Trade History")
    st.caption(
        "Each position simulates **daily overnight cycling** since entry: "
        "buy at close → sell at open, every trading day. "
        "P&L compounds actual overnight returns — not buy-and-hold."
    )

    _on_all      = db.load_overnight_paper_trades()
    _on_open_pos = [t for t in _on_all if t.get("status") == "open"]
    _on_cls_pos  = [t for t in _on_all if t.get("status") == "closed"]

    if _on_open_pos:
        st.markdown(f"**Open positions ({len(_on_open_pos)})**")
        _on_open_rows = []
        for _ot in _on_open_pos:
            _ot_ep  = float(_ot.get("entry_price") or 0)
            _ot_sh  = float(_ot.get("shares") or 0)
            _ot_dol = round(_ot_ep * _ot_sh, 2)
            _cpct, _cusd, _ncyc, _wr = _on_sim_pnl(
                _ot["ticker"], _ot.get("entry_date", ""), _ot_dol
            )
            _on_open_rows.append({
                "Ticker"     : _ot["ticker"],
                "Entry Date" : _ot.get("entry_date"),
                "Invested $" : _ot_dol,
                "O/N Cycles" : _ncyc,
                "O/N P&L %"  : _cpct,
                "O/N P&L $"  : _cusd,
                "Win Rate %"  : _wr,
                "Net Sharpe" : _ot.get("net_sharpe"),
                "t-stat"     : _ot.get("t_stat"),
            })
        st.dataframe(
            pd.DataFrame(_on_open_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Invested $" : st.column_config.NumberColumn(format="$%.2f"),
                "O/N Cycles" : st.column_config.NumberColumn(format="%d"),
                "O/N P&L %"  : st.column_config.NumberColumn(format="%+.3f%%"),
                "O/N P&L $"  : st.column_config.NumberColumn(format="$%+.2f"),
                "Win Rate %"  : st.column_config.NumberColumn(format="%.1f%%"),
                "Net Sharpe" : st.column_config.NumberColumn(format="%.3f"),
                "t-stat"     : st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "**O/N Cycles** = # of close→open trades since entry.  "
            "**O/N P&L** = compounded overnight-only return. "
            "Intraday is excluded — position is flat during market hours."
        )
    else:
        st.info("No open overnight positions. Run a scan and click the paper trade button.")

    if _on_cls_pos:
        _on_cdf = pd.DataFrame(_on_cls_pos)
        _oc1, _oc2, _oc3, _oc4 = st.columns(4)
        _oc1.metric("Total P&L",  f"${float(_on_cdf['pnl_dollars'].fillna(0).sum()):+.2f}")
        _oc2.metric("Avg return", f"{float(_on_cdf['pnl_pct'].fillna(0).mean()):+.3f}%")
        _oc3.metric("Win rate",   f"{float((_on_cdf['pnl_pct'].fillna(0) > 0).mean() * 100):.1f}%")
        _oc4.metric("Trades",     str(len(_on_cls_pos)))

        st.markdown(f"**Closed trades ({len(_on_cls_pos)})**")
        _on_show = [c for c in
            ["ticker", "entry_date", "entry_price", "exit_date", "exit_price",
             "pnl_pct", "pnl_dollars", "net_sharpe", "t_stat"]
            if c in _on_cdf.columns]
        st.dataframe(
            _on_cdf[_on_show].sort_values("exit_date", ascending=False),
            hide_index=True, use_container_width=True,
            column_config={
                "ticker"      : st.column_config.TextColumn("Ticker"),
                "entry_date"  : st.column_config.DateColumn("Entry Date"),
                "entry_price" : st.column_config.NumberColumn("Entry $",  format="$%.4f"),
                "exit_date"   : st.column_config.DateColumn("Exit Date"),
                "exit_price"  : st.column_config.NumberColumn("Exit $",   format="$%.4f"),
                "pnl_pct"     : st.column_config.NumberColumn("P&L %",    format="%+.3f%%"),
                "pnl_dollars" : st.column_config.NumberColumn("P&L $",    format="$%+.2f"),
                "net_sharpe"  : st.column_config.NumberColumn("Net Sharpe", format="%.3f"),
                "t_stat"      : st.column_config.NumberColumn("t-stat",     format="%.2f"),
            },
        )

    # ── Scanner results ────────────────────────────────────────────────────────
    if "on_scan_result" in st.session_state:
        _on_df     = st.session_state["on_scan_result"]
        _on_c_used = st.session_state.get("on_scan_cost", _on_cost)
        _on_p_used = st.session_state.get("on_scan_period", _on_period)

        st.divider()
        st.subheader(f"Results — {_on_p_used} lookback · {_on_c_used} bps round-trip cost")
        st.caption(
            "**Net Sharpe**: overnight-only strategy Sharpe after costs  |  "
            "**Edge (bps/day)**: overnight mean minus intraday mean  |  "
            "**Win Rate**: % of days with positive overnight return  |  "
            "**Breakeven**: max cost/side where strategy still earns > 0"
        )

        # Display columns
        _on_show_cols = [
            "rank", "ticker", "cur_price",
            "overnight_mean_bps", "intraday_mean_bps", "edge_bps",
            "net_sharpe", "win_rate", "t_stat", "p_val",
            "breakeven_bps", "gross_cum_pct", "net_cum_pct",
        ]
        _on_show_cols = [c for c in _on_show_cols if c in _on_df.columns]
        _on_display   = _on_df[_on_show_cols].copy()
        _on_display.columns = [
            c.replace("_", " ").replace("bps", "(bps)").replace("cum pct", "cum %").title()
            for c in _on_show_cols
        ]

        st.dataframe(
            _on_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Net Sharpe"       : st.column_config.NumberColumn(format="%.3f"),
                "Edge (Bps)"       : st.column_config.NumberColumn("Edge (bps/day)", format="%.2f"),
                "Overnight Mean (Bps)": st.column_config.NumberColumn("O/N mean (bps)", format="%.2f"),
                "Intraday Mean (Bps)" : st.column_config.NumberColumn("Intraday mean (bps)", format="%.2f"),
                "Win Rate"         : st.column_config.NumberColumn("Win Rate %", format="%.1f%%"),
                "Gross Cum %"      : st.column_config.NumberColumn("Gross cum %", format="%.1f%%"),
                "Net Cum %"        : st.column_config.NumberColumn("Net cum %", format="%.1f%%"),
                "Breakeven (Bps)"  : st.column_config.NumberColumn("Breakeven (bps/side)", format="%.1f"),
                "Cur Price"        : st.column_config.NumberColumn("Price", format="$%.2f"),
            },
        )

        # ── Manual paper trade ────────────────────────────────────────────────
        _on_qualify = _on_df[
            (_on_df["t_stat"]        >= 2.0)
            & (_on_df["net_sharpe"]  >= 0.30)
            & (_on_df["win_rate"]    >= 52.0)
            & (_on_df["breakeven_bps"] >= 3.0)
        ].head(5)

        if not _on_qualify.empty:
            st.info(
                f"**{len(_on_qualify)} qualifying tickers** meet all drift criteria: "
                + ", ".join(_on_qualify["ticker"].tolist())
            )
            if st.button("📥 Paper trade top 5 at last close", key="on_paper_now",
                         type="primary"):
                import uuid as _uuid
                _on_existing_open = {
                    t["ticker"]
                    for t in db.load_overnight_paper_trades()
                    if t.get("status") == "open"
                }
                _on_added = []
                _on_skipped = []
                for _, _qrow in _on_qualify.iterrows():
                    _qtk = _qrow["ticker"]
                    if _qtk in _on_existing_open:
                        _on_skipped.append(_qtk)
                        continue
                    _qclose = _on_last_close(_qtk)   # settled close, not intraday
                    if not _qclose or _qclose <= 0:
                        _on_skipped.append(_qtk)
                        continue
                    _qsh = round(1_000.0 / _qclose, 4)
                    db.add_overnight_paper_trade({
                        "id"         : str(_uuid.uuid4()),
                        "ticker"     : _qtk,
                        "entry_date" : str(date.today()),
                        "entry_price": round(_qclose, 4),
                        "shares"     : _qsh,
                        "dollars"    : round(_qclose * _qsh, 2),
                        "status"     : "open",
                        "t_stat"     : float(_qrow.get("t_stat") or 0),
                        "net_sharpe" : float(_qrow.get("net_sharpe") or 0),
                        "win_rate"   : float(_qrow.get("win_rate") or 0),
                        "placed_at"  : datetime.now(timezone.utc).isoformat(),
                    })
                    _on_added.append(f"{_qtk} @ ${_qclose:.2f}")
                if _on_added:
                    st.success(f"Opened: {', '.join(_on_added)}")
                if _on_skipped:
                    st.warning(f"Skipped (already open or no price): {', '.join(_on_skipped)}")
                st.rerun()
        else:
            st.warning("No tickers in the current scan meet all overnight drift criteria.")

        # ── Deep-dive ─────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Deep Dive")
        _on_detail_tk = st.selectbox(
            "Select ticker for detailed analysis",
            _on_df["ticker"].tolist(),
            key="on_detail_tk",
        )

        if _on_detail_tk:
            with st.spinner(f"Loading {_on_detail_tk} detail…"):
                _on_detail = _on_mod.compute_overnight_stats(
                    _on_detail_tk, _on_period_days, _on_c_used / 2
                )

            if _on_detail:
                # Stats row
                _od1, _od2, _od3, _od4, _od5, _od6 = st.columns(6)
                _od1.metric("O/N mean",    f"{_on_detail['overnight_mean_bps']:+.2f} bps/day")
                _od2.metric("Intraday mean", f"{_on_detail['intraday_mean_bps']:+.2f} bps/day")
                _od3.metric("Edge",        f"{_on_detail['edge_bps']:+.2f} bps/day")
                _od4.metric("Net Sharpe",  f"{_on_detail['net_sharpe']:+.3f}")
                _od5.metric("Win rate",    f"{_on_detail['win_rate']:.1f}%")
                _od6.metric("Breakeven",   f"{_on_detail['breakeven_bps']:.1f} bps/side")

                _od7, _od8, _od9 = st.columns(3)
                _od7.metric("t-stat (O/N > 0)", f"{_on_detail['t_stat']:.2f}",
                            help="Values > 2.0 suggest the overnight edge is statistically significant (p < 0.05)")
                _od8.metric("p-value", f"{_on_detail['p_val']:.4f}" if _on_detail["p_val"] else "—")
                _od9.metric("Days in sample", f"{_on_detail['n_days']:,}")

                # Cumulative return chart
                st.markdown("**Cumulative return — strategy comparison**")
                _chart_df = pd.DataFrame({
                    "O/N gross"   : _on_detail["_on_cum"] - 1,
                    f"O/N net ({_on_c_used} bps RT)": _on_detail["_on_net_cum"] - 1,
                    "Intraday"    : _on_detail["_id_cum"] - 1,
                    "Buy & Hold"  : _on_detail["_bah_cum"] - 1,
                }, index=_on_detail["_dates"])
                st.line_chart(_chart_df, use_container_width=True)

                # Monthly overnight returns heatmap
                st.markdown("**Monthly overnight returns (%)**")
                _monthly = _on_detail["_monthly"]
                if not _monthly.empty:
                    try:
                        _mo_df = _monthly.unstack(level=1)
                        _mo_df.index.name   = "Year"
                        _mo_df.columns      = [
                            ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
                            for m in _mo_df.columns
                        ]
                        st.dataframe(
                            _mo_df.style.background_gradient(cmap="RdYlGn", axis=None)
                                        .format("{:+.2f}%"),
                            use_container_width=True,
                        )
                    except Exception:
                        st.dataframe(_monthly.rename("O/N return (%)").reset_index(),
                                     hide_index=True, use_container_width=True)

                # Rolling 63-day overnight Sharpe
                st.markdown("**Rolling 63-day overnight Sharpe**")
                _roll_on = _on_detail["_on_series"]
                _roll_sh = (_roll_on.rolling(63).mean() / _roll_on.rolling(63).std() * np.sqrt(252)).rename("Rolling Sharpe")
                st.line_chart(_roll_sh.dropna(), use_container_width=True)

                # Cost sensitivity table
                st.markdown("**Net Sharpe vs round-trip cost assumption**")
                _cost_rows = []
                for _cp in [0, 2, 4, 6, 8, 10, 15, 20]:
                    _c_daily = _cp / 10_000
                    _on_net_c = _on_detail["_on_series"] - _c_daily
                    _sh = (_on_net_c.mean() / _on_net_c.std() * np.sqrt(252)) if _on_net_c.std() > 0 else 0
                    _cum = float((1 + _on_net_c).prod() - 1) * 100
                    _cost_rows.append({
                        "Round-trip cost (bps)": _cp,
                        "Net Sharpe"           : round(_sh, 3),
                        f"Net cum % ({_on_p_used})": round(_cum, 1),
                    })
                st.dataframe(pd.DataFrame(_cost_rows), hide_index=True, use_container_width=True)

            else:
                st.warning(f"Could not load detail for {_on_detail_tk}.")


# ══════════════════════════════════════════════════════════════════════════════
# 200d SMA TAB
# ══════════════════════════════════════════════════════════════════════════════
_SP500_CSV = ROOT / "sp500_constituents.csv"


def _sma_run_scan() -> pd.DataFrame:
    """Batch-download 1yr of closes for the full S&P 500 and compute 200d SMA proximity."""
    import yfinance as yf
    if not _SP500_CSV.exists():
        st.warning(
            f"Ticker list not found at {_SP500_CSV.name}. "
            "Commit stocks/sp500_constituents.csv to the repo."
        )
        return pd.DataFrame()

    _uni = pd.read_csv(_SP500_CSV)
    _tickers = _uni["Ticker"].dropna().astype(str).str.strip().tolist()
    _sector_map = dict(zip(_uni["Ticker"], _uni.get("Sector", pd.Series(dtype=str))))

    _data = yf.download(_tickers, period="1y", auto_adjust=True,
                        progress=False, threads=True)
    _closes = _data["Close"] if "Close" in _data.columns else _data.xs("Close", axis=1, level=0)

    _rows = []
    for _tk in _tickers:
        try:
            _s = _closes[_tk].dropna() if _tk in _closes.columns else pd.Series()
            if len(_s) < 201:
                continue
            _sma = _s.rolling(200).mean()
            _c0, _c1 = float(_s.iloc[-2]), float(_s.iloc[-1])
            _m0, _m1 = float(_sma.iloc[-2]), float(_sma.iloc[-1])
            _pct = (_c1 - _m1) / _m1 * 100
            if _c0 > _m0 and _c1 <= _m1:
                _sig = "crossunder"
            elif _pct < 0:
                _sig = "below"
            elif _pct <= 2:
                _sig = "near"
            else:
                _sig = "above"
            _rows.append({"Ticker": _tk, "Sector": _sector_map.get(_tk, ""),
                           "Close": round(_c1, 2),
                           "200d SMA": round(_m1, 2), "% vs SMA": round(_pct, 2),
                           "Signal": _sig})
        except Exception:
            continue

    return pd.DataFrame(_rows).sort_values("% vs SMA") if _rows else pd.DataFrame()


with tab_sma:
    st.header("📉 200-Day SMA Scanner")
    st.caption(
        "Scan the full S&P 500 for stocks pulling back to their 200-day moving average. "
        "Log long entries and track P&L."
    )

    # ── Scan controls ──────────────────────────────────────────────────────────
    _sma_c1, _sma_c2 = st.columns([4, 1])
    _sma_c1.markdown(
        "Batch-downloads prices for ~500 tickers via yfinance. "
        "Results are stored in your session — click **Run Scan** to refresh."
    )
    if _sma_c2.button("🔄 Run Scan", type="primary", key="sma_run_btn"):
        with st.spinner("Scanning S&P 500 (500 tickers) — takes ~30 seconds…"):
            st.session_state["sma_scan_df"]   = _sma_run_scan()
            st.session_state["sma_scan_label"] = date.today().isoformat()

    if "sma_scan_df" in st.session_state and not st.session_state["sma_scan_df"].empty:
        _sdf = st.session_state["sma_scan_df"]
        _sma_ts = st.session_state.get("sma_scan_label", "")
        st.caption(f"Last scan: {_sma_ts} · {len(_sdf)} tickers")

        # Summary chips
        _n_cross  = int((_sdf["Signal"] == "crossunder").sum())
        _n_below  = int((_sdf["Signal"] == "below").sum())
        _n_near   = int((_sdf["Signal"] == "near").sum())
        _sm1, _sm2, _sm3 = st.columns(3)
        _sm1.metric("🔴 Crossed below",    _n_cross)
        _sm2.metric("🔻 Below SMA",        _n_below)
        _sm3.metric("🟡 Within 2% above",  _n_near)

        # Sector filter
        if "Sector" in _sdf.columns:
            _sectors = sorted(s for s in _sdf["Sector"].dropna().unique() if s)
            _sel_sectors = st.multiselect(
                "Filter by sector", options=_sectors, default=[],
                placeholder="All sectors",
            )
            if _sel_sectors:
                _sdf = _sdf[_sdf["Sector"].isin(_sel_sectors)]

        # Filter tabs
        _sv1, _sv2, _sv3 = st.tabs([
            f"🎯 Actionable — near or below ({_n_cross + _n_below + _n_near})",
            "🔴 Crossed Below Today",
            "📋 Full Universe",
        ])

        _col_cfg = {
            "Sector"   : st.column_config.TextColumn("Sector"),
            "Close"    : st.column_config.NumberColumn("Close $",    format="$%.2f"),
            "200d SMA" : st.column_config.NumberColumn("200d SMA $", format="$%.2f"),
            "% vs SMA" : st.column_config.NumberColumn("% vs SMA",   format="%+.2f%%"),
        }

        with _sv1:
            _near_df = _sdf[_sdf["Signal"].isin(["crossunder", "below", "near"])].copy()
            if _near_df.empty:
                st.info("No stocks currently at or near their 200-day SMA.")
            else:
                st.dataframe(_near_df, hide_index=True,
                             use_container_width=True, column_config=_col_cfg)

        with _sv2:
            _cross_df = _sdf[_sdf["Signal"] == "crossunder"].copy()
            if _cross_df.empty:
                st.info("No crossunders today — no stock closed below 200d SMA for the first time.")
            else:
                st.dataframe(_cross_df, hide_index=True,
                             use_container_width=True, column_config=_col_cfg)

        with _sv3:
            st.dataframe(_sdf, hide_index=True,
                         use_container_width=True, column_config=_col_cfg)

    elif "sma_scan_df" not in st.session_state:
        st.info("Click **Run Scan** to load current SMA data for the full S&P 500.")

    # ── Trade log ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💼 200d SMA Trade Log")
    st.caption("Log long entries at the 200-day SMA. P&L is computed automatically on close.")

    _sma_trades_all    = db.load_sma_trades()
    _sma_trades_open   = [t for t in _sma_trades_all if t.get("status") == "open"]
    _sma_trades_closed = [t for t in _sma_trades_all if t.get("status") == "closed"]

    with st.expander("➕ Log New Trade", expanded=len(_sma_trades_open) == 0):
        with st.form("sma_log_form", clear_on_submit=True):
            _slc1, _slc2 = st.columns(2)
            _sl_ticker = _slc1.text_input("Ticker").upper().strip()
            _sl_date   = _slc2.date_input("Entry date", value=date.today())
            _slc3, _slc4 = st.columns(2)
            _sl_price  = _slc3.number_input("Entry price ($)", min_value=0.0001,
                                             value=100.0, step=0.01, format="%.4f")
            _sl_dollars = _slc4.number_input("Dollar amount ($)",
                                              min_value=1.0, value=1000.0, step=100.0)
            _sl_notes  = st.text_input("Notes (optional)", placeholder="e.g. bounced off 200d with volume")
            if st.form_submit_button("💾 Log Trade", type="primary"):
                if not _sl_ticker:
                    st.error("Ticker is required.")
                else:
                    import uuid as _sma_uuid
                    _sl_shares = round(_sl_dollars / _sl_price, 4) if _sl_price > 0 else 0
                    db.add_sma_trade({
                        "id":          str(_sma_uuid.uuid4()),
                        "ticker":      _sl_ticker,
                        "entry_date":  _sl_date.isoformat(),
                        "entry_price": round(_sl_price, 4),
                        "shares":      _sl_shares,
                        "dollars":     round(_sl_price * _sl_shares, 2),
                        "status":      "open",
                        "notes":       _sl_notes or None,
                        "placed_at":   datetime.now(timezone.utc).isoformat(),
                    })
                    st.success(
                        f"Logged: {_sl_ticker} @ ${_sl_price:.4f} · "
                        f"{_sl_shares:.4f} shares · ${_sl_price * _sl_shares:.2f}"
                    )
                    st.rerun()

    # Open positions
    if _sma_trades_open:
        st.markdown(f"**Open positions ({len(_sma_trades_open)})**")
        _sma_open_rows = []
        for _st_row in _sma_trades_open:
            _st_ep = float(_st_row.get("entry_price") or 0)
            _st_sh = float(_st_row.get("shares") or 0)
            _sma_open_rows.append({
                "Ticker"     : _st_row["ticker"],
                "Entry Date" : _st_row.get("entry_date", ""),
                "Entry $"    : _st_ep,
                "Shares"     : _st_sh,
                "Invested $" : round(_st_ep * _st_sh, 2),
                "Notes"      : _st_row.get("notes", ""),
            })
        st.dataframe(
            pd.DataFrame(_sma_open_rows),
            hide_index=True, use_container_width=True,
            column_config={
                "Entry $"   : st.column_config.NumberColumn(format="$%.4f"),
                "Shares"    : st.column_config.NumberColumn(format="%.4f"),
                "Invested $": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        st.caption("Close a position with your actual exit price:")
        for _st_row in _sma_trades_open:
            _st_ep = float(_st_row.get("entry_price") or 0)
            with st.expander(
                f"✅ Close {_st_row['ticker']} — entered ${_st_ep:.4f} on {_st_row.get('entry_date', '?')}"
            ):
                with st.form(f"sma_close_{_st_row['id']}"):
                    _st_exit_px = st.number_input(
                        "Exit price ($)", min_value=0.0001,
                        value=_st_ep, step=0.01, format="%.4f",
                    )
                    _st_exit_dt = st.date_input("Exit date", value=date.today())
                    if st.form_submit_button("Close Trade"):
                        try:
                            db.close_sma_trade(
                                _st_row["id"],
                                round(float(_st_exit_px), 4),
                                _st_exit_dt.isoformat(),
                            )
                            _st_pnl = (_st_exit_px - _st_ep) / _st_ep * 100 if _st_ep > 0 else 0
                            st.success(
                                f"Closed {_st_row['ticker']} @ ${_st_exit_px:.4f} "
                                f"| P&L {_st_pnl:+.3f}%"
                            )
                            st.rerun()
                        except Exception as _st_ce:
                            st.error(f"Close failed: {_st_ce}")
    else:
        st.info("No open SMA trades. Use the form above to log a new entry.")

    # Closed positions
    if _sma_trades_closed:
        _sma_cdf     = pd.DataFrame(_sma_trades_closed)
        _sma_total   = float(_sma_cdf["pnl_dollars"].fillna(0).sum())
        _sma_avg_pct = float(_sma_cdf["pnl_pct"].fillna(0).mean())
        _sma_wr      = float((_sma_cdf["pnl_pct"].fillna(0) > 0).mean() * 100)

        _scm1, _scm2, _scm3, _scm4 = st.columns(4)
        _scm1.metric("Total P&L",  f"${_sma_total:+.2f}")
        _scm2.metric("Avg return", f"{_sma_avg_pct:+.3f}%")
        _scm3.metric("Win rate",   f"{_sma_wr:.1f}%")
        _scm4.metric("Closed",     str(len(_sma_trades_closed)))

        _sma_show = [c for c in [
            "ticker", "entry_date", "entry_price", "exit_date", "exit_price",
            "pnl_pct", "pnl_dollars", "notes",
        ] if c in _sma_cdf.columns]
        st.dataframe(
            _sma_cdf[_sma_show].sort_values("exit_date", ascending=False),
            hide_index=True, use_container_width=True,
            column_config={
                "ticker"      : st.column_config.TextColumn("Ticker"),
                "entry_date"  : st.column_config.DateColumn("Entry Date"),
                "entry_price" : st.column_config.NumberColumn("Entry $",  format="$%.4f"),
                "exit_date"   : st.column_config.DateColumn("Exit Date"),
                "exit_price"  : st.column_config.NumberColumn("Exit $",   format="$%.4f"),
                "pnl_pct"     : st.column_config.NumberColumn("P&L %",    format="%+.3f%%"),
                "pnl_dollars" : st.column_config.NumberColumn("P&L $",    format="$%+.2f"),
                "notes"       : st.column_config.TextColumn("Notes"),
            },
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


