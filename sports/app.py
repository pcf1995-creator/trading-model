"""
sports/app.py — Sports EV Scanner Streamlit dashboard

Tabs:
  🔍 Scanner   — run a fresh scan for NCAAF + NFL, see recommendations
  📋 Open Bets — paper trades in progress (from Supabase sports_bets)
  📈 P&L       — settled bet history and performance summary

Run locally:
  streamlit run sports/app.py

Deploy on Streamlit Cloud:
  Entry point: sports/app.py
  Secrets (Settings → Secrets):
    CFBD_API_KEY = "..."
    ODDS_API_KEY = "..."
    SUPABASE_URL = "..."
    SUPABASE_KEY = "..."
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_SPORTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SPORTS_DIR))
sys.path.insert(0, str(_SPORTS_DIR.parent))

# Inject Streamlit secrets into os.environ so cfbd_client / odds_client pick them up
import os
for _k in ("CFBD_API_KEY", "ODDS_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"):
    if _k not in os.environ:
        try:
            _v = st.secrets.get(_k)
            if _v:
                os.environ[_k] = _v
        except Exception:
            pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sports EV Scanner", layout="wide")
st.title("🏈 Sports EV Scanner")

tab_scan, tab_open, tab_pnl = st.tabs(["🔍 Scanner", "📋 Open Bets", "📈 P&L"])


# ── Supabase helper ───────────────────────────────────────────────────────────
@st.cache_resource
def _get_db_client():
    try:
        from shared.db_common import _get_client
        return _get_client()
    except Exception:
        return None


def load_sports_bets(status: str | None = None) -> list[dict]:
    client = _get_db_client()
    if client:
        try:
            q = client.table("sports_bets").select("*").order("scanned_at", desc=True)
            if status:
                q = q.eq("status", status)
            return q.execute().data or []
        except Exception as e:
            st.warning(f"Supabase load failed: {e}")
    return []


def save_bets(recs: list[dict], bankroll: float) -> int:
    client = _get_db_client()
    if not client:
        st.warning("No Supabase connection — bets not saved.")
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for r in recs:
        bet_dollars = round(bankroll * r["kelly_pct"] / 100, 2)
        rows.append({
            "game_id"    : r["game_id"],
            "sport"      : r["sport"],
            "home"       : r["home"],
            "away"       : r["away"],
            "commence"   : r.get("commence"),
            "market"     : r["market"],
            "side"       : r["side"],
            "odds"       : r["odds"],
            "line"       : r.get("line"),
            "book"       : r.get("book"),
            "model_prob" : r["model_prob"],
            "market_prob": r["market_prob"],
            "edge"       : r["edge"],
            "ev"         : r["ev"],
            "kelly_pct"  : r["kelly_pct"],
            "bet_dollars": bet_dollars,
            "status"     : "open",
            "scanned_at" : now,
        })
    try:
        client.table("sports_bets").insert(rows).execute()
        return len(rows)
    except Exception as e:
        st.error(f"Save failed: {e}")
        return 0


def settle_bet(bet_id: int, result: str, pnl: float) -> None:
    client = _get_db_client()
    if not client:
        return
    client.table("sports_bets").update({
        "status"    : "settled",
        "result"    : result,
        "pnl_dollars": pnl,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", bet_id).execute()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    col_cfg, col_run = st.columns([3, 1])

    with col_cfg:
        sport_choice = st.selectbox(
            "Sport", ["Both (NCAAF + NFL)", "NCAAF only", "NFL only"],
            index=0,
        )
        min_ev_input = st.slider(
            "Minimum EV threshold", min_value=0.02, max_value=0.15,
            value=0.04, step=0.01,
            help="4pp is the default for G5; 5pp for Power 4 / NFL",
        )
        bankroll = st.number_input(
            "Bankroll ($)", min_value=100, max_value=50_000,
            value=500, step=100,
        )
        save_to_db = st.checkbox("Save recommendations to Supabase", value=False)

    with col_run:
        st.write("")
        st.write("")
        run_scan = st.button("▶ Run Scan", use_container_width=True, type="primary")

    if run_scan:
        sport_map = {
            "Both (NCAAF + NFL)": "both",
            "NCAAF only": "ncaaf",
            "NFL only": "nfl",
        }
        sport_key = sport_map[sport_choice]

        with st.spinner("Fetching odds and building efficiency ratings…"):
            try:
                from scanner import run_scan as _run_scan
                recs = _run_scan(
                    sport=sport_key,
                    min_ev=min_ev_input,
                    bankroll=bankroll,
                    save=False,   # handle save below
                )
            except RuntimeError as e:
                st.error(f"⚠️ {e}")
                st.info(
                    "**To fix:** Add `CFBD_API_KEY` to this app's Streamlit secrets "
                    "(Settings → Secrets). Get a free key at collegefootballdata.com/key"
                )
                recs = []
            except Exception as e:
                st.error(f"Scan failed: {e}")
                recs = []

        # Scan diagnostics — how much of the slate was actually modelled
        try:
            from scanner import LAST_SCAN_STATS as _stats
        except Exception:
            _stats = {}
        if _stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Games with odds", _stats.get("games", 0))
            c2.metric("Modelled",        _stats.get("scored", 0))
            c3.metric("Skipped (FCS)",   _stats.get("skipped_unrated", 0),
                      help="No S&P+ coverage — not modelled rather than guessed at")
            c4.metric("Unmatched names", _stats.get("skipped_unknown", 0),
                      help="Team name in the odds feed couldn't be matched to CFBD")
            if _stats.get("skipped_nfl"):
                st.caption("NFL skipped — CFBD is college-only; no NFL data source wired up yet.")
            if _stats.get("unknown_names"):
                with st.expander(f"Unmatched team names ({_stats.get('skipped_unknown', 0)})"):
                    for n in _stats["unknown_names"]:
                        st.text(n)

        if not recs:
            st.info("No bets cleared the edge threshold for this slate.")
        else:
            st.success(f"Found **{len(recs)}** recommendation{'s' if len(recs) != 1 else ''}")

            df = pd.DataFrame(recs)
            df["Game"]     = df["away"] + " @ " + df["home"]
            df["Kickoff"]  = pd.to_datetime(df["commence"], utc=True, errors="coerce") \
                               .dt.strftime("%a %-m/%-d %-I:%M%p UTC")
            df["Market"]   = df["market"].str.capitalize()
            df["Side"]     = df["side"]
            df["Odds"]     = df["odds"].apply(lambda x: f"{x:+d}")
            # Moneylines have no line; pandas turns those Nones into NaN
            df["Line"]     = df["line"].apply(
                lambda x: "—" if x is None or pd.isna(x) else f"{float(x):+.1f}")
            df["Mdl %"]    = (df["model_prob"]  * 100).round(1).astype(str) + "%"
            df["Mkt %"]    = (df["market_prob"] * 100).round(1).astype(str) + "%"
            df["Edge"]     = (df["edge"] * 100).round(1).apply(lambda x: f"{x:+.1f}pp")
            df["EV"]       = df["ev"].round(3).apply(lambda x: f"{x:+.3f}")
            df["Kelly %"]  = df["kelly_pct"].round(2).astype(str) + "%"
            df["Bet $"]    = (bankroll * df["kelly_pct"] / 100).round(2).apply(
                                lambda x: f"${x:.2f}")
            df["Book"]     = df["book"]
            df["∆ pts"]    = df["model_vs_market"].apply(lambda x: f"{x:+.1f}")

            display_cols = ["Market", "Side", "Game", "Kickoff", "Odds", "Line",
                            "Mdl %", "Mkt %", "Edge", "EV", "Kelly %", "Bet $",
                            "∆ pts", "Book"]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

            if save_to_db:
                n = save_bets(recs, bankroll)
                st.success(f"Saved {n} bets to Supabase.")

            # Per-market breakdown
            for mkt in ["total", "spread", "moneyline"]:
                sub = df[df["market"] == mkt]
                if not sub.empty:
                    with st.expander(f"{mkt.capitalize()}s ({len(sub)})"):
                        st.dataframe(sub[display_cols], use_container_width=True,
                                     hide_index=True)

    else:
        st.info("Configure options above and click **▶ Run Scan** to fetch live recommendations.")
        cfbd_ok = bool(os.environ.get("CFBD_API_KEY"))
        odds_ok  = bool(os.environ.get("ODDS_API_KEY"))
        db_ok    = bool(os.environ.get("SUPABASE_URL"))
        col1, col2, col3 = st.columns(3)
        col1.metric("CFBD API", "✅ Set" if cfbd_ok else "❌ Missing")
        col2.metric("Odds API", "✅ Set" if odds_ok  else "❌ Missing")
        col3.metric("Supabase", "✅ Set" if db_ok    else "❌ Missing")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPEN BETS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_open:
    st.subheader("Open Bets")

    if st.button("🔄 Refresh", key="refresh_open"):
        st.cache_data.clear()

    open_bets = load_sports_bets(status="open")

    if not open_bets:
        st.info("No open bets. Run a scan and save recommendations first.")
    else:
        df_open = pd.DataFrame(open_bets)
        df_open["Game"]   = df_open["away"] + " @ " + df_open["home"]
        df_open["Kickoff"] = pd.to_datetime(
            df_open["commence"], utc=True, errors="coerce"
        ).dt.strftime("%a %-m/%-d %-I%p UTC")
        df_open["Market"] = df_open["market"].str.capitalize()
        df_open["Odds"]   = df_open["odds"].apply(lambda x: f"{x:+d}")
        df_open["Line"]   = df_open["line"].apply(
            lambda x: f"{float(x):+.1f}" if x is not None else "—")
        df_open["Edge"]   = (df_open["edge"] * 100).round(1).apply(
            lambda x: f"{x:+.1f}pp")
        df_open["EV"]     = df_open["ev"].round(3).apply(lambda x: f"{x:+.3f}")
        df_open["Bet $"]  = df_open["bet_dollars"].apply(
            lambda x: f"${float(x):.2f}" if x else "—")
        df_open["Book"]   = df_open["book"]
        df_open["Saved"]  = pd.to_datetime(
            df_open["scanned_at"], utc=True, errors="coerce"
        ).dt.strftime("%-m/%-d %H:%M")

        display = ["Game", "Kickoff", "Market", "side", "Odds", "Line",
                   "Edge", "EV", "Bet $", "Book", "Saved"]
        st.dataframe(df_open[display].rename(columns={"side": "Side"}),
                     use_container_width=True, hide_index=True)

        st.write(f"**{len(open_bets)} open bets** | "
                 f"Total at risk: **${df_open['bet_dollars'].astype(float).sum():.2f}**")

        # ── Settle a bet ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Settle a Bet")
        bet_options = {
            f"{r['away']} @ {r['home']} — {r['market']} {r['side']} "
            f"(${float(r.get('bet_dollars',0)):.2f})": r
            for r in open_bets
        }
        chosen_label = st.selectbox("Select bet to settle", list(bet_options.keys()))
        chosen_bet   = bet_options[chosen_label]

        col_res, col_pnl = st.columns(2)
        result_choice = col_res.selectbox("Result", ["win", "loss", "push"])
        bet_size = float(chosen_bet.get("bet_dollars", 0))
        odds_val = int(chosen_bet.get("odds", -110))
        if result_choice == "win":
            from odds_client import american_to_decimal
            pnl_auto = round(bet_size * (american_to_decimal(odds_val) - 1), 2)
        elif result_choice == "loss":
            pnl_auto = -bet_size
        else:
            pnl_auto = 0.0
        pnl_input = col_pnl.number_input("P&L ($)", value=pnl_auto, step=0.01)

        if st.button("✅ Mark Settled", type="primary"):
            settle_bet(chosen_bet["id"], result_choice, pnl_input)
            st.success(f"Settled: {result_choice.upper()}  ${pnl_input:+.2f}")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — P&L
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pnl:
    st.subheader("Settled Bet Performance")

    if st.button("🔄 Refresh", key="refresh_pnl"):
        st.cache_data.clear()

    settled = load_sports_bets(status="settled")

    if not settled:
        st.info("No settled bets yet.")
    else:
        df_s = pd.DataFrame(settled)
        df_s["pnl_dollars"] = pd.to_numeric(df_s["pnl_dollars"], errors="coerce").fillna(0)
        df_s["bet_dollars"] = pd.to_numeric(df_s["bet_dollars"], errors="coerce").fillna(0)

        total_pnl    = df_s["pnl_dollars"].sum()
        total_risk   = df_s["bet_dollars"].sum()
        wins         = (df_s["result"] == "win").sum()
        losses       = (df_s["result"] == "loss").sum()
        pushes       = (df_s["result"] == "push").sum()
        win_rate     = wins / (wins + losses) if (wins + losses) > 0 else 0
        roi          = total_pnl / total_risk if total_risk > 0 else 0

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total P&L",  f"${total_pnl:+.2f}")
        c2.metric("ROI",        f"{roi*100:+.1f}%")
        c3.metric("Win Rate",   f"{win_rate*100:.1f}%")
        c4.metric("Record",     f"{wins}W–{losses}L–{pushes}P")
        c5.metric("Total Risked", f"${total_risk:.2f}")

        # Cumulative P&L chart
        df_s_sorted = df_s.sort_values("settled_at").copy()
        df_s_sorted["cum_pnl"] = df_s_sorted["pnl_dollars"].cumsum()
        st.line_chart(df_s_sorted.set_index("settled_at")["cum_pnl"],
                      use_container_width=True)

        # Breakdown by market type
        st.subheader("By Market")
        mkt_summary = (
            df_s.groupby("market")
            .agg(bets=("id","count"), pnl=("pnl_dollars","sum"),
                 risked=("bet_dollars","sum"),
                 wins=("result", lambda x: (x=="win").sum()))
            .assign(win_rate=lambda d: d["wins"]/d["bets"],
                    roi=lambda d: d["pnl"]/d["risked"])
            .reset_index()
        )
        mkt_summary["P&L"]      = mkt_summary["pnl"].apply(lambda x: f"${x:+.2f}")
        mkt_summary["Win Rate"] = (mkt_summary["win_rate"]*100).round(1).astype(str)+"%"
        mkt_summary["ROI"]      = (mkt_summary["roi"]*100).round(1).apply(
                                    lambda x: f"{x:+.1f}%")
        st.dataframe(
            mkt_summary[["market","bets","P&L","Win Rate","ROI"]]
                .rename(columns={"market":"Market","bets":"Bets"}),
            use_container_width=True, hide_index=True,
        )

        # Full history table
        with st.expander("Full history"):
            df_s["Game"]   = df_s["away"] + " @ " + df_s["home"]
            df_s["Odds"]   = df_s["odds"].apply(lambda x: f"{x:+d}")
            df_s["P&L"]    = df_s["pnl_dollars"].apply(lambda x: f"${x:+.2f}")
            df_s["Bet $"]  = df_s["bet_dollars"].apply(lambda x: f"${x:.2f}")
            df_s["Result"] = df_s["result"].str.upper()
            df_s["Date"]   = pd.to_datetime(df_s["settled_at"],
                                             utc=True, errors="coerce") \
                               .dt.strftime("%-m/%-d")
            st.dataframe(
                df_s[["Date","Game","market","side","Odds","Bet $","Result","P&L"]]
                    .rename(columns={"market":"Market","side":"Side"}),
                use_container_width=True, hide_index=True,
            )
