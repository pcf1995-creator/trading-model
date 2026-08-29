"""
sports/test_display.py — regression tests for the dashboard display helpers.

Run:  python sports/test_display.py

Covers: totals rendered without a meaningless +/- sign, spreads keeping their
sign, moneylines showing no line at all, and all times rendered in ET.
Stubs Streamlit so nothing needs a browser or API keys.
"""
import sys, types, pandas as pd

# Minimal streamlit stub so app.py's module-level calls are inert
st = types.ModuleType("streamlit")
class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return lambda *a, **k: _Ctx()
    def __bool__(self): return False
def _noop(*a, **k): return _Ctx()
for name in ("set_page_config","title","tabs","columns","selectbox","slider",
             "number_input","checkbox","button","spinner","error","info","success",
             "warning","caption","dataframe","data_editor","metric","expander",
             "text","write","divider","subheader","rerun","line_chart","cache_data",
             "cache_resource","column_config","secrets","session_state"):
    setattr(st, name, _noop)
st.tabs = lambda labels: tuple(_Ctx() for _ in labels)
st.columns = lambda spec, **k: tuple(_Ctx() for _ in (range(spec) if isinstance(spec,int) else spec))
st.button = lambda *a, **k: False
st.checkbox = lambda *a, **k: False
st.session_state = {}
st.secrets = {}
st.cache_resource = lambda f=None, **k: (f if f else (lambda g: g))
st.cache_data = st.cache_resource
sys.modules["streamlit"] = st

_HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import app

print("1. LINE FORMATTING BY MARKET")
cases = [("total", 47.5), ("total", 56.5), ("spread", -29.0), ("spread", 20.5),
         ("moneyline", None), ("moneyline", float("nan"))]
for market, line in cases:
    print(f"   {market:<10} {str(line):<8} -> {app._fmt_line(market, line)!r}")

assert app._fmt_line("total", 47.5)     == "47.5",  "total must not carry a sign"
assert app._fmt_line("total", 56.5)     == "56.5"
assert app._fmt_line("spread", -29.0)   == "-29.0", "spread must keep its sign"
assert app._fmt_line("spread", 20.5)    == "+20.5", "spread must show explicit +"
assert app._fmt_line("moneyline", None) == "—"
assert app._fmt_line("moneyline", float("nan")) == "—"

print("\n2. KICKOFF TIMES IN ET")
times = pd.Series(["2026-08-29T16:08:00Z", "2026-09-05T23:00:00Z",
                   "2026-09-06T02:00:00Z"])
for iso, out in zip(times, app._fmt_et(times)):
    print(f"   {iso}  ->  {out}")
assert "12:08PM ET" in app._fmt_et(times)[0], "UTC->ET conversion wrong"
assert app._fmt_et(times)[0].startswith("Sat 8/29")

print("\n3. FULL DISPLAY FRAME")
recs = [
    {"game_id":"g1","sport":"ncaaf","home":"TCU Horned Frogs","away":"North Carolina Tar Heels",
     "commence":"2026-08-29T16:08:00Z","market":"moneyline","side":"North Carolina Tar Heels",
     "odds":470,"line":None,"book":"fanduel","model_prob":0.216,"market_prob":0.171,
     "edge":0.045,"ev":0.233,"kelly_pct":2.48,"model_vs_market":-9.0},
    {"game_id":"g2","sport":"ncaaf","home":"Colorado State Rams","away":"Wyoming Cowboys",
     "commence":"2026-09-05T22:00:00Z","market":"total","side":"Under",
     "odds":-106,"line":47.5,"book":"fanduel","model_prob":0.572,"market_prob":0.505,
     "edge":0.067,"ev":0.111,"kelly_pct":5.0,"model_vs_market":-12.6},
    {"game_id":"g3","sport":"ncaaf","home":"Rutgers Scarlet Knights","away":"UMass Minutemen",
     "commence":"2026-09-03T22:00:00Z","market":"spread","side":"Rutgers Scarlet Knights -29",
     "odds":-106,"line":-29.0,"book":"pinnacle","model_prob":0.552,"market_prob":0.498,
     "edge":0.054,"ev":0.073,"kelly_pct":3.86,"model_vs_market":8.5},
]
df = app.build_display_df(recs, 500.0)
print(df[["Market","Side","Kickoff","Odds","Line","Bet $","∆ pts"]].to_string(index=False))

lines = dict(zip(df["Market"], df["Line"]))
assert lines["Total"] == "47.5",  f"total line shows a sign: {lines['Total']}"
assert lines["Spread"] == "-29.0"
assert lines["Moneyline"] == "—"
assert all("ET" in k for k in df["Kickoff"]), "kickoffs not in ET"
assert set(app.DISPLAY_COLS).issubset(df.columns), "missing display columns"

print("\nAll display assertions passed.")
