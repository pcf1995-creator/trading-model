"""
stocks/test_edgar_spinoff.py — regression tests for the EDGAR spin-off search.

Run:  python stocks/test_edgar_spinoff.py

Guards the failure that broke the spin-off scanner: EDGAR answers 500 (not an
empty result set) when the `from` offset runs past the number of matching
filings, so blindly requesting from=0,10,20,30,40 blew up on a 23-result query
and discarded the pages that had already succeeded.

Execs the real function source out of app.py with requests/streamlit stubbed,
so it needs no network and no API keys.
"""
import re, sys, types, time
from datetime import date, timedelta
from pathlib import Path

SRC = (Path(__file__).resolve().parent / "app.py").read_text()
start = SRC.index("_SEC_HEADERS = {")
end   = SRC.index("# ── Page config ─")
block = SRC[start:end]

# ── Stub requests ────────────────────────────────────────────────────────────
requests = types.ModuleType("requests")
class HTTPError(Exception): pass
requests.HTTPError = HTTPError

CALLS = []          # (url, from-offset, timestamp)
SCRIPT = {}         # offset -> ("ok", n_hits, total) | ("500", ) | ("ok_then", ...)
FAIL_COUNT = {}     # offset -> how many times it has 500'd so far

class Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
    def json(self): return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"{self.status_code} Server Error")

def _get(url, params=None, headers=None, timeout=None):
    params = params or {}
    off = int(params.get("from", 0))
    CALLS.append((url, off, time.monotonic()))
    if "submissions" in url:                      # filer-type lookup
        return Resp(200, {"filings": {"recent": {"form": ["S-1"]}}})
    plan = SCRIPT.get(off, ("ok", 10, 100))
    if plan[0] == "500":
        return Resp(500, {})
    if plan[0] == "flaky":                        # 500 once, then succeed
        n = FAIL_COUNT.get(off, 0)
        FAIL_COUNT[off] = n + 1
        if n < plan[1]:
            return Resp(500, {})
        plan = ("ok", plan[2], plan[3])
    _, n_hits, total = plan
    hits = [{"_source": {"display_names": [f"Co {off+i}"], "ciks": [f"{off+i:010d}"],
                         "file_date": "2026-09-01", "form": "10-12B"}}
            for i in range(n_hits)]
    return Resp(200, {"hits": {"total": {"value": total}, "hits": hits}})

requests.get = _get
sys.modules["requests"] = requests

# ── Stub streamlit (only cache_data is used in this block) ───────────────────
st = types.ModuleType("streamlit")
st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = st

ns = {"st": st, "date": date, "timedelta": timedelta}
exec(compile(block, "app_block", "exec"), ns)
spinoff  = ns["_spinoff_edgar"]
sec_get  = ns["_sec_get"]

def reset(script):
    CALLS.clear(); FAIL_COUNT.clear()
    SCRIPT.clear(); SCRIPT.update(script)
    ns["_sec_last_request"] = 0.0

def fts_offsets():
    return [off for url, off, _ in CALLS if "search-index" in url]

ok = True
def check(label, cond, detail=""):
    global ok
    print(f"   {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    ok = ok and cond

# ── 1. The reported bug: 23 results, must not probe past the total ───────────
print("1. STOPS AT REPORTED TOTAL (the from=40 500 the user hit)")
reset({0: ("ok", 10, 23), 10: ("ok", 10, 23), 20: ("ok", 3, 23),
       30: ("500",), 40: ("500",)})
rows, warn = spinoff("spin-off", 120)
check("never requested from=30 or from=40", not ({30, 40} & set(fts_offsets())),
      f"offsets={fts_offsets()}")
check("returned all 23 companies", len(rows) == 23, f"got {len(rows)}")
check("no spurious warning", warn is None, str(warn))

# ── 2. Mid-run failure keeps what was already retrieved ─────────────────────
print("\n2. MID-RUN FAILURE RETURNS PARTIAL RESULTS (not a total wipeout)")
reset({0: ("ok", 10, 100), 10: ("ok", 10, 100), 20: ("500",),
       30: ("500",), 40: ("500",)})
rows, warn = spinoff("spin-off", 120)
check("kept the 20 rows from pages 1-2", len(rows) == 20, f"got {len(rows)}")
check("warned about the truncation", warn is not None and "Showing what was retrieved" in warn)

# ── 3. Total failure on the first page still raises ─────────────────────────
print("\n3. FIRST-PAGE FAILURE STILL RAISES (a real outage is a real error)")
reset({0: ("500",)})
try:
    spinoff("spin-off", 120)
    check("raised", False, "no exception")
except Exception as e:
    check("raised", True, type(e).__name__)

# ── 4. Transient 500 is retried rather than fatal ───────────────────────────
print("\n4. TRANSIENT 500 IS RETRIED")
reset({0: ("flaky", 1, 5, 5)})      # 500 once, then 5 hits / total 5
rows, warn = spinoff("spin-off", 120)
check("recovered after one 500", len(rows) == 5, f"got {len(rows)}")
check("retried the same offset", fts_offsets().count(0) >= 2, f"offsets={fts_offsets()}")

# ── 5. Rate limiting between SEC calls ──────────────────────────────────────
print("\n5. RATE LIMITED UNDER 10 req/s")
reset({0: ("ok", 10, 40), 10: ("ok", 10, 40), 20: ("ok", 10, 40), 30: ("ok", 10, 40)})
t0 = time.monotonic()
rows, warn = spinoff("spin-off", 120)
elapsed = time.monotonic() - t0
gaps = [b[2] - a[2] for a, b in zip(CALLS, CALLS[1:])]
worst = min(gaps) if gaps else 1
check("all calls spaced >=0.15s", worst >= 0.14, f"min gap {worst*1000:.0f}ms")
check(f"{len(CALLS)} calls in {elapsed:.1f}s => under 10/s",
      len(CALLS) / max(elapsed, 1e-9) < 10, f"{len(CALLS)/max(elapsed,1e-9):.1f} req/s")

print("\nRESULT:", "all checks passed" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
