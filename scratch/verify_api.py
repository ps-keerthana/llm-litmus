"""Live endpoint verification for api/app.py"""
import urllib.request
import json
import sys
import time

time.sleep(2)  # give the server a moment if just started

BASE = "http://127.0.0.1:8000"
all_ok = True


def get(path):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
            return r.status, json.loads(r.read())
    except Exception as exc:
        return None, str(exc)


# -- Basic endpoints ----------------------------------------------------------
for path in ["/health", "/runs", "/runs/latest", "/history"]:
    status, data = get(path)
    if status is None:
        print(f"  FAIL  GET {path}  ->  {data}")
        all_ok = False
    elif isinstance(data, list):
        print(f"  OK    GET {path}  ->  {status}  ({len(data)} items)")
    else:
        keys = list(data.keys())[:6]
        print(f"  OK    GET {path}  ->  {status}  keys={keys}")

# -- Parameterized endpoints --------------------------------------------------
status, runs = get("/runs")
if status and runs:
    run_id = runs[0]["run_id"]
    for path in [f"/runs/{run_id}", f"/runs/{run_id}/results", f"/queue/{run_id}"]:
        s, d = get(path)
        if s is None:
            print(f"  FAIL  GET {path}  ->  {d}")
            all_ok = False
        elif isinstance(d, list):
            print(f"  OK    GET {path}  ->  {s}  ({len(d)} items)")
        else:
            print(f"  OK    GET {path}  ->  {s}  run_id={d.get('run_id')}")
else:
    print("  SKIP  No runs found, skipping parameterized checks")

# -- POST /runs/enqueue -------------------------------------------------------
try:
    body = json.dumps({"mode": "smoke", "no_judge": True}).encode()
    req = urllib.request.Request(
        f"{BASE}/runs/enqueue",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        d = json.loads(r.read())
        print(f"  OK    POST /runs/enqueue  ->  {r.status}  run_id={d.get('run_id')}")
except Exception as exc:
    print(f"  FAIL  POST /runs/enqueue  ->  {exc}")
    all_ok = False

print()
if all_ok:
    print("[SUCCESS] All API endpoints verified successfully!")
else:
    print("[FAIL] Some endpoints failed — see above.")
    sys.exit(1)
