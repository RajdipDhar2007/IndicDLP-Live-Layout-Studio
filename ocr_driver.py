"""Feed annotations 1..100 to the OCR backend one at a time, watching its log for completion.

The backend must be running with GEMINI_MIN_INTERVAL set (free tier: 15 req/min). Stops on a per-day quota error.
Each pass re-triggers pages that had any 429 so blank regions get filled; the backend skips regions that already have text.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "backend.log"
PROGRESS = HERE / "ocr_progress.json"
IDS = range(1, 101)
MAX_PASSES = 3


def trigger(i):
    req = urllib.request.Request("http://localhost:9090/webhook", method="POST",
                                 data=json.dumps({"action": "ANNOTATION_UPDATED", "annotation": {"id": i}}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()


def run_page(i):
    start = LOG.stat().st_size
    trigger(i)
    done = None
    while done is None:
        time.sleep(5)
        with open(LOG, "rb") as f:
            f.seek(start)
            new = f.read().decode("utf-8", "replace")
        if f"patched annotation {i} " in new or "patched annotation %s ===" % i in new:
            done = "patched"
        elif "Nothing to OCR" in new:
            done = "nothing"
        elif "OCR failed while processing webhook" in new or "Could not resolve task" in new or "No registered OCR key" in new:
            done = "error"
    with open(LOG, "rb") as f:
        f.seek(start)
        new = f.read().decode("utf-8", "replace")
    # In page mode a failed/short chunk is logged instead of a 429; count it so the page gets another pass.
    short = new.count("Crop chunk") + new.count("Chunk answered")
    return done, new.count("429 RESOURCE_EXHAUSTED") + short, "PerDay" in new


state = {}
for p in range(1, MAX_PASSES + 1):
    todo = [i for i in IDS if state.get(i, {}).get("status") != "complete"]
    if not todo:
        break
    for i in todo:
        done, n429, per_day = run_page(i)
        state[i] = dict(status="complete" if done in ("nothing", "patched") and n429 == 0 else "partial",
                        result=done, n429=n429, passes=p)
        PROGRESS.write_text(json.dumps(dict(pass_no=p, last=i, state=state)), encoding="utf-8")
        print(f"pass {p} annotation {i}: {done}, 429s={n429}", flush=True)
        if per_day:
            print("Daily quota reached - stopping.", flush=True)
            sys.exit(2)
        if done == "error":
            print("Backend error - stopping.", flush=True)
            sys.exit(3)
print("finished", flush=True)
