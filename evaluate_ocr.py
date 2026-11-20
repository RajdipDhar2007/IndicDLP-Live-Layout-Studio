"""Character error rate (CER) of EasyOCR (Bengali + English) on annotated regions of a Label Studio export.

  python evaluate_ocr.py --export <export_dir> [--limit 250]

Reference text is the `ocr_text` stored with each region (Gemini transcription kept by the annotator, so it is a
silver standard, not independent ground truth). Boxes are the annotated boxes, so only the OCR step is measured,
not detection. Uses the odd-numbered (held-out) pages, as evaluate_fusion.py does.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
TEXT_CLASSES = {"Headline", "Subtitle", "BodyText", "Caption"}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    files = sorted((a.export / "images").glob("*.jpg"))[1::2]
    regions = []
    for f in files:
        d = json.loads((a.export / "json" / (f.stem + ".json")).read_text(encoding="utf-8"))
        for r in d["regions"]:
            ref = norm(r.get("ocr_text", ""))
            if r["label"] in TEXT_CLASSES and len(ref) >= 5:
                regions.append((f, r["box_xyxy"], r["label"], ref))
    random.Random(0).shuffle(regions)
    regions = regions[:a.limit]

    import easyocr
    reader = easyocr.Reader(["bn", "en"], gpu=not a.cpu, verbose=False)
    pages: dict[Path, Image.Image] = {}
    errs, chars, per_class, secs = 0, 0, {}, []
    for f, box, label, ref in regions:
        im = pages.setdefault(f, Image.open(f).convert("RGB"))
        x1, y1, x2, y2 = (int(v) for v in box)
        crop = im.crop((max(0, x1 - 4), max(0, y1 - 4), min(im.width, x2 + 4), min(im.height, y2 + 4)))
        t0 = time.perf_counter()
        try:
            hyp = norm(" ".join(str(t) for t in reader.readtext(np.array(crop), detail=0, paragraph=True)))
        except Exception:
            hyp = ""
        secs.append(time.perf_counter() - t0)
        e = edit_distance(hyp, ref)
        errs += e
        chars += len(ref)
        per_class.setdefault(label, [0, 0])
        per_class[label][0] += e
        per_class[label][1] += len(ref)
    out = {"regions": len(regions), "reference_chars": chars, "cer": errs / max(chars, 1),
           "cer_per_class": {k: v[0] / v[1] for k, v in per_class.items()},
           "median_ms_per_region": 1000 * statistics.median(secs), "engine": "EasyOCR bn+en"}
    (ROOT / "batch06_ocr_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
