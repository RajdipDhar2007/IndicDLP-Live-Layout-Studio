"""Statistics of the human-checked OCR text of the 100 Batch 06 pages.

The text comes from the per-page JSON files in Rajdip_Dhar/2_readable_export/json (one file per page; every
region lists its class, its box in pixels and its transcription). No OCR engine is run here: these
transcriptions are the project's OCR reference, and they are what an OCR engine would be scored against.

Run:  python ocr_reference_stats.py [--json-dir DIR] [--out batch06_ocr_reference.json]
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def script_of(ch: str) -> str:
    o = ord(ch)
    if 0x0980 <= o <= 0x09FF:
        return "bengali"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit() or 0x09E6 <= o <= 0x09EF:
        return "digit"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default=str(ROOT / "Rajdip_Dhar" / "2_readable_export" / "json"))
    ap.add_argument("--out", default=str(ROOT / "batch06_ocr_reference.json"))
    a = ap.parse_args()

    files = sorted(Path(a.json_dir).glob("*.json"))
    per_class = collections.defaultdict(lambda: dict(regions=0, with_text=0, chars=0))
    scripts = collections.Counter()
    page_regions, page_chars = [], []
    for f in files:
        page = json.loads(f.read_text(encoding="utf-8"))
        regions = page["regions"]
        page_regions.append(len(regions))
        chars = 0
        for r in regions:
            text = (r.get("ocr_text") or "").strip()
            c = per_class[r["label"]]
            c["regions"] += 1
            if text:
                c["with_text"] += 1
                c["chars"] += len(text)
                chars += len(text)
                scripts.update(script_of(ch) for ch in text if not ch.isspace())
        page_chars.append(chars)

    total = {k: sum(v[k] for v in per_class.values()) for k in ("regions", "with_text", "chars")}
    letters = sum(scripts[k] for k in ("bengali", "latin"))
    out = dict(
        source=str(Path(a.json_dir).relative_to(ROOT)) if Path(a.json_dir).is_relative_to(ROOT) else str(a.json_dir),
        pages=len(files), **total,
        mean_chars_per_region=round(total["chars"] / max(total["with_text"], 1), 1),
        mean_regions_per_page=round(statistics.mean(page_regions), 1), median_regions_per_page=statistics.median(page_regions),
        mean_chars_per_page=round(statistics.mean(page_chars)),
        per_class={k: dict(v, mean_chars=round(v["chars"] / max(v["with_text"], 1), 1)) for k, v in sorted(per_class.items())},
        script_share_of_letters=dict(bengali=round(scripts["bengali"] / max(letters, 1), 4), latin=round(scripts["latin"] / max(letters, 1), 4)),
    )
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
