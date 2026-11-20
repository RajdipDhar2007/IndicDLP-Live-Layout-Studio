"""Evaluates single models, the multi-scale boost and ensemble fusion on labelled pages, with per-class AP,
and tunes the per-model fusion weights.

Mode 1: the IndicDLP dataset in YOLO format (images + one .txt label file per image: `class cx cy w h`, normalised).
It is NOT part of this repository: https://huggingface.co/datasets/ai4bharat/indicdlp

  python evaluate_fusion.py --images D:/indic/images/test --labels D:/indic/labels/test \
         --tune-images D:/indic/images/val --tune-labels D:/indic/labels/val --limit 500

Weights are tuned on the validation split and the numbers are reported on the test split, so the test split is never
used to choose a setting.

Mode 2: a human-corrected Label Studio export (as produced by batch_06_ensemble/export_annotations.py). No download needed.

  python evaluate_fusion.py --ls-images <export>/images --ls-json <export>/json --out-prefix batch06

The pages are split by position: every even page tunes the fusion weights, every odd page is the held-out test set.
The ten annotation classes (Headline, Subtitle, BodyText, Caption, Image, Header, Footer, Folio, PageNumber, Flag) are
scored; the detectors' 42 classes are folded onto them by LS_CLASS_MAP.

Outputs: [prefix_]evaluation_results.json, [prefix_]evaluation_per_class.md and [prefix_]fusion_weights.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from detmetrics import evaluate

ROOT = Path(__file__).resolve().parent
SPECS = {"yolov10x": ("md2_PT_indicdlp_FT.pt", "yolo"), "doclayout": ("doclayout_yolo_indicdlp.pt", "doclayout")}
CONF_AP = 0.001  # AP needs the full precision-recall curve, so the confidence cut-off is very low
SCALE_IOU, MODEL_IOU = 0.60, 0.55
WEIGHT_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)

# Folds the detectors' 42 classes onto the ten classes of the Label Studio annotation scheme.
# Assumption: advertisements were annotated as Image, and every other text-like class as BodyText.
LS_CLASSES = ["Headline", "Subtitle", "BodyText", "Caption", "Image", "Header", "Footer", "Folio", "PageNumber", "Flag"]
LS_CLASS_MAP = {
    "headline": "Headline",
    "sub-headline": "Subtitle", "subsub-headline": "Subtitle", "chapter-title": "Subtitle", "section-title": "Subtitle",
    "sub-section-title": "Subtitle", "subsub-section-title": "Subtitle",
    "figure": "Image", "table": "Image", "formula": "Image", "advertisement": "Image",
    "figure-caption": "Caption", "table-caption": "Caption",
    "header": "Header", "footer": "Footer", "folio": "Folio", "page-number": "PageNumber", "flag": "Flag",
}


def class_names() -> list[str]:
    names: dict[int, str] = {}
    for line in (ROOT / "data" / "indic.yaml").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+(\d+):\s*'(.+)'", line)
        if m:
            names[int(m[1])] = m[2]
    return [names[i] for i in range(len(names))]


def ls_remap(names: list[str]) -> list[int]:
    """For every detector class id, its index in LS_CLASSES (anything not listed in LS_CLASS_MAP is BodyText)."""
    return [LS_CLASSES.index(LS_CLASS_MAP.get(n, "BodyText")) for n in names]


def load_ls_gt(img_path: Path, json_dir: Path) -> list[dict]:
    d = json.loads((json_dir / (img_path.stem + ".json")).read_text(encoding="utf-8"))
    return [{"cls": LS_CLASSES.index(r["label"]), "box": tuple(r["box_xyxy"])}
            for r in d["regions"] if r.get("label") in LS_CLASSES]


def load_gt(img_path: Path, labels_dir: Path, size: tuple[int, int]) -> list[dict]:
    f = labels_dir / (img_path.stem + ".txt")
    w, h = size
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append({"cls": c, "box": ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)})
    return out


def load_model(key: str):
    f, kind = SPECS[key]
    path = ROOT / "weights" / f
    if kind == "yolo":
        from ultralytics import YOLO
        return YOLO(str(path))
    from doclayout_yolo import YOLOv10
    return YOLOv10(str(path))


def predict(model, img: Image.Image, imgsz: int, device) -> list[dict]:
    bgr = np.ascontiguousarray(np.array(img.convert("RGB"))[:, :, ::-1])
    r = list(model.predict(bgr, imgsz=imgsz, conf=CONF_AP, iou=0.7, device=device, verbose=False))[0]
    return [{"box": tuple(b), "conf": float(c), "cls": int(k)}
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


def wbf(lists: list[list[dict]], size: tuple[int, int], iou_thr: float, weights: list[float] | None = None) -> list[dict]:
    from ensemble_boxes import weighted_boxes_fusion
    w, h = size
    B, S, L = [], [], []
    for dets in lists:
        B.append([[min(max(v / s, 0.0), 1.0) for v, s in zip(d["box"], (w, h, w, h))] for d in dets] or [[0, 0, 0, 0]])
        S.append([d["conf"] for d in dets] or [0.0])
        L.append([d["cls"] for d in dets] or [0])
    fb, fs, fl = weighted_boxes_fusion(B, S, L, weights=weights, iou_thr=iou_thr, skip_box_thr=CONF_AP, conf_type="max")
    return [{"box": (a * w, b * h, c * w, d * h), "conf": float(s), "cls": int(k)} for (a, b, c, d), s, k in zip(fb, fs, fl)]


def list_images(images: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if limit and len(files) > limit:
        files = sorted(random.Random(0).sample(files, limit))
    return files


def collect(files: list[Path], gt_of, keys: list[str], imgsz: int, device) -> tuple[list, list, list]:
    """Runs every model at two sizes once per image and caches the raw predictions, so fusion settings can be re-scored cheaply.
    gt_of(path, (width, height)) returns that page's ground-truth boxes."""
    models = {k: load_model(k) for k in keys}
    second = 1280 if imgsz < 1280 else 1024
    sizes, gts, raw = [], [], []
    for n, f in enumerate(files, 1):
        img = Image.open(f).convert("RGB")
        sizes.append(img.size)
        gts.append(gt_of(f, img.size))
        raw.append({k: (predict(m, img, imgsz, device), predict(m, img, second, device)) for k, m in models.items()})
        if n % 10 == 0 or n == len(files):
            print(f"  predicted {n}/{len(files)} pages", flush=True)
    return sizes, gts, raw


def build(raw: list, sizes: list, keys: list[str], mode: str, weights: dict[str, float] | None = None) -> list[list[dict]]:
    """mode: 'single:<key>', 'boost:<key>' or 'ensemble'."""
    out = []
    for r, size in zip(raw, sizes):
        if mode.startswith("single:"):
            out.append(r[mode.split(":")[1]][0])
        elif mode.startswith("boost:"):
            out.append(wbf(list(r[mode.split(":")[1]]), size, SCALE_IOU))
        else:
            per_model = [wbf(list(r[k]), size, SCALE_IOU) for k in keys]
            out.append(wbf(per_model, size, MODEL_IOU, [weights[k] for k in keys] if weights else None))
    return out


def score(preds, gts, n_classes, remap: list[int] | None = None) -> dict:
    if remap:  # fold detector classes onto the annotation classes
        preds = [[{**p, "cls": remap[p["cls"]]} for p in img] for img in preds]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # classes absent from a small subset
        return evaluate(preds, gts, n_classes)


def tune_weights(raw, sizes, gts, keys, n_classes, remap=None) -> tuple[dict[str, float], list[dict]]:
    """Grid search over the model weights of ensemble fusion (first model fixed at 1.0), scored by mAP50-95."""
    trials = []
    for ws in itertools.product(WEIGHT_GRID, repeat=len(keys) - 1):
        weights = dict(zip(keys, (1.0, *ws)))
        m = score(build(raw, sizes, keys, "ensemble", weights), gts, n_classes, remap)
        trials.append({"weights": weights, "map50_95": m["map50_95"], "map50": m["map50"]})
        print(f"  weights {weights}: mAP50-95 {m['map50_95']:.4f}", flush=True)
    best = max(trials, key=lambda t: t["map50_95"])
    return best["weights"], trials


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="test images folder (IndicDLP YOLO format)")
    ap.add_argument("--labels", type=Path, help="test YOLO label folder")
    ap.add_argument("--tune-images", type=Path, help="validation images, used only to tune the fusion weights")
    ap.add_argument("--tune-labels", type=Path)
    ap.add_argument("--ls-images", type=Path, help="Label Studio export: images folder")
    ap.add_argument("--ls-json", type=Path, help="Label Studio export: per-page json folder")
    ap.add_argument("--out-prefix", default="", help="prefix for the output files, e.g. batch06")
    ap.add_argument("--models", default="yolov10x,doclayout")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, help="evaluate a random subset of this many pages per split")
    ap.add_argument("--device", default=None, help="0 for the first GPU, or cpu (default: GPU if available)")
    a = ap.parse_args()
    if not (a.ls_images and a.ls_json) and not (a.images and a.labels):
        ap.error("give --images and --labels, or --ls-images and --ls-json")

    import torch
    device = a.device if a.device is not None else (0 if torch.cuda.is_available() else "cpu")
    keys = [k.strip() for k in a.models.split(",") if k.strip()]
    detector_names = class_names()
    t0 = time.time()

    if a.ls_images:
        files = list_images(a.ls_images)
        tune_files, test_files = files[0::2], files[1::2]
        names, remap = LS_CLASSES, ls_remap(detector_names)

        def gt_of(f, size):
            return load_ls_gt(f, a.ls_json)
        tune_gt_of = gt_of
        print(f"Label Studio export: {len(tune_files)} pages tune the weights, {len(test_files)} held-out pages are scored.")
    else:
        names, remap = detector_names, None

        def gt_of(f, size):
            return load_gt(f, a.labels, size)
        test_files = list_images(a.images, a.limit)
        tune_files = list_images(a.tune_images, a.limit) if a.tune_images else []

        def tune_gt_of(f, size):
            return load_gt(f, a.tune_labels, size)
    n = len(names)
    pre = f"{a.out_prefix}_" if a.out_prefix else ""

    weights = None
    trials: list[dict] = []
    if tune_files and len(keys) > 1:
        print("Tuning fusion weights on the tuning pages ...")
        sizes, gts, raw = collect(tune_files, tune_gt_of, keys, a.imgsz, device)
        weights, trials = tune_weights(raw, sizes, gts, keys, n, remap)
        print("Best weights:", weights)
        (ROOT / f"{pre}fusion_weights.json").write_text(json.dumps(weights, indent=2))
        del raw

    print("Evaluating on the held-out test pages ...")
    sizes, gts, raw = collect(test_files, gt_of, keys, a.imgsz, device)
    runs = {}
    for k in keys:
        runs[f"{k} (single scale)"] = build(raw, sizes, keys, f"single:{k}")
        runs[f"{k} (multi-scale boost)"] = build(raw, sizes, keys, f"boost:{k}")
    if len(keys) > 1:
        runs["ensemble (equal weights)"] = build(raw, sizes, keys, "ensemble")
        if weights:
            runs["ensemble (tuned weights)"] = build(raw, sizes, keys, "ensemble", weights)
    results = {name: score(p, gts, n, remap) for name, p in runs.items()}

    out = {"pages": len(gts), "tune_pages": len(tune_files), "classes": names, "imgsz": a.imgsz, "tuned_weights": weights,
           "tuning_trials": trials, "results": results, "seconds": round(time.time() - t0)}
    (ROOT / f"{pre}evaluation_results.json").write_text(json.dumps(out, indent=2))

    lines = ["| Pipeline | mAP50 | mAP50-95 |", "|---|---|---|"]
    lines += [f"| {k} | {v['map50'] * 100:.1f} | {v['map50_95'] * 100:.1f} |" for k, v in results.items()]
    lines += ["", "| Class | GT boxes | " + " | ".join(f"{k} AP50-95" for k in results) + " |", "|---|---|" + "---|" * len(results)]
    for c, name in enumerate(names):
        row = [results[k]["ap50_95"][c] for k in results]
        lines.append(f"| {name} | {next(iter(results.values()))['n_gt'][c]} | "
                     + " | ".join("-" if np.isnan(v) else f"{v * 100:.1f}" for v in row) + " |")
    (ROOT / f"{pre}evaluation_per_class.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {pre}evaluation_results.json and {pre}evaluation_per_class.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
