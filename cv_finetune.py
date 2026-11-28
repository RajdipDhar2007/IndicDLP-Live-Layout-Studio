"""5-fold cross-validated fine-tuning of YOLOv10x and DocLayout-YOLO on all 100 annotated Batch 06 pages,
plus one final model per detector trained on all 100 pages.

Labels: Rajdip_Dhar/2_readable_export/json (4 classes: Headline, Subtitle, BodyText, Caption; Image boxes were removed).

  python cv_finetune.py train            # 5 folds + final model, for both detectors (resumable)
  python cv_finetune.py eval             # scores every page with a model that never saw it, plus the untouched originals
  python cv_finetune.py smoke            # 1 epoch on fold 0 only, to check the setup

Folds: pages are sorted by name; page i is in test fold (i % 5). In each fold the 80 remaining pages are split into
70 for training and 10 for checkpoint selection only (every 8th page). The final model uses the same 10 pages for
checkpoint selection and the other 90 for training, so every page influences it.
Outputs: data/cv_batch06/, runs/cv_batch06/, weights/*_batch06_cv_final.pt, cv_batch06_results.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # DocLayout-YOLO checkpoints are full pickles (our own, trusted files)

import numpy as np
from PIL import Image

from detmetrics import evaluate
from evaluate_fusion import (SCALE_IOU, MODEL_IOU, class_names, ls_remap, predict, wbf, load_model)

if not hasattr(np, "trapz"):  # doclayout_yolo still calls np.trapz, which NumPy 2 renamed to trapezoid
    np.trapz = np.trapezoid

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "batch_06"
LABELS = ROOT / "Rajdip_Dhar" / "2_readable_export" / "json"
DATA = ROOT / "data" / "cv_batch06"
RUNS = ROOT / "runs" / "cv_batch06"
CLASSES = ["Headline", "Subtitle", "BodyText", "Caption"]
K = 5
EPOCHS = 40
IMGSZ = 1024
MODELS = {  # key -> (starting checkpoint, library)
    "yolov10x": ("md2_PT_indicdlp_FT.pt", "ultralytics"),
    "doclayout": ("doclayout_yolo_indicdlp.pt", "doclayout"),
}


TRAIN_KEYS = ["yolov10x"]  # only YOLOv10x is fine-tuned; DocLayout-YOLO stays as the untouched original (baseline only)


def pages() -> list[Path]:
    return sorted(p for p in IMAGES.iterdir() if p.suffix.lower() == ".jpg")


def gt_of(f: Path) -> list[dict]:
    d = json.loads((LABELS / (f.stem + ".json")).read_text(encoding="utf-8"))
    return [{"cls": CLASSES.index(r["label"]), "box": tuple(r["box_xyxy"])} for r in d["regions"] if r.get("label") in CLASSES]


def make_data() -> dict:
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    (DATA / "labels").mkdir(parents=True, exist_ok=True)
    files = pages()
    for f in files:
        dst = DATA / "images" / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        w, h = Image.open(f).size
        lines = []
        for g in gt_of(f):
            x1, y1, x2, y2 = g["box"]
            x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{g['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (DATA / "labels" / (f.stem + ".txt")).write_text("\n".join(lines))
    folds = {}
    for k in list(range(K)) + ["final"]:
        test = [f for i, f in enumerate(files) if k != "final" and i % K == k]
        rest = [f for f in files if f not in test]
        val = rest[::8][:10]
        train = [f for f in rest if f not in val]
        folds[k] = {"train": train, "val": val, "test": test}
        for name in ("train", "val"):
            (DATA / f"fold{k}_{name}.txt").write_text("\n".join(str(DATA / "images" / f.name) for f in folds[k][name]))
        yaml = [f"path: {DATA.as_posix()}", f"train: fold{k}_train.txt", f"val: fold{k}_val.txt", "names:"]
        yaml += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
        (DATA / f"fold{k}.yaml").write_text("\n".join(yaml) + "\n")
    return folds


def train_one(key: str, k, epochs: int) -> Path:
    name = f"{key}_fold{k}" + ("" if epochs == EPOCHS else "_smoke")
    best = RUNS / name / "weights" / "best.pt"
    if best.exists() and epochs == EPOCHS:
        print(f"[skip] {name}", flush=True)
        return best
    base = ROOT / "weights" / MODELS[key][0]
    if MODELS[key][1] == "ultralytics":
        from ultralytics import YOLO
    else:
        from doclayout_yolo import YOLOv10 as YOLO
        import doclayout_yolo.engine.trainer as _tr
        _tr.check_amp = lambda model: True  # the built-in AMP self-test downloads yolov8n.pt, which is not available offline
    t = time.time()
    m = YOLO(str(base))
    try:
        m.train(data=str(DATA / f"fold{k}.yaml"), epochs=epochs, imgsz=IMGSZ, batch=2, workers=2, patience=15,
                fliplr=0.0, flipud=0.0, degrees=0.0, project=str(RUNS), name=name, exist_ok=True, seed=0, verbose=False, plots=False)
    except AttributeError as e:
        # DocLayout-YOLO's final validation fuses Conv+BN and then its G2L module asks for the removed BN. The weights are
        # already saved by then, so this only costs the redundant final validation.
        if "'bn'" not in str(e) or not best.exists():
            raise
        print(f"[note] {name}: final fused validation skipped ({e})", flush=True)
    print(f"[done] {name} in {(time.time() - t) / 60:.1f} min", flush=True)
    return best


def cmd_train(epochs: int, ks) -> None:
    make_data()
    for key in TRAIN_KEYS:
        for k in ks:
            train_one(key, k, epochs)
    if epochs == EPOCHS:
        for key in TRAIN_KEYS:
            shutil.copy2(RUNS / f"{key}_foldfinal" / "weights" / "best.pt", ROOT / "weights" / f"{key}_batch06_cv_final.pt")


def fold_preds(files, models: dict, remap=None) -> list[dict]:
    """Per page: {key: (pred at 1024, pred at 1280)}."""
    out = []
    for f in files:
        img = Image.open(f).convert("RGB")
        out.append({k: (predict(m, img, IMGSZ, 0), predict(m, img, 1280, 0)) for k, m in models.items()})
    return out


def fold_remap(preds, remap):
    if remap is None:
        return preds
    return [[{**p, "cls": remap[p["cls"]]} for p in img if remap[p["cls"]] < len(CLASSES)] for img in preds]


def build(raw, sizes, keys, mode, remap=None):
    out = []
    for r, size in zip(raw, sizes):
        if mode.startswith("single:"):
            p = r[mode.split(":")[1]][0]
        elif mode.startswith("boost:"):
            p = wbf(list(r[mode.split(":")[1]]), size, SCALE_IOU)
        else:
            p = wbf([wbf(list(r[k]), size, SCALE_IOU) for k in keys], size, MODEL_IOU)
        out.append(p)
    return [fold_remap([p], remap)[0] for p in out]


def summarise(preds, gts):
    m = evaluate(preds, gts, len(CLASSES))
    return {"map50": m["map50"], "map50_95": m["map50_95"], "ap50_95": m["ap50_95"], "n_gt": m["n_gt"]}


def cmd_eval() -> None:
    folds = make_data()
    files = pages()
    sizes_all = {f: Image.open(f).size for f in files}
    keys = list(MODELS)  # originals use both detectors
    cv_keys = TRAIN_KEYS
    cache = ROOT / "data" / "cv_batch06" / "raw_preds.json"
    raw_cv: dict = json.loads(cache.read_text()) if cache.exists() else {}
    # cross-validated predictions of the fine-tuned models
    for k in range(K):
        if str(k) in raw_cv:
            continue
        models = {key: (__import__("ultralytics").YOLO if MODELS[key][1] == "ultralytics" else __import__("doclayout_yolo").YOLOv10)(
            str(RUNS / f"{key}_fold{k}" / "weights" / "best.pt")) for key in cv_keys}
        raw = fold_preds(folds[k]["test"], models)
        raw_cv[str(k)] = {f.name: r for f, r in zip(folds[k]["test"], raw)}
        cache.write_text(json.dumps(raw_cv))
        print(f"fold {k}: predicted {len(raw)} pages", flush=True)
    # originals (never fine-tuned) on all 100 pages
    orig_cache = ROOT / "data" / "cv_batch06" / "raw_orig.json"
    if orig_cache.exists():
        raw_orig = json.loads(orig_cache.read_text())
    else:
        models = {key: load_model(key) for key in keys}
        raw_orig = {f.name: r for f, r in zip(files, fold_preds(files, models))}
        orig_cache.write_text(json.dumps(raw_orig))
    remap = ls_remap(class_names())

    def tuplify(r):
        return {k: tuple([{**d, "box": tuple(d["box"])} for d in lst] for lst in v) for k, v in r.items()}
    gts = [gt_of(f) for f in files]
    sizes = [sizes_all[f] for f in files]
    cv_raw = [tuplify(next(raw_cv[str(k)][f.name] for k in range(K) if f.name in raw_cv[str(k)])) for f in files]
    or_raw = [tuplify(raw_orig[f.name]) for f in files]
    res = {"pages": len(files), "classes": CLASSES, "folds": K, "epochs": EPOCHS}
    for label, raw, rm, modes in (("original", or_raw, remap, ("single:yolov10x", "boost:yolov10x", "single:doclayout", "boost:doclayout", "ensemble")),
                                  ("finetuned_cv", cv_raw, None, ("single:yolov10x", "boost:yolov10x"))):
        res[label] = {}
        for mode in modes:
            res[label][mode] = summarise(build(raw, sizes, keys, mode, rm), gts)
            r = res[label][mode]
            print(f"{label:13s} {mode:18s} mAP50 {r['map50'] * 100:5.1f}  mAP50-95 {r['map50_95'] * 100:5.1f}", flush=True)
    # per fold mAP50-95 of the cross-validated ensemble (spread between folds)
    res["fold_boost_map50_95"] = []
    for k in range(K):
        idx = [i for i, f in enumerate(files) if f in folds[k]["test"]]
        p = build([cv_raw[i] for i in idx], [sizes[i] for i in idx], cv_keys, "boost:yolov10x")
        res["fold_boost_map50_95"].append(evaluate(p, [gts[i] for i in idx], len(CLASSES))["map50_95"])
    (ROOT / "cv_batch06_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["fold_boost_map50_95"]))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "smoke":
        cmd_train(1, [0])
    elif cmd == "train":
        cmd_train(EPOCHS, list(range(K)) + ["final"])
    elif cmd == "eval":
        cmd_eval()
    else:
        sys.exit(__doc__)
