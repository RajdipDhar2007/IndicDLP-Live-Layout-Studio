"""Kaggle version of cv_finetune.py (training only), for YOLOv10x and DocLayout-YOLO.

Run it in a Kaggle notebook with a GPU and Internet turned on:

    !pip -q install doclayout-yolo ultralytics
    !python /kaggle/input/<dataset>/kaggle_train_cv.py

It expects the dataset folder to contain  batch_06/ (page images), labels/ (one JSON per page, 4 classes) and
doclayout_yolo_indicdlp.pt and md2_PT_indicdlp_FT.pt, builds the same 5 folds + final split as cv_finetune.py and writes the six trained
checkpoints to /kaggle/working/<model>_cv/ (plus a zip each). Finished folds are skipped if the notebook is re-run.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # our own trusted checkpoints are full pickles

import numpy as np
from PIL import Image

if not hasattr(np, "trapz"):  # doclayout_yolo calls np.trapz, which NumPy 2 renamed
    np.trapz = np.trapezoid

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else next(p for p in Path("/kaggle/input").iterdir() if (p / "batch_06").exists())
WORK = Path("/kaggle/working")
DATA = WORK / "data"
RUNS = WORK / "runs"
OUT = WORK / "doclayout_cv"
CLASSES = ["Headline", "Subtitle", "BodyText", "Caption"]
K, EPOCHS, IMGSZ = 5, 40, 1024


def pages():
    return sorted(p for p in (SRC / "batch_06").iterdir() if p.suffix.lower() == ".jpg")


def make_data():
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    (DATA / "labels").mkdir(parents=True, exist_ok=True)
    files = pages()
    for f in files:
        dst = DATA / "images" / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        w, h = Image.open(f).size
        d = json.loads((SRC / "labels" / (f.stem + ".json")).read_text(encoding="utf-8"))
        lines = []
        for r in d["regions"]:
            if r.get("label") not in CLASSES:
                continue
            x1, y1, x2, y2 = r["box_xyxy"]
            x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{CLASSES.index(r['label'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (DATA / "labels" / (f.stem + ".txt")).write_text("\n".join(lines))
    for k in list(range(K)) + ["final"]:   # identical to cv_finetune.make_data()
        test = [f for i, f in enumerate(files) if k != "final" and i % K == k]
        rest = [f for f in files if f not in test]
        val = rest[::8][:10]
        train = [f for f in rest if f not in val]
        for name, fs in (("train", train), ("val", val)):
            (DATA / f"fold{k}_{name}.txt").write_text("\n".join(str(DATA / "images" / f.name) for f in fs))
        y = [f"path: {DATA.as_posix()}", f"train: fold{k}_train.txt", f"val: fold{k}_val.txt", "names:"]
        y += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
        (DATA / f"fold{k}.yaml").write_text("\n".join(y) + "\n")


MODELS = {"yolov10x": "md2_PT_indicdlp_FT.pt", "doclayout": "doclayout_yolo_indicdlp.pt"}


def weights(key):
    """The released checkpoint: from the dataset folder if present, else downloaded from the official Zenodo record."""
    f = SRC / MODELS[key]
    if f.exists():
        return f
    import urllib.request
    f = WORK / MODELS[key]
    if not f.exists():
        urllib.request.urlretrieve(f"https://zenodo.org/records/15881917/files/{MODELS[key]}?download=1", f)
    return f


def train_model(key):
    if key == "doclayout":
        from doclayout_yolo import YOLOv10 as YOLO
        import doclayout_yolo.engine.trainer as tr
        tr.check_amp = lambda model: True  # the AMP self-test downloads a file; skip it
    else:
        from ultralytics import YOLO
    out = WORK / f"{key}_cv"
    out.mkdir(parents=True, exist_ok=True)
    for k in list(range(K)) + ["final"]:
        name = f"{key}_fold{k}"
        dst = out / f"{name}.pt"
        if dst.exists():
            print("[skip]", name, flush=True)
            continue
        t = time.time()
        m = YOLO(str(weights(key)))
        best = RUNS / name / "weights" / "best.pt"
        try:
            m.train(data=str(DATA / f"fold{k}.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=2, workers=2, patience=15,
                    fliplr=0.0, flipud=0.0, degrees=0.0, project=str(RUNS), name=name, exist_ok=True, seed=0, verbose=False, plots=False)
        except AttributeError as e:   # DocLayout's final fused validation trips over G2L's BN; weights are already saved
            if "'bn'" not in str(e) or not best.exists():
                raise
            print("[note] final fused validation skipped", flush=True)
        shutil.copy2(best, dst)
        print(f"[done] {name} in {(time.time() - t) / 60:.1f} min", flush=True)
    shutil.make_archive(str(WORK / f"{key}_cv"), "zip", out)


def main():
    make_data()
    for key in (sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS):
        train_model(key)
    print("all done")


if __name__ == "__main__":
    main()
