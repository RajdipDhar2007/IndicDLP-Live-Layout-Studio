"""Fine-tunes YOLOv10x (from the IndicDLP checkpoint) on human-annotated Bengali newspaper pages (a Label Studio export),
then scores it on pages that were never used for training or checkpoint selection.

  python finetune_batch06.py --export <export_dir> [--epochs 40]

Split of the 100 pages (by position in the sorted file list), identical to evaluate_fusion.py:
  odd-numbered pages (50)            held-out TEST pages, never touched by training or model selection
  even-numbered pages, 40 of 50      training
  even-numbered pages, last 10 of 50 validation (only used by Ultralytics to keep the best checkpoint)

The head is re-initialised for the 10 annotation classes (Headline, Subtitle, BodyText, Caption, Image, Header, Footer,
Folio, PageNumber, Flag); the backbone starts from the IndicDLP weights. Writes batch06_finetune_results.json and
weights/yolov10x_batch06_ft.pt.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

from detmetrics import evaluate
from evaluate_fusion import LS_CLASSES, list_images, load_ls_gt, predict, wbf, SCALE_IOU

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "weights" / "md2_PT_indicdlp_FT.pt"
OUT = ROOT / "weights" / "yolov10x_batch06_ft.pt"
DATA = ROOT / "data" / "batch06_yolo"  # generated; not committed


def make_dataset(images_dir: Path, json_dir: Path) -> dict[str, list[Path]]:
    files = list_images(images_dir)
    even, odd = files[0::2], files[1::2]
    splits = {"train": even[:-10], "val": even[-10:], "test": odd}
    for name, fs in splits.items():
        (DATA / "images" / name).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / name).mkdir(parents=True, exist_ok=True)
        for f in fs:
            shutil.copy2(f, DATA / "images" / name / f.name)
            w, h = Image.open(f).size
            lines = []
            for g in load_ls_gt(f, json_dir):
                x1, y1, x2, y2 = g["box"]
                x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(f"{g['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (DATA / "labels" / name / (f.stem + ".txt")).write_text("\n".join(lines))
    yaml = ["path: " + DATA.as_posix(), "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(LS_CLASSES)]
    (DATA / "batch06.yaml").write_text("\n".join(yaml) + "\n")
    return splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--skip-train", action="store_true", help="only evaluate an existing weights/yolov10x_batch06_ft.pt")
    a = ap.parse_args()
    splits = make_dataset(a.export / "images", a.export / "json")
    print({k: len(v) for k, v in splits.items()})

    from ultralytics import YOLO
    if not a.skip_train:
        model = YOLO(str(BASE))
        model.train(data=str(DATA / "batch06.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=2, workers=2, patience=15,
                    fliplr=0.0, flipud=0.0, degrees=0.0, project=str(ROOT / "runs"), name="finetune_batch06",
                    exist_ok=True, seed=0, verbose=False)
        best = ROOT / "runs" / "finetune_batch06" / "weights" / "best.pt"
        shutil.copy2(best, OUT)

    model = YOLO(str(OUT))
    second = 1280 if a.imgsz < 1280 else 1024
    sizes, gts, single, boost = [], [], [], []
    for f in splits["test"]:
        img = Image.open(f).convert("RGB")
        sizes.append(img.size)
        gts.append(load_ls_gt(f, a.export / "json"))
        p1 = predict(model, img, a.imgsz, 0)
        p2 = predict(model, img, second, 0)
        single.append(p1)
        boost.append(wbf([p1, p2], img.size, SCALE_IOU))
    out = {"test_pages": len(gts), "train_pages": len(splits["train"]), "classes": LS_CLASSES,
           "single": evaluate(single, gts, len(LS_CLASSES)), "boost": evaluate(boost, gts, len(LS_CLASSES))}
    (ROOT / "batch06_finetune_results.json").write_text(json.dumps(out, indent=2))
    for k in ("single", "boost"):
        print(k, f"mAP50 {out[k]['map50'] * 100:.1f}  mAP50-95 {out[k]['map50_95'] * 100:.1f}")
    for c, n in enumerate(LS_CLASSES):
        print(f"{n:11s} n={out['single']['n_gt'][c]:5d}  single {out['single']['ap50_95'][c] * 100:5.1f}  boost {out['boost']['ap50_95'][c] * 100:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
