"""Fine-tune YOLO26 on IndicDLP (42 layout classes).

Run:  python yolo26/train.py --data P:/indicdlp_yolo/indicdlp.yaml --model yolo26l.pt --epochs 30
The best checkpoint is copied to weights/yolo26_indicdlp.pt, which is the file app.py loads.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset yaml written by prepare_data.py")
    ap.add_argument("--model", default="yolo26l.pt", help="pretrained YOLO26 checkpoint (n, s, m, l or x)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=-1, help="-1 picks the largest batch that fits in GPU memory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="yolo26_indicdlp")
    ap.add_argument("--lr0", type=float, default=None, help="initial learning rate (default: ultralytics auto)")
    ap.add_argument("--close-mosaic", type=int, default=5, help="epochs at the end without mosaic")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run of the same --name")
    args = ap.parse_args()

    runs = ROOT / "runs"
    if args.resume:
        model = YOLO(runs / args.name / "weights" / "last.pt")
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        # Document pages are never mirrored or rotated, so flips are off and mosaic is closed early.
        extra = {"lr0": args.lr0, "optimizer": "AdamW"} if args.lr0 else {}
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                    project=str(runs), name=args.name, exist_ok=True, fliplr=0.0, flipud=0.0, degrees=0.0,
                    mosaic=1.0, close_mosaic=args.close_mosaic, cos_lr=True, amp=True, patience=args.patience, plots=True, **extra)
    best = runs / args.name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "weights").mkdir(exist_ok=True)
        shutil.copy2(best, ROOT / "weights" / "yolo26_indicdlp.pt")
        print(f"Copied {best} -> weights/yolo26_indicdlp.pt")


if __name__ == "__main__":
    main()
