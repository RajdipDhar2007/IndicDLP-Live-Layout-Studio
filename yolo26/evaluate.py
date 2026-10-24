"""Evaluate a trained detector on the IndicDLP validation split and print mAP.

Run:  python yolo26/evaluate.py --data P:/indicdlp_yolo/indicdlp.yaml --weights weights/yolo26_indicdlp.pt
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="weights/yolo26_indicdlp.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    m = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, batch=args.batch, split=args.split, plots=False)
    print(f"mAP50-95 {m.box.map * 100:.1f} | mAP50 {m.box.map50 * 100:.1f} | "
          f"precision {m.box.mp * 100:.1f} | recall {m.box.mr * 100:.1f}")


if __name__ == "__main__":
    main()
