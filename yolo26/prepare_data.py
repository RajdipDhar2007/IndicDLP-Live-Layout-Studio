"""Convert IndicDLP parquet shards (Hugging Face `ai4bharat/indicdlp`) into the YOLO detection format.

Each shard row holds an image, COCO-style [x, y, w, h] pixel boxes and class ids 0-41. The output is
    <out>/images/{train,val}/<name>.<ext>   and   <out>/labels/{train,val}/<name>.txt
plus <out>/indicdlp.yaml, ready for `python yolo26/train.py --data <out>/indicdlp.yaml`.

Run:  python yolo26/prepare_data.py --raw P:/indicdlp_raw/data --out P:/indicdlp_yolo
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from PIL import Image

CLASS_NAMES = [
    "advertisement", "answer", "author", "chapter-title", "contact-info", "dateline", "figure", "figure-caption",
    "first-level-question", "flag", "folio", "footer", "footnote", "formula", "header", "headline", "index", "jumpline",
    "options", "ordered-list", "page-number", "paragraph", "placeholder-text", "quote", "reference",
    "second-level-question", "section-title", "sidebar", "sub-headline", "sub-ordered-list", "sub-section-title",
    "subsub-ordered-list", "subsub-section-title", "sub-unordered-list", "subsub-headline", "subsub-unordered-list",
    "table", "table-caption", "table-of-contents", "third-level-question", "unordered-list", "website-link",
]
SPLITS = {"train": "train", "validation": "val"}


def convert_shard(path: Path, split: str, out: Path) -> tuple[int, int]:
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    pages = boxes = 0
    pf = pq.ParquetFile(path)
    stem = path.name.split("-of-")[0]  # e.g. train-00008
    for g in range(pf.num_row_groups):
        for i, row in enumerate(pf.read_row_group(g).to_pylist()):
            data = row["image"]["bytes"]
            w, h = Image.open(io.BytesIO(data)).size
            lines = []
            for (x, y, bw, bh), c in zip(row["bboxes"], row["category_ids"]):
                x1, y1, x2, y2 = max(x, 0.0), max(y, 0.0), min(x + bw, w), min(y + bh, h)
                if x2 - x1 < 1 or y2 - y1 < 1 or not 0 <= c < len(CLASS_NAMES):
                    continue
                lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            name = f"{stem}_{g}_{i}"
            (img_dir / f"{name}.png").write_bytes(data)
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            pages += 1
            boxes += len(lines)
    return pages, boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder holding the downloaded *.parquet shards")
    ap.add_argument("--out", type=Path, required=True, help="output dataset folder")
    args = ap.parse_args()

    total = {"train": [0, 0], "val": [0, 0]}
    for hf_split, split in SPLITS.items():
        for shard in sorted(args.raw.glob(f"{hf_split}-*.parquet")):
            p, b = convert_shard(shard, split, args.out)
            total[split][0] += p
            total[split][1] += b
            print(f"{shard.name}: {p} pages, {b} boxes", flush=True)

    cfg = {"path": args.out.resolve().as_posix(), "train": "images/train", "val": "images/val",
           "names": dict(enumerate(CLASS_NAMES))}
    (args.out / "indicdlp.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"train {total['train'][0]} pages / {total['train'][1]} boxes; val {total['val'][0]} pages / {total['val'][1]} boxes")


if __name__ == "__main__":
    main()
