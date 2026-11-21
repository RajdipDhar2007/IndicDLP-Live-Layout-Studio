"""Export every Label Studio annotation + OCR transcription, plus images drawn with the bounding boxes.

Usage: python export_annotations.py [out_dir]      (default: export_<today> next to this script)

Output folder layout:
    all_annotations.json        every page in one file
    json/<image_stem>.json      one file per page (regions with label, box, OCR text)
    images/<image_stem>.jpg     the page image with labelled bounding boxes drawn on it
The Label Studio DB is opened read-only.
"""
import datetime
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
LS_DIR = Path(os.path.expanduser(r"~\AppData\Local\label-studio\label-studio"))
DB = LS_DIR / "label_studio.sqlite3"
MEDIA = LS_DIR / "media"
COLORS = {  # from the project's label config
    "Headline": "#D32F2F", "Subtitle": "#F57C00", "BodyText": "#4A90D9", "Caption": "#388E3C", "Image": "#7B1FA2",
    "Header": "#EC4899", "Footer": "#F43F5E", "Folio": "#DB2777", "PageNumber": "#FB7185", "Flag": "#EF4444",
}

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / f"export_{datetime.date.today().isoformat()}"
(out_dir / "json").mkdir(parents=True, exist_ok=True)
(out_dir / "images").mkdir(parents=True, exist_ok=True)

db = sqlite3.connect("file:" + str(DB).replace("\\", "/") + "?mode=ro", uri=True)
rows = db.execute(
    "select a.id, a.task_id, a.result, a.created_at, a.updated_at, t.data, f.file "
    "from task_completion a join task t on t.id = a.task_id "
    "left join data_import_fileupload f on f.id = t.file_upload_id "
    "where a.was_cancelled = 0 order by a.task_id, a.id").fetchall()


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def corners(v, W, H):
    """Four (x, y) pixel corners of a (possibly rotated) rectangle given in LS percent units."""
    x, y, w, h = v["x"] / 100 * W, v["y"] / 100 * H, v["width"] / 100 * W, v["height"] / 100 * H
    a = math.radians(v.get("rotation", 0) or 0)
    c, s = math.cos(a), math.sin(a)
    return [(x + px * c - py * s, y + px * s + py * c) for px, py in ((0, 0), (w, 0), (w, h), (0, h))]


pages = []
for ann_id, task_id, result, created, updated, data, file in rows:
    result = json.loads(result)
    data = json.loads(data)
    rel = file or next(iter(data.values()))
    img_path = MEDIA / rel.replace("/data/", "", 1).lstrip("/") if not file else MEDIA / file
    stem = Path(rel).name
    stem = stem.split("-", 1)[1] if len(stem.split("-", 1)[0]) == 8 else stem  # drop LS's 8-char upload prefix
    stem = Path(stem).stem

    text = {r["id"]: r["value"]["text"] for r in result if r["type"] == "textarea"}
    boxes = [r for r in result if r["type"] == "rectanglelabels"]
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(max(14, W // 90))
    lw = max(2, W // 700)

    regions = []
    for n, r in enumerate(boxes, 1):
        v = r["value"]
        pts = corners(v, W, H)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        label = v["rectanglelabels"][0] if v.get("rectanglelabels") else "Unknown"
        t = text.get(r["id"], [])
        regions.append(dict(
            n=n, region_id=r["id"], label=label,
            box_xyxy=[round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
            box_percent=dict(x=v["x"], y=v["y"], width=v["width"], height=v["height"], rotation=v.get("rotation", 0)),
            ocr_text="\n".join(t),
            ocr_lines=t,
            origin=r.get("origin"),
        ))
        col = COLORS.get(label, "#00AA00")
        draw.polygon(pts, fill=col + "30")
        draw.line(pts + [pts[0]], fill=col, width=lw)
        tag = f"{n} {label}"
        l, t_, rr, b = draw.textbbox((0, 0), tag, font=f)
        tx, ty = min(xs), max(0, min(ys) - (b - t_) - 4)
        draw.rectangle([tx, ty, tx + rr - l + 6, ty + b - t_ + 4], fill=col)
        draw.text((tx + 3, ty + 1 - t_), tag, fill="white", font=f)

    img.save(out_dir / "images" / f"{stem}.jpg", quality=92)
    page = dict(
        task_id=task_id, annotation_id=ann_id, image=stem + ".jpg", width=W, height=H,
        created_at=str(created), updated_at=str(updated),
        region_count=len(regions), ocr_count=sum(1 for r in regions if r["ocr_text"].strip()),
        regions=regions,
    )
    (out_dir / "json" / f"{stem}.json").write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
    pages.append(page)
    print(f"{stem}: {page['region_count']} regions, {page['ocr_count']} with OCR", flush=True)

(out_dir / "all_annotations.json").write_text(
    json.dumps(dict(exported_on=datetime.date.today().isoformat(), page_count=len(pages),
                    region_count=sum(p["region_count"] for p in pages),
                    ocr_count=sum(p["ocr_count"] for p in pages), pages=pages), ensure_ascii=False, indent=2),
    encoding="utf-8")
print(f"\nDone: {len(pages)} pages -> {out_dir}")
