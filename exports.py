"""Export formats for detected layouts: COCO JSON, PAGE-XML and HTML.

A *page* is a dict: {"name": str, "width": int, "height": int, "dets": [...], "image": PIL.Image | None}.
Each detection is {"box": (x1, y1, x2, y2), "conf": float, "cls": int, "label": str, optional "source_label", "text"}.
The 42 original class names are always exported (`source_label`), even when the app shows the simplified classes.
"""
from __future__ import annotations

import base64
import datetime
import io
import json
from html import escape
from xml.sax.saxutils import quoteattr

from layout_order import reading_order

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
CREATOR = "IndicDLP Live Layout Studio"


# Pictures and advertisements: left out of exports when the user chooses so (sidebar: "Leave out images and ads").
MEDIA_CLASSES = {"figure", "advertisement", "image"}


def is_media(d: dict) -> bool:
    return d.get("source_label", d["label"]) in MEDIA_CLASSES or d["label"] in MEDIA_CLASSES


def without_media(dets: list[dict]) -> list[dict]:
    return [d for d in dets if not is_media(d)]


def _src(d: dict) -> str:
    return d.get("source_label", d["label"])


def _ordered(page: dict) -> list[dict]:
    return reading_order(page["dets"], (page["width"], page["height"]))


# ----------------------------------------------------------------------------
# COCO
# ----------------------------------------------------------------------------
def coco_json(pages: list[dict], class_names: list[str]) -> str:
    """COCO detection format. Category ids are 1..N in the order of `class_names`; bbox is [x, y, width, height] in pixels."""
    cat_id = {n: i + 1 for i, n in enumerate(class_names)}
    images, annotations = [], []
    for image_id, page in enumerate(pages, 1):
        images.append({"id": image_id, "file_name": page["name"], "width": page["width"], "height": page["height"]})
        for order, d in enumerate(_ordered(page), 1):
            x1, y1, x2, y2 = (float(v) for v in d["box"])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            ann = {"id": len(annotations) + 1, "image_id": image_id, "category_id": cat_id[_src(d)],
                   "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)], "area": round(w * h, 2),
                   "iscrowd": 0, "score": round(float(d["conf"]), 4), "reading_order": order}
            if d.get("text"):
                ann["text"] = d["text"]
            if d.get("translation"):
                ann["text_en"] = d["translation"]
            annotations.append(ann)
    return json.dumps({
        "info": {"description": "IndicDLP Live Layout Studio detections", "date_created": datetime.date.today().isoformat()},
        "images": images, "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# PAGE-XML
# ----------------------------------------------------------------------------
_PAGE_TYPE = {
    "headline": "heading", "sub-headline": "heading", "subsub-headline": "heading", "chapter-title": "heading",
    "section-title": "heading", "sub-section-title": "heading", "subsub-section-title": "heading",
    "paragraph": "paragraph", "quote": "paragraph", "figure-caption": "caption", "table-caption": "caption",
    "header": "header", "footer": "footer", "page-number": "page-number", "folio": "page-number",
    "footnote": "footnote", "sidebar": "marginalia", "author": "credit", "dateline": "credit",
}
_PAGE_ELEMENT = {"figure": "ImageRegion", "flag": "GraphicRegion", "table": "TableRegion", "formula": "MathsRegion",
                 "advertisement": "AdvertRegion"}


def page_xml(page: dict) -> str:
    """One page as PAGE-XML (2019 schema): regions as polygons (rectangles), reading order and OCR text."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    ordered = _ordered(page)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<PcGts xmlns="{PAGE_NS}">',
             f"  <Metadata><Creator>{escape(CREATOR)}</Creator><Created>{now}</Created><LastChange>{now}</LastChange></Metadata>",
             f"  <Page imageFilename={quoteattr(page['name'])} imageWidth=\"{page['width']}\" imageHeight=\"{page['height']}\">",
             '    <ReadingOrder><OrderedGroup id="ro">']
    for i in range(len(ordered)):
        lines.append(f'      <RegionRefIndexed index="{i}" regionRef="r{i + 1}"/>')
    lines.append("    </OrderedGroup></ReadingOrder>")
    for i, d in enumerate(ordered, 1):
        x1, y1, x2, y2 = (round(float(v)) for v in d["box"])
        pts = f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"
        src = _src(d)
        element = _PAGE_ELEMENT.get(src, "TextRegion")
        attrs = f'id="r{i}" custom={quoteattr("class:" + src)}'
        if element == "TextRegion":
            attrs += f' type="{_PAGE_TYPE.get(src, "other")}"'
        lines.append(f"    <{element} {attrs}>")
        lines.append(f'      <Coords points="{pts}"/>')
        if d.get("text"):
            lines.append(f'      <TextEquiv index="0" conf="{float(d["conf"]):.4f}"><Unicode>{escape(d["text"])}</Unicode></TextEquiv>')
            if d.get("translation"):  # the English translation is a second TextEquiv of the same region
                lines.append(f'      <TextEquiv index="1" comments="English translation"><Unicode>{escape(d["translation"])}</Unicode></TextEquiv>')
        lines.append(f"    </{element}>")
    lines += ["  </Page>", "</PcGts>"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
_H1 = {"headline"}
_H2 = {"sub-headline", "chapter-title"}
_H3 = {"subsub-headline", "section-title", "sub-section-title", "subsub-section-title"}
_IMAGE = {"figure", "flag", "table", "formula", "advertisement"}
_FURNITURE = {"header", "footer", "page-number", "folio", "jumpline", "dateline"}


def _crop_uri(image, box) -> str | None:
    if image is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    buf = io.BytesIO()
    image.crop((x1, y1, x2, y2)).convert("RGB").save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _html_block(d: dict, image) -> str:
    src = _src(d)
    text = escape(d.get("text") or "").replace("\n", "<br>")
    if text and d.get("translation"):  # English translation right below the text of the region
        text += f'<span class="en" lang="en">{escape(d["translation"]).replace(chr(10), "<br>")}</span>'
    box = ",".join(str(round(float(v))) for v in d["box"])
    meta = f'data-class="{escape(src)}" data-conf="{float(d["conf"]):.3f}" data-bbox="{box}"'
    if src in _IMAGE:
        uri = _crop_uri(image, d["box"])
        body = f'<img src="{uri}" alt="{escape(src)}">' if uri else f'<span class="ph">[{escape(src)}]</span>'
        return f'<figure class="r image" {meta}>{body}</figure>'
    if not text:
        return ""
    if src in _H1:
        return f'<h1 class="r" {meta}>{text}</h1>'
    if src in _H2:
        return f'<h2 class="r" {meta}>{text}</h2>'
    if src in _H3:
        return f'<h3 class="r" {meta}>{text}</h3>'
    if src in _FURNITURE:
        return f'<div class="r furniture" {meta}>{text}</div>'
    if src in {"figure-caption", "table-caption"}:
        return f'<p class="r caption" {meta}>{text}</p>'
    return f'<p class="r" {meta}>{text}</p>'


def html_document(pages: list[dict], title: str = "Layout export") -> str:
    """A self-contained HTML file: every page's regions in reading order, with figures cropped from the page."""
    sections = []
    for n, page in enumerate(pages, 1):
        blocks = "\n".join(b for d in _ordered(page) if (b := _html_block(d, page.get("image"))))
        sections.append(f'<section class="page" id="page-{n}"><h6 class="pn">{escape(page["name"])} '
                        f'({page["width"]}×{page["height"]} px)</h6>\n{blocks}\n</section>')
    return f"""<!doctype html>
<html lang="bn"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="generator" content="{escape(CREATOR)}"><title>{escape(title)}</title>
<style>
body{{font:17px/1.65 "Noto Sans Bengali","Nirmala UI","Segoe UI",system-ui,sans-serif;background:#f4f4f6;color:#1b1b1f;margin:0}}
.page{{max-width:820px;margin:24px auto;padding:28px 36px;background:#fff;border-radius:10px;box-shadow:0 2px 12px #0002}}
.pn{{margin:0 0 18px;color:#777;font-weight:500;font-size:.8rem;text-transform:none}}
h1{{font-size:1.9rem;line-height:1.25;margin:.6em 0 .3em}} h2{{font-size:1.4rem;margin:.8em 0 .3em}} h3{{font-size:1.15rem;margin:.8em 0 .3em}}
p{{margin:.5em 0}} .caption{{font-size:.88rem;color:#555;font-style:italic}}
.furniture{{font-size:.75rem;color:#888;margin:.3em 0}} figure{{margin:.9em 0;text-align:center}}
figure img{{max-width:100%;border:1px solid #ddd;border-radius:4px}} .ph{{color:#999}}
.en{{display:block;margin-top:.25em;font:italic 400 .82em/1.5 "Segoe UI",system-ui,sans-serif;color:#2f5d8a}}
h1 .en,h2 .en,h3 .en{{font-size:.5em}}
</style></head><body>
{chr(10).join(sections)}
</body></html>"""
