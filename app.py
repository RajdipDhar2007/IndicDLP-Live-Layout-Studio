"""IndicDLP Live Layout Studio.

Upload a document page, watch a scan sweep over it, then see every IndicDLP model detect
layout regions one box at a time, live in the browser. Region names can be shown in English
or any of 11 Indic languages, and models can be fused for higher recall.

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import base64
import collections
import io
import json
import time
import urllib.request
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from i18n import LANGS, label as tr

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
EXAMPLE_DIRS = [ROOT / "data" / "examples", ROOT / "fig" / "images"]
ZENODO = "https://zenodo.org/records/15881917/files/{}?download=1"
DEFAULTS = dict(conf=0.25, iou=0.70, imgsz=1024, speed="Normal", boost=True, fuse=True, lang="en")

st.set_page_config(page_title="IndicDLP Live Layout Studio", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    blurb: str
    file: str
    kind: str  # "yolo" | "doclayout" | "fusion"
    map: str
    icon: str
    size: int  # expected file size in bytes, used to reject partial downloads


MODELS = {
    "yolov10x": ModelSpec("yolov10x", "YOLOv10x", "Best single-model accuracy, pretrained on UED-mini then fine-tuned on IndicDLP.",
                          "md2_PT_indicdlp_FT.pt", "yolo", "57.7", "⚡", 64280114),
    "doclayout": ModelSpec("doclayout", "DocLayout-YOLO", "Document-specialised YOLOv10 variant with global-to-local perception.",
                           "doclayout_yolo_indicdlp.pt", "doclayout", "54.5", "🧭", 161475350),
    "yolo26": ModelSpec("yolo26", "YOLO26", "Latest Ultralytics YOLO (extra-large size), NMS-free, fine-tuned on IndicDLP's 42 layout classes (train it with yolo26/train.py).",
                        "yolo26_indicdlp.pt", "yolo", "43.4", "🚀", 1),
}
FUSION = ModelSpec("fusion", "Ensemble Fusion", "All models voting together: boxes from every model merged with weighted box fusion.",
                   "", "fusion", "-", "🧬", 0)

CLASS_NAMES = [
    "advertisement", "answer", "author", "chapter-title", "contact-info", "dateline", "figure", "figure-caption",
    "first-level-question", "flag", "folio", "footer", "footnote", "formula", "header", "headline", "index", "jumpline",
    "options", "ordered-list", "page-number", "paragraph", "placeholder-text", "quote", "reference",
    "second-level-question", "section-title", "sidebar", "sub-headline", "sub-ordered-list", "sub-section-title",
    "subsub-ordered-list", "subsub-section-title", "sub-unordered-list", "subsub-headline", "subsub-unordered-list",
    "table", "table-caption", "table-of-contents", "third-level-question", "unordered-list", "website-link",
]

# Colour families keep related regions recognisable (greens = headlines, oranges = titles, blues = body text,
# purples = lists, cyans = figures/tables, pinks/reds = page furniture, limes = exam parts).
GROUPS: list[tuple[str, dict[str, str]]] = [
    ("Headlines", {"headline": "#22c55e", "sub-headline": "#4ade80", "subsub-headline": "#86efac"}),
    ("Titles", {"chapter-title": "#f97316", "section-title": "#fb923c", "sub-section-title": "#fbbf24",
                "subsub-section-title": "#fde047"}),
    ("Body text", {"paragraph": "#3b82f6", "quote": "#6366f1", "sidebar": "#818cf8", "author": "#d97706",
                   "reference": "#0284c7", "footnote": "#be185d"}),
    ("Lists", {"ordered-list": "#a855f7", "sub-ordered-list": "#c084fc", "subsub-ordered-list": "#d8b4fe",
               "unordered-list": "#d946ef", "sub-unordered-list": "#e879f9", "subsub-unordered-list": "#f0abfc"}),
    ("Figures & tables", {"figure": "#06b6d4", "figure-caption": "#67e8f9", "table": "#14b8a6", "table-caption": "#5eead4",
                          "formula": "#0ea5e9", "table-of-contents": "#94a3b8", "index": "#cbd5e1"}),
    ("Page furniture", {"header": "#ec4899", "footer": "#f43f5e", "page-number": "#fb7185", "folio": "#db2777",
                        "flag": "#ef4444", "dateline": "#f9a8d4", "jumpline": "#fda4af"}),
    ("Exam parts", {"first-level-question": "#84cc16", "second-level-question": "#a3e635",
                    "third-level-question": "#bef264", "options": "#65a30d", "answer": "#eab308"}),
    ("Other", {"advertisement": "#b45309", "contact-info": "#10b981", "placeholder-text": "#a8a29e",
               "website-link": "#2563eb"}),
]
COLORS = {name: col for _, members in GROUPS for name, col in members.items()}
assert set(COLORS) == set(CLASS_NAMES), set(CLASS_NAMES) ^ set(COLORS)


def color_of(det_or_name) -> str:
    name = det_or_name["label"] if isinstance(det_or_name, dict) else det_or_name
    return COLORS.get(name, "#ffffff")


def rgb(hexcol: str) -> tuple[int, int, int]:
    return int(hexcol[1:3], 16), int(hexcol[3:5], 16), int(hexcol[5:7], 16)


@st.cache_resource(show_spinner=False)
def get_font(size: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def device_name() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def model_status(spec: ModelSpec) -> tuple[bool, str]:
    path = WEIGHTS / spec.file
    if not path.exists():
        return False, "Weights not downloaded."
    if path.stat().st_size < spec.size:
        return False, "Weights incomplete (download still running or interrupted)."
    return True, ""


# ----------------------------------------------------------------------------
# Loading + inference (all return a unified list of detections)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(key: str, device: str):
    spec = MODELS[key]
    path = str(WEIGHTS / spec.file)
    if spec.kind == "yolo":
        from ultralytics import YOLO
        return YOLO(path)
    from doclayout_yolo import YOLOv10
    return YOLOv10(path)


def predict_once(key: str, image: Image.Image, conf: float, iou: float, imgsz: int, device: str) -> list[dict]:
    import numpy as np
    model = load_model(key, device)
    bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])  # ultralytics expects BGR arrays
    res = list(model.predict(bgr, imgsz=imgsz, conf=conf, iou=iou,
                             device=0 if device.startswith("cuda") else "cpu", verbose=False))[0]
    names = res.names
    return [dict(box=(x1, y1, x2, y2), conf=float(c), cls=int(k), label=names[int(k)])
            for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]


def fuse(det_lists: list[list[dict]], size: tuple[int, int], conf: float, iou_thr: float = 0.55) -> list[dict]:
    """Weighted box fusion across several detection lists (scales or models)."""
    from ensemble_boxes import weighted_boxes_fusion
    w, h = size
    boxes, scores, labels = [], [], []
    for dets in det_lists:
        boxes.append([[min(max(v / s, 0.0), 1.0) for v, s in zip(d["box"], (w, h, w, h))] for d in dets] or [[0, 0, 0, 0]])
        scores.append([d["conf"] for d in dets] or [0.0])
        labels.append([d["cls"] for d in dets] or [0])
    fb, fs, fl = weighted_boxes_fusion(boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=conf, conf_type="max")
    out = []
    for (x1, y1, x2, y2), s, k in zip(fb, fs, fl):
        k = int(k)
        out.append(dict(box=(x1 * w, y1 * h, x2 * w, y2 * h), conf=float(s), cls=k, label=CLASS_NAMES[k]))
    return out


def reading_order(dets: list[dict]) -> list[dict]:
    return sorted(dets, key=lambda d: (round(d["box"][1] / 40), d["box"][0]))


def run_model(key: str, image: Image.Image, cfg: dict, device: str) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    scales = [cfg["imgsz"]]
    if cfg["boost"]:
        scales.append(1280 if cfg["imgsz"] < 1280 else 1024)  # a second scale recovers small / faint regions
    passes = [predict_once(key, image, cfg["conf"], cfg["iou"], s, device) for s in scales]
    dets = fuse(passes, image.size, cfg["conf"], 0.6) if len(passes) > 1 else passes[0]
    return reading_order(dets), time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def fit(image: Image.Image, max_w: int = 900) -> tuple[Image.Image, float]:
    s = min(1.0, max_w / image.width)
    return (image.resize((int(image.width * s), int(image.height * s)), Image.Resampling.LANCZOS) if s < 1 else image.copy()), s


def draw_box(canvas: Image.Image, det: dict, scale: float, number: int) -> Image.Image:
    """Colour-coded box with a numbered badge (ASCII only: Indic text is rendered in the browser instead)."""
    base = canvas.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    col = rgb(color_of(det))
    x1, y1, x2, y2 = [v * scale for v in det["box"]]
    d.rectangle([x1, y1, x2, y2], fill=col + (55,), outline=col + (255,), width=2)
    font = get_font(max(10, int(base.width / 85)))
    text = str(number)
    tw = d.textlength(text, font=font)
    d.rectangle([x1, y1, x1 + tw + 8, y1 + font.size + 4], fill=col + (240,))  # type: ignore[attr-defined]
    d.text((x1 + 4, y1 + 1), text, fill=(10, 10, 20, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def annotate_png(image: Image.Image, dets: list[dict]) -> Image.Image:
    """Full-resolution export with English labels (paragraph labels are dropped on busy pages)."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = get_font(max(12, int(base.width / 75)))
    dense = len(dets) > 30
    for det in dets:
        col = rgb(color_of(det))
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], fill=col + (50,), outline=col + (255,), width=max(2, base.width // 500))
        if dense and det["label"] == "paragraph":
            continue
        text = f'{det["label"]} {det["conf"]:.2f}'
        tw = d.textlength(text, font=font)
        th = font.size + 6  # type: ignore[attr-defined]
        ty = y1 - th if y1 - th > 0 else y1
        d.rectangle([x1, ty, x1 + tw + 10, ty + th], fill=col + (235,))
        d.text((x1 + 5, ty + 2), text, fill=(10, 10, 20, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def b64_jpeg(image: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def scan_html(image: Image.Image, caption: str) -> str:
    return f"""
    <div class="scan-wrap">
      <img src="data:image/jpeg;base64,{b64_jpeg(image)}" class="scan-img"/>
      <div class="scan-grid"></div>
      <div class="scan-beam"></div>
      <div class="scan-corner tl"></div><div class="scan-corner tr"></div>
      <div class="scan-corner bl"></div><div class="scan-corner br"></div>
      <div class="scan-label">{escape(caption)}</div>
    </div>"""


def overlay_doc(image: Image.Image, dets: list[dict], lang: str) -> str:
    """Self-contained HTML page (shown in an iframe): the document with colour-coded, localised region labels.

    Labels are real HTML text, so the browser shapes Indic scripts correctly and sizes each label to its text.
    """
    w, h = image.size
    dense = len(dets) > 30  # on busy pages paragraph labels are shown on hover only
    boxes = []
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = d["box"]
        col = color_of(d)
        name = escape(tr(lang, d["label"]))
        quiet = " q" if dense and d["label"] == "paragraph" else ""
        edge = " in" if y1 / h < 0.025 else ""
        boxes.append(
            f'<div class="b{quiet}" style="left:{x1 / w * 100:.3f}%;top:{y1 / h * 100:.3f}%;width:{(x2 - x1) / w * 100:.3f}%;'
            f'height:{(y2 - y1) / h * 100:.3f}%;--c:{col};animation-delay:{min(i * 10, 1200)}ms" title="#{i} {name} · {d["conf"] * 100:.0f}%">'
            f'<span class="l{edge}">{name}</span></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent}}
.wrap{{position:relative;line-height:0;container-type:inline-size;border-radius:14px;overflow:hidden;
 border:1px solid #2a2f5a;box-shadow:0 0 30px rgba(124,58,237,.25)}}
.wrap img{{width:100%;display:block}}
.b{{position:absolute;box-sizing:border-box;border:2px solid var(--c);background:color-mix(in srgb,var(--c) 16%,transparent);
 animation:in .5s ease both}}
.b:hover{{background:color-mix(in srgb,var(--c) 38%,transparent);z-index:5}}
.l{{position:absolute;left:-2px;bottom:100%;white-space:nowrap;background:var(--c);color:#0a0b1a;font-weight:700;
 font-size:clamp(8px,1.3cqw,17px);line-height:1.35;padding:0 .4em;border-radius:3px 3px 0 0;
 font-family:'Nirmala UI','Noto Sans','Noto Sans Devanagari','Noto Sans Bengali','Noto Sans Tamil','Noto Sans Telugu',
 'Noto Sans Kannada','Noto Sans Malayalam','Noto Sans Gujarati','Noto Sans Gurmukhi','Noto Sans Oriya','Segoe UI',sans-serif}}
.l.in{{bottom:auto;top:0;border-radius:0 0 3px 0}}
.b.q .l{{display:none}}.b.q:hover .l{{display:block}}
@keyframes in{{from{{opacity:0;transform:scale(.97)}}to{{opacity:1;transform:none}}}}
</style></head><body><div class="wrap"><img src="data:image/jpeg;base64,{b64_jpeg(image, 85)}">{"".join(boxes)}</div></body></html>"""

def feed_html(dets: list[dict], shown: int, lang: str) -> str:
    rows = []
    for i, d in list(enumerate(dets[:shown], 1))[-9:][::-1]:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        rows.append(
            f'<div class="feed-row" style="--c:{color_of(d)}"><span class="dot"></span>'
            f'<b>{escape(tr(lang, d["label"]))}</b><span class="conf">{d["conf"] * 100:.0f}%</span>'
            f'<span class="xy">#{i} · [{x1},{y1},{x2},{y2}]</span></div>')
    return '<div class="feed">' + "".join(rows) + "</div>"


def stats_html(shown: int, total: int, dets: list[dict], elapsed: float, device: str) -> str:
    seen = dets[:shown]
    avg = sum(d["conf"] for d in seen) / len(seen) * 100 if seen else 0
    kinds = len({d["label"] for d in seen})
    tiles = [("Regions", f"{shown}/{total}"), ("Classes", str(kinds)), ("Avg conf", f"{avg:.0f}%"),
             ("Inference", f"{elapsed * 1000:.0f} ms"), ("Device", device.upper())]
    return '<div class="tiles">' + "".join(f'<div class="tile"><span>{k}</span><b>{v}</b></div>' for k, v in tiles) + "</div>"


def chips_html(dets: list[dict], shown: int, lang: str) -> str:
    counts = collections.Counter(d["label"] for d in dets[:shown])
    return '<div class="chips">' + "".join(
        f'<span class="chip" style="--c:{color_of(lbl)}">{escape(tr(lang, lbl))} <i>{n}</i></span>'
        for lbl, n in counts.most_common()) + "</div>"


def legend_html(lang: str, present: set[str] | None = None) -> str:
    out = []
    for group, members in GROUPS:
        names = [n for n in members if present is None or n in present]
        if not names:
            continue
        chips = "".join(f'<span class="chip" style="--c:{members[n]}">{escape(tr(lang, n))}</span>' for n in names)
        out.append(f'<div class="legend-group"><span class="legend-title">{group}</span>{chips}</div>')
    return '<div class="legend">' + "".join(out) + "</div>"


def progress_html(p: float, text: str) -> str:
    return (f'<div class="pbar"><div class="pfill" style="width:{p * 100:.0f}%"></div>'
            f'<span>{escape(text)}</span></div>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');
:root{--indic:'Nirmala UI','Noto Sans','Noto Sans Devanagari','Noto Sans Bengali','Noto Sans Tamil','Noto Sans Telugu',
 'Noto Sans Kannada','Noto Sans Malayalam','Noto Sans Gujarati','Noto Sans Gurmukhi','Noto Sans Oriya','Segoe UI',sans-serif}
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#1b1f4a 0%,transparent 60%),
 radial-gradient(900px 500px at 100% 0%,#3a1450 0%,transparent 55%),#0a0b1a;}
.hero{padding:1.2rem 0 .4rem;animation:rise .8s ease both}
.hero h1{font-family:'Space Grotesk',sans-serif;font-size:3rem;margin:0;line-height:1.1;
 background:linear-gradient(90deg,#7cf3ff,#a78bfa,#ff7ad9,#7cf3ff);background-size:300% 100%;
 -webkit-background-clip:text;background-clip:text;color:transparent;animation:flow 6s linear infinite}
.hero p{color:#9aa0c3;margin:.4rem 0 0;font-size:1.05rem}
@keyframes flow{to{background-position:300% 0}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:translateX(-12px) scale(.97)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(124,243,255,.45)}50%{box-shadow:0 0 0 10px rgba(124,243,255,0)}}
@keyframes reveal{from{opacity:0}to{opacity:1}}
.model-card{border:1px solid #2a2f5a;border-radius:16px;padding:.9rem 1.1rem;margin:.2rem 0 .6rem;
 background:linear-gradient(135deg,rgba(30,34,80,.7),rgba(20,22,50,.7));animation:rise .6s ease both}
.model-card h3{margin:0;font-family:'Space Grotesk',sans-serif;color:#eef;display:flex;align-items:center;gap:.5rem}
.model-card .badge{margin-left:auto;font-size:.8rem;padding:.15rem .6rem;border-radius:99px;
 background:#16315a;color:#7cf3ff;border:1px solid #2b5c9a}
.model-card.live{animation:pulse 1.6s infinite}
.model-card p{margin:.3rem 0 0;color:#9aa0c3;font-size:.9rem}
.scan-wrap{position:relative;max-width:900px;margin:auto;border-radius:14px;overflow:hidden;
 border:1px solid #3b82f6;box-shadow:0 0 40px rgba(59,130,246,.35)}
.scan-img{display:block;width:100%;filter:brightness(.8) contrast(1.05)}
.scan-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(124,243,255,.12) 1px,transparent 1px),
 linear-gradient(90deg,rgba(124,243,255,.12) 1px,transparent 1px);background-size:34px 34px;animation:gridfade 1.2s ease-in-out infinite alternate}
@keyframes gridfade{from{opacity:.25}to{opacity:.9}}
.scan-beam{position:absolute;left:0;right:0;height:90px;top:-90px;
 background:linear-gradient(180deg,transparent,rgba(124,243,255,.55) 85%,#e8fdff 100%);
 border-bottom:2px solid #e8fdff;box-shadow:0 6px 30px 6px rgba(124,243,255,.7);animation:sweep 1.5s ease-in-out infinite}
@keyframes sweep{0%{top:-90px}100%{top:100%}}
.scan-corner{position:absolute;width:30px;height:30px;border:3px solid #7cf3ff}
.tl{top:10px;left:10px;border-right:0;border-bottom:0}.tr{top:10px;right:10px;border-left:0;border-bottom:0}
.bl{bottom:10px;left:10px;border-right:0;border-top:0}.br{bottom:10px;right:10px;border-left:0;border-top:0}
.scan-label{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);padding:.25rem .9rem;border-radius:99px;
 background:rgba(8,10,30,.8);color:#7cf3ff;font-family:'Space Grotesk',sans-serif;letter-spacing:.12em;font-size:.8rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:.5rem;margin-bottom:.6rem}
.tile{background:#12153a;border:1px solid #262b5e;border-radius:12px;padding:.5rem .7rem;animation:rise .4s ease both}
.tile span{display:block;color:#8a90b8;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em}
.tile b{color:#fff;font-size:1.2rem;font-family:'Space Grotesk',sans-serif}
.feed{display:flex;flex-direction:column;gap:.35rem;min-height:300px;font-family:var(--indic)}
.feed-row{display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;border-radius:10px;border-left:4px solid var(--c);
 background:rgba(255,255,255,.04);animation:pop .35s ease both;color:#e8eaff;font-size:.9rem}
.feed-row .dot{width:9px;height:9px;border-radius:50%;background:var(--c);box-shadow:0 0 10px var(--c)}
.feed-row .conf{margin-left:auto;color:var(--c);font-weight:700}
.feed-row .xy{color:#7d83ab;font-size:.72rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin:.5rem 0;font-family:var(--indic)}
.chip{padding:.15rem .6rem;border-radius:99px;border:1px solid var(--c);color:var(--c);font-size:.8rem;animation:pop .3s ease both}
.chip i{font-style:normal;background:var(--c);color:#0a0b1a;border-radius:99px;padding:0 .4rem;margin-left:.2rem;font-weight:700}
.legend{display:flex;flex-direction:column;gap:.4rem;margin:.6rem 0;font-family:var(--indic)}
.legend-group{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center}
.legend-title{min-width:120px;color:#8a90b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;font-family:'Space Grotesk',sans-serif}
.pbar{position:relative;height:26px;border-radius:99px;background:#12153a;border:1px solid #262b5e;overflow:hidden;margin:.3rem 0 .6rem}
.pfill{height:100%;background:linear-gradient(90deg,#3b82f6,#a78bfa,#ff7ad9);transition:width .2s;
 background-size:100% 100%}
.pbar span{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:.78rem;color:#fff;font-family:var(--indic)}
</style>
"""


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def list_examples() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in EXAMPLE_DIRS:
        if d.exists():
            for p in sorted([*d.glob("*.png"), *d.glob("*.jpg")]):
                out[p.stem] = p
    return out


def download_weights(spec: ModelSpec) -> None:
    WEIGHTS.mkdir(exist_ok=True)
    dest = WEIGHTS / spec.file
    tmp = dest.with_suffix(dest.suffix + ".part")
    bar = st.progress(0.0, text=f"Downloading {spec.file}…")
    with urllib.request.urlopen(ZENODO.format(spec.file)) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or 1
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            bar.progress(min(done / total, 1.0), text=f"Downloading {spec.file} · {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    tmp.replace(dest)
    bar.empty()


def reset_defaults() -> None:
    st.session_state.update(DEFAULTS)
    for k in ("result", "sig"):
        st.session_state.pop(k, None)


def sidebar() -> dict:
    with st.sidebar:
        st.markdown("### 🎛️ Control room")
        just_downloaded = st.session_state.pop("_enable_after_download", None)
        if just_downloaded:
            st.session_state[f"use_{just_downloaded}"] = True
        chosen: list[str] = []
        for key, spec in MODELS.items():
            ok, why = model_status(spec)
            label = f"{spec.icon} {spec.title}" + (f"  ·  mAP {spec.map}" if spec.map != "-" else "")
            on = st.toggle(label, value=ok, key=f"use_{key}", help=why or spec.blurb)
            if on and ok:
                chosen.append(key)
            if not ok and key == "yolo26":
                st.caption(f"⚠️ {why} Train it with `yolo26/train.py`.")
            elif not ok:
                if st.button(f"⬇ Download {spec.title} weights", key=f"dl_{key}"):
                    download_weights(spec)
                    st.session_state["_enable_after_download"] = key  # applied before the toggle is drawn on the rerun
                    st.rerun()
        st.divider()
        st.markdown("##### 🌐 Region names")
        st.selectbox("Language", list(LANGS), format_func=lambda k: LANGS[k], key="lang",
                     help="Class names in the feed, chips, legend, tables and the final annotated page.")
        st.divider()
        st.markdown("##### 🎯 Accuracy")
        boost = st.toggle("Multi-scale boost", key="boost",
                          help="Runs every model at two image sizes and fuses the boxes, recovering small or faint regions.")
        fuse_on = st.toggle("Ensemble fusion", key="fuse",
                            help="Merges the boxes of all enabled models into one more complete result.")
        conf = st.slider("Confidence threshold", 0.05, 0.95, step=0.05, key="conf")
        iou = st.slider("NMS IoU", 0.1, 0.95, step=0.05, key="iou")
        imgsz = st.select_slider("Inference size", [640, 800, 1024, 1280], key="imgsz")
        speed = st.select_slider("Reveal speed", ["Cinematic", "Normal", "Fast", "Instant"], key="speed")
        st.divider()
        st.caption(f"Device: **{device_name().upper()}**")
        st.button("🔄 Reset to defaults", width="stretch", on_click=reset_defaults)
    return dict(models=chosen, conf=conf, iou=iou, imgsz=imgsz, speed=speed, boost=boost, fuse=fuse_on)


def step_delay(speed: str, n: int) -> float:
    total = {"Cinematic": 6.0, "Normal": 3.0, "Fast": 1.2, "Instant": 0.0}[speed]
    return min(total / max(n, 1), 0.6)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def reveal(spec: ModelSpec, compute, shown_img: Image.Image, scale: float, cfg: dict, lang: str, device: str) -> dict | None:
    """One model's animated panel: show scan, run `compute`, then draw its boxes one by one."""
    card = st.empty()
    card.html(f'<div class="model-card live"><h3>{spec.icon} {spec.title}<span class="badge">loading…</span></h3>'
              f'<p>{spec.blurb}</p></div>')
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        canvas_ph = st.empty()
        canvas_ph.html(scan_html(shown_img, f"{spec.title.upper()} · ANALYSING"))
    with right:
        stats_ph, prog_ph, chips_ph, feed_ph = st.empty(), st.empty(), st.empty(), st.empty()

    try:
        dets, elapsed = compute()
    except Exception as exc:  # keep the app alive if one model fails
        canvas_ph.empty()
        card.html(f'<div class="model-card"><h3>{spec.icon} {spec.title}<span class="badge">failed</span></h3></div>')
        right.error(f"{spec.title} failed: {exc}")
        return None

    card.html(f'<div class="model-card live"><h3>{spec.icon} {spec.title}<span class="badge">detecting</span></h3>'
              f'<p>{spec.blurb}</p></div>')
    canvas = shown_img.copy()
    delay = step_delay(cfg["speed"], len(dets))
    canvas_ph.image(canvas, width="stretch", output_format="JPEG")
    for i, det in enumerate(dets, 1):
        canvas = draw_box(canvas, det, scale, i)
        canvas_ph.image(canvas, width="stretch", output_format="JPEG")
        stats_ph.html(stats_html(i, len(dets), dets, elapsed, device))
        prog_ph.html(progress_html(i / len(dets), f"Region {i} of {len(dets)}: {tr(lang, det['label'])}"))
        chips_ph.html(chips_html(dets, i, lang))
        feed_ph.html(feed_html(dets, i, lang))
        if delay:
            time.sleep(delay)
    if not dets:
        stats_ph.html(stats_html(0, 0, dets, elapsed, device))
        right.info("No regions above the confidence threshold. Try lowering it.")
    else:
        prog_ph.html(progress_html(1.0, "Complete ✓"))
    card.html(f'<div class="model-card"><h3>{spec.icon} {spec.title}<span class="badge">done · {len(dets)} regions</span></h3>'
              f'<p>{spec.blurb}</p></div>')
    st.divider()
    return dict(dets=dets, elapsed=elapsed)


def analyse(image: Image.Image, cfg: dict, lang: str) -> None:
    device = device_name()
    shown_img, scale = fit(image)

    stage = st.empty()
    stage.html(scan_html(shown_img, "SCANNING DOCUMENT"))
    time.sleep(1.8)
    stage.empty()

    results: dict[str, dict] = {}
    for key in cfg["models"]:
        r = reveal(MODELS[key], lambda k=key: run_model(k, image, cfg, device), shown_img, scale, cfg, lang, device)
        if r:
            results[key] = r

    if cfg["fuse"] and len(results) >= 2:
        def fused():
            t0 = time.perf_counter()
            dets = fuse([r["dets"] for r in results.values()], image.size, cfg["conf"], 0.55)
            return reading_order(dets), sum(r["elapsed"] for r in results.values()) + time.perf_counter() - t0
        r = reveal(FUSION, fused, shown_img, scale, cfg, lang, device)
        if r:
            results["fusion"] = r

    st.session_state["result"] = results
    st.session_state["shown_img"] = image  # full-resolution page for the final SVG / PNG


def title_of(key: str) -> str:
    return FUSION.title if key == "fusion" else MODELS[key].title


def summary(image_name: str, lang: str) -> None:
    results = st.session_state.get("result")
    page = st.session_state.get("shown_img")
    if not results or page is None:
        return
    st.markdown("## 📊 Results")
    rows = []
    for key, r in results.items():
        d = r["dets"]
        rows.append({"Model": title_of(key), "Regions": len(d), "Classes": len({x["label"] for x in d}),
                     "Avg confidence": round(sum(x["conf"] for x in d) / len(d), 3) if d else 0.0,
                     "Time (ms)": round(r["elapsed"] * 1000)})
    st.dataframe(rows, width="stretch", hide_index=True)

    counts: dict[str, dict[str, int]] = {}
    for key, r in results.items():
        for lbl, n in collections.Counter(x["label"] for x in r["dets"]).items():
            counts.setdefault(tr(lang, lbl), {})[title_of(key)] = n
    if counts:
        st.caption("Regions per class")
        st.bar_chart(counts, horizontal=True, height=max(240, 28 * len(counts)))

    keys = list(results)
    tabs = st.tabs([f"{FUSION.icon if k == 'fusion' else MODELS[k].icon} {title_of(k)}" for k in keys])
    for tab, key in zip(tabs, keys):
        r = results[key]
        with tab:
            st.iframe(overlay_doc(page, r["dets"], lang), height="content")
            st.html(legend_html(lang, {d["label"] for d in r["dets"]}))
            buf = io.BytesIO()
            annotate_png(page, r["dets"]).save(buf, "PNG")
            c1, c2 = st.columns(2)
            c1.download_button("⬇ Annotated image (PNG)", buf.getvalue(), f"{image_name}_{key}.png", "image/png", key=f"dp_{key}")
            payload = json.dumps([{**d, "name_local": tr(lang, d["label"]), "box": [round(v, 1) for v in d["box"]]}
                                  for d in r["dets"]], indent=2, ensure_ascii=False)
            c2.download_button("⬇ Detections (JSON)", payload, f"{image_name}_{key}.json", "application/json", key=f"dj_{key}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.html(CSS)
    st.html('<div class="hero"><h1>IndicDLP Live Layout Studio</h1>'
            '<p>Drop in a document page and watch YOLOv10x, DocLayout-YOLO and YOLO26x read its layout, region by region, in 12 languages.</p></div>')
    cfg = sidebar()
    lang = st.session_state["lang"]

    examples = list_examples()
    src = st.radio("Source", ["Upload", "Sample gallery"], horizontal=True, label_visibility="collapsed")
    image: Image.Image | None = None
    name = "document"
    if src == "Upload":
        up = st.file_uploader("Upload a document page", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if up is not None:
            try:
                image = Image.open(up).convert("RGB")
                name = Path(up.name).stem
            except Exception:
                st.error("That file couldn't be read as an image.")
    else:
        if examples:
            pick = st.selectbox("Sample page", list(examples), index=0)
            image = Image.open(examples[pick]).convert("RGB")
            name = pick
        else:
            st.warning("No sample images found.")

    if image is None:
        st.info("⬆️ Upload a page (or pick a sample) to begin.")
        return
    if not cfg["models"]:
        st.warning("Enable at least one model in the sidebar.")
        return

    preview, _ = fit(image, 700)
    c1, c2 = st.columns([1, 1.2], gap="large")
    c1.image(preview, caption=f"{name} · {image.width}×{image.height}px", width="stretch")
    with c2:
        st.markdown("#### Ready to analyse")
        st.write(f"**Models:** {', '.join(MODELS[k].title for k in cfg['models'])}"
                 + (" + Ensemble Fusion" if cfg["fuse"] and len(cfg["models"]) > 1 else ""))
        st.write(f"**Confidence ≥** {cfg['conf']:.2f} · **IoU** {cfg['iou']:.2f} · **Size** {cfg['imgsz']}px"
                 + (" · **multi-scale**" if cfg["boost"] else ""))
        go = st.button("🚀 Scan & detect", type="primary", width="stretch")

    sig = (name, image.size, tuple(cfg["models"]), cfg["conf"], cfg["iou"], cfg["imgsz"], cfg["boost"], cfg["fuse"])
    if go:
        st.session_state["result"] = None
        st.session_state["sig"] = sig
        st.divider()
        analyse(image, cfg, lang)
        st.balloons()
    if st.session_state.get("sig") == sig:
        summary(name, lang)


main()
