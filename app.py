"""IndicDLP Live Layout Studio.

Upload a document page, watch a scan sweep over it, then see every IndicDLP model detect
layout regions one box at a time, live in the browser. Region names can be shown in English
or any of 11 Indic languages, and models can be fused for higher recall.

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

from i18n import LANGS, label as _tr
from news import news_html

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; environment variables still work
    load_dotenv = None

ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(ROOT / ".env")
WEIGHTS = ROOT / "weights"
ESPCN_MODEL = ROOT / "ESPCN_x2.pb"
EXAMPLE_DIRS = [ROOT / "data" / "examples", ROOT / "fig" / "images"]
annotation_view = components.declare_component("indicdlp_annotation_view", path=str(ROOT / "annotation_view"))
ZENODO = "https://zenodo.org/records/15881917/files/{}?download=1"
DEFAULTS = dict(ocr_on=True, ocr_engine="gemini", conf=0.25, iou=0.70, imgsz=1024, speed="Normal", boost=True, fuse=True, lang="en",
                gemini_model="Gemini 3.1 Flash-Lite (faster)", simplify=False)

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

# Optional simplified view: the 42 classes are folded into 6 families (sidebar toggle "Simplified classes").
SIMPLE_GROUPS: list[tuple[str, dict[str, str]]] = [
    ("Headlines", {"headline": "#22c55e"}),
    ("Subtitles", {"subtitle": "#f97316"}),
    ("Paragraphs (Body / Text)", {"paragraph": "#3b82f6"}),
    ("Images", {"image": "#06b6d4"}),
    ("Captions", {"caption": "#67e8f9"}),
    ("Advertisements", {"advertisement": "#b45309"}),
]
COLORS.update({name: col for _, members in SIMPLE_GROUPS for name, col in members.items()})
SUBTITLE_CLASSES = {"sub-headline", "subsub-headline", "chapter-title", "section-title",
                    "sub-section-title", "subsub-section-title"}
IMAGE_CLASSES = {"figure", "table", "formula", "flag"}
CAPTION_CLASSES = {"figure-caption", "table-caption"}


def normalized_label(source_label: str) -> str:
    if source_label == "headline":
        return "headline"
    if source_label in SUBTITLE_CLASSES:
        return "subtitle"
    if source_label in IMAGE_CLASSES:
        return "image"
    if source_label in CAPTION_CLASSES:
        return "caption"
    if source_label == "advertisement":
        return "advertisement"
    return "paragraph"


def relabel(dets: list[dict], simplify: bool) -> list[dict]:
    """Keep the model's own class in `source_label`; `label` is the class shown to the user."""
    for d in dets:
        d["source_label"] = CLASS_NAMES[d["cls"]]
        d["label"] = normalized_label(d["source_label"]) if simplify else d["source_label"]
    return dets


def tr(lang: str, label: str) -> str:
    """Localised class name (also knows the simplified classes)."""
    return _tr(lang, {"subtitle": "section-title", "image": "figure", "caption": "figure-caption"}.get(label, label))


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


# Regions that hold no running text; OCR skips them.
NON_TEXT = {"figure", "table", "formula", "advertisement", "flag", "placeholder-text", "image"}


@st.cache_resource(show_spinner=False)
def load_ocr(device: str):
    import easyocr
    return easyocr.Reader(["bn", "en"], gpu=device != "cpu", verbose=False)  # Bengali + English


GEMINI_MODELS = [m for m in dict.fromkeys([os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"), "gemini-3.5-flash",
                                           "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]) if m]
GEMINI_PROMPT = ("The image is a contact sheet of {n} snippets cut from a document page. Each snippet sits under a yellow label "
                 "'#<number>'. The text may be Bengali, English or both. Transcribe the text of every snippet exactly as written, "
                 "in reading order, keeping line breaks. Do not translate, correct, summarise or add anything. Answer with a JSON "
                 "object that maps each label number (as a string) to its transcribed text; use an empty string when a snippet "
                 "has no readable text.")


def gemini_key() -> str:
    key = st.session_state.get("gemini_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return key.strip()


def contact_sheet(items: list[tuple[int, Image.Image]]) -> Image.Image:
    """Stacks the numbered crops into one tall image so a whole page needs a single Gemini request."""
    width = min(1600, max(c.width for _, c in items))
    font, rows, height = get_font(30), [], 0
    for i, crop in items:
        c = crop.convert("RGB")
        if c.width > width:
            c = c.resize((width, max(1, round(c.height * width / c.width))))
        rows.append((i, c))
        height += c.height + 48
    sheet = Image.new("RGB", (width, height), "white")
    d, y = ImageDraw.Draw(sheet), 0
    for i, c in rows:
        d.rectangle([0, y, 110, y + 40], fill=(255, 214, 0))
        d.text((8, y + 2), f"#{i}", fill=(0, 0, 0), font=font)
        sheet.paste(c, (0, y + 44))
        y += c.height + 48
    return sheet


def gemini_read_sheet(sheet: Image.Image, ids: list[int], key: str) -> dict[int, str]:
    """One request for a whole sheet. Falls back to the next model when one is out of quota or overloaded."""
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    body = json.dumps({"contents": [{"parts": [
        {"text": GEMINI_PROMPT.format(n=len(ids))},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
    last = "no model answered"
    for model in GEMINI_MODELS:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            raw = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            parsed = json.loads(raw)
            return {i: str(parsed.get(str(i), "")).strip() for i in ids}
        except urllib.error.HTTPError as e:
            last = f"{model}: HTTP {e.code}"
            if e.code in (400, 401, 403):
                raise PermissionError(f"Gemini rejected the request (HTTP {e.code}). Check the API key.") from e
        except Exception as e:  # timeout, blocked or malformed answer: try the next model
            last = f"{model}: {type(e).__name__}"
    raise RuntimeError(f"Gemini could not read the page ({last}). Daily free-tier limits may be used up; try EasyOCR.")


def ocr_page(page: Image.Image, dets: list[dict], device: str, progress=None, engine: str = "easyocr") -> None:
    """Reads the text of every text-bearing region and stores it in det["text"] ('' for non-text regions)."""
    import numpy as np
    w, h = page.size
    todo = [d for d in dets if d["label"] not in NON_TEXT]
    for d in dets:
        d["text"] = ""

    def crop_of(d):
        x1, y1, x2, y2 = d["box"]
        pad = 4
        return page.crop((max(0, int(x1) - pad), max(0, int(y1) - pad), min(w, int(x2) + pad), min(h, int(y2) + pad)))

    if engine == "gemini":
        key = gemini_key()
        if not key:
            raise PermissionError("No Gemini API key. Enter it in the sidebar or set GEMINI_API_KEY.")
        items = [(n, d, crop_of(d)) for n, d in enumerate(todo, 1)]
        chunks, cur, tall = [], [], 0
        for it in items:  # keep every sheet reasonably small so Gemini can read it
            if cur and (len(cur) >= 25 or tall + it[2].height > 6000):
                chunks.append(cur)
                cur, tall = [], 0
            cur.append(it)
            tall += it[2].height + 48
        if cur:
            chunks.append(cur)
        for n, chunk in enumerate(chunks, 1):
            texts = gemini_read_sheet(contact_sheet([(i, c) for i, _, c in chunk]), [i for i, _, _ in chunk], key)
            for i, d, _ in chunk:
                d["text"] = texts.get(i, "")
            if progress:
                progress(n / len(chunks), f"Gemini {n}/{len(chunks)}")
        return
    reader = load_ocr(device)
    for n, d in enumerate(todo, 1):
        try:
            lines = reader.readtext(np.array(crop_of(d)), detail=0, paragraph=True)
        except Exception:
            lines = []
        d["text"] = "\n".join(t.strip() for t in lines if t.strip())
        if progress:
            progress(n / len(todo), f"OCR {n}/{len(todo)}")


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
    return relabel(reading_order(dets), cfg.get("simplify", False)), time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def fit(image: Image.Image, max_w: int = 900) -> tuple[Image.Image, float]:
    s = min(1.0, max_w / image.width)
    return (image.resize((int(image.width * s), int(image.height * s)), Image.Resampling.LANCZOS) if s < 1 else image.copy()), s


def draw_box(canvas: Image.Image, det: dict, scale: float, number: int) -> Image.Image:
    """Colour-coded box with a numbered badge (ASCII only: Bengali text is rendered in the browser instead)."""
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
:root{--script:'Nirmala UI','Noto Sans','Noto Sans Bengali','Segoe UI',sans-serif}
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
.feed{display:flex;flex-direction:column;gap:.35rem;min-height:300px;font-family:var(--script)}
.feed-row{display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;border-radius:10px;border-left:4px solid var(--c);
 background:rgba(255,255,255,.04);animation:pop .35s ease both;color:#e8eaff;font-size:.9rem}
.feed-row .dot{width:9px;height:9px;border-radius:50%;background:var(--c);box-shadow:0 0 10px var(--c)}
.feed-row .conf{margin-left:auto;color:var(--c);font-weight:700}
.feed-row .xy{color:#7d83ab;font-size:.72rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin:.5rem 0;font-family:var(--script)}
.chip{padding:.15rem .6rem;border-radius:99px;border:1px solid var(--c);color:var(--c);font-size:.8rem;animation:pop .3s ease both}
.chip i{font-style:normal;background:var(--c);color:#0a0b1a;border-radius:99px;padding:0 .4rem;margin-left:.2rem;font-weight:700}
.legend{display:flex;flex-direction:column;gap:.4rem;margin:.6rem 0;font-family:var(--script)}
.legend-group{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center}
.legend-title{min-width:120px;color:#8a90b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;font-family:'Space Grotesk',sans-serif}
.pbar{position:relative;height:26px;border-radius:99px;background:#12153a;border:1px solid #262b5e;overflow:hidden;margin:.3rem 0 .6rem}
.pfill{height:100%;background:linear-gradient(90deg,#3b82f6,#a78bfa,#ff7ad9);transition:width .2s;
 background-size:100% 100%}
.pbar span{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:.78rem;color:#fff;font-family:var(--script)}
</style>
"""


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------



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
            if not ok:
                if st.button(f"⬇ Download {spec.title} weights", key=f"dl_{key}"):
                    download_weights(spec)
                    st.session_state["_enable_after_download"] = key  # applied before the toggle is drawn on the rerun
                    st.rerun()
        st.divider()
        st.markdown("##### 🌐 Region names")
        st.selectbox("Language", list(LANGS), format_func=lambda k: LANGS[k], key="lang",
                     help="Class names in the feed, chips, legend, tables and the final annotated page.")
        st.toggle("Simplified classes", key="simplify",
                  help="Folds the 42 classes into 6 families: headline, subtitle, paragraph, image, caption, advertisement.")
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
        st.markdown("##### 📝 Text extraction (OCR)")
        st.toggle("Enable OCR", key="ocr_on",
                  help="Off: only the layout annotations are produced, no text is read and no Gemini requests are made.")
        if st.session_state["ocr_on"]:
            st.selectbox("Engine", ["gemini", "easyocr"], key="ocr_engine",
                         format_func=lambda k: {"gemini": "Gemini (cloud, most accurate)", "easyocr": "EasyOCR (local, offline)"}[k])
        if st.session_state["ocr_on"] and st.session_state["ocr_engine"] == "gemini":
            st.text_input("Gemini API key", type="password", key="gemini_key",
                          placeholder="or set GEMINI_API_KEY", help="Kept only in this browser session. Leave empty if GEMINI_API_KEY is set.")
            st.selectbox("Gemini model for per-region OCR", list(GEMINI_OCR_MODELS), key="gemini_model",
                         help="Used when you click a region (or press Transcribe All). Each region is sent only once.")
            st.caption("Gemini sends each text region's image to Google. Use EasyOCR to stay offline.")
        st.divider()
        st.caption(f"Device: **{device_name().upper()}**")
        st.button("🔄 Reset to defaults", width="stretch", on_click=reset_defaults)
    return dict(models=chosen, conf=conf, iou=iou, imgsz=imgsz, speed=speed, boost=boost, fuse=fuse_on,
                simplify=st.session_state["simplify"])


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
            return relabel(reading_order(dets), cfg.get("simplify", False)), sum(r["elapsed"] for r in results.values()) + time.perf_counter() - t0
        r = reveal(FUSION, fused, shown_img, scale, cfg, lang, device)
        if r:
            results["fusion"] = r

    st.session_state["result"] = results
    st.session_state["shown_img"] = image.copy()  # full-resolution page for exports and crop previews
    st.session_state["result_id"] = uuid.uuid4().hex


def title_of(key: str) -> str:
    return FUSION.title if key == "fusion" else MODELS[key].title


# ----------------------------------------------------------------------------
# Crops, ESPCN upscaling and per-region Gemini OCR (click a box to open it)
# ----------------------------------------------------------------------------
def crop_bounds(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
    """Round a valid xyxy box outward, clamp it to the source image, and reject empty boxes."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    left = max(0, min(image.width, math.floor(x1)))
    top = max(0, min(image.height, math.floor(y1)))
    right = max(0, min(image.width, math.ceil(x2)))
    bottom = max(0, min(image.height, math.ceil(y2)))
    return (left, top, right, bottom) if right > left and bottom > top else None


def image_fingerprint(image: Image.Image) -> str:
    """Identify image content so same-name, same-size uploads do not reuse stale results."""
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def load_espcn():
    """Load the ESPCN x2 super-resolution model once for the app session."""
    import cv2
    if not ESPCN_MODEL.is_file():
        raise FileNotFoundError(f"ESPCN model not found: {ESPCN_MODEL.name}")
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError("ESPCN upscaling needs OpenCV contrib (dnn_superres): pip install opencv-contrib-python")
    create_model = getattr(cv2.dnn_superres, "DnnSuperResImpl_create", None)
    model = create_model() if create_model else cv2.dnn_superres.DnnSuperResImpl.create()
    model.readModel(str(ESPCN_MODEL))
    model.setModel("espcn", 2)
    return model


@st.cache_data(show_spinner=False, max_entries=2)
def upscale_espcn(image_bytes: bytes, width: int, height: int) -> Image.Image:
    """Return an RGB x2 ESPCN image, caching the result across ordinary reruns."""
    import cv2
    import numpy as np
    rgb_arr = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width, 3)
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return Image.fromarray(cv2.cvtColor(load_espcn().upsample(bgr), cv2.COLOR_BGR2RGB))


GEMINI_OCR_PROMPT = """You are an expert OCR transcription system for Indian-language and other vernacular documents, including historical, degraded, and mixed-script print.

Transcribe only text that is visibly present in the supplied image crop. Identify the script and language from the glyphs, then reproduce the text in that original script. Preserve spelling, vowel signs, diacritics, conjuncts, punctuation, digits, capitalization, word spacing, and line breaks as printed. Follow the script's reading direction and preserve the visible reading order. For mixed-language text, transcribe each part in its own original script.

Do not translate, transliterate, modernize, normalize, correct, expand abbreviations, complete cut-off words, or paraphrase. Do not infer text hidden by blur, crop edges, stains, or damage; omit portions that cannot be read reliably. Use the surrounding visible glyphs only to distinguish ambiguous characters, not to invent missing content.

The detected region category is context only: {region_label}. Do not use it to invent text.

Return only the OCR transcription as plain text. Do not add a title, language name, label, translation, explanation, confidence estimate, quotation marks, markdown, or any other commentary. If no text is readable, return an empty response."""
GEMINI_OCR_MODELS = {
    "Gemini 3.1 Flash-Lite (faster)": "gemini-3.1-flash-lite",
    "Gemini 3.8 Flash": "gemini-3.8-flash",
}


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))


def gemini_ocr(image_jpeg: bytes, region_label: str, model_name: str) -> str:
    """One Gemini request for a single cropped region; logs timing to the terminal."""
    started = time.perf_counter()

    def stamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    print(f"[Gemini OCR] START {stamp()} | model={model_name} | class={region_label} | crop_bytes={len(image_jpeg)}", flush=True)
    api_key = gemini_key()
    try:
        from google.genai import types
        if not api_key:
            raise RuntimeError("No Gemini API key. Enter it in the sidebar, or set GEMINI_API_KEY / GOOGLE_GEMINI_API_KEY in .env.")
        response = get_gemini_client(api_key).models.generate_content(
            model=model_name,
            contents=[GEMINI_OCR_PROMPT.format(region_label=region_label),
                      types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(
                response_mime_type="text/plain", max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_level="low")),  # type: ignore
        )
        text = (response.text or "").strip()
        print(f"[Gemini OCR] SUCCESS {stamp()} | model={model_name} | class={region_label} "
              f"| elapsed={time.perf_counter() - started:.2f}s | chars={len(text)}", flush=True)
        return text
    except Exception as exc:
        message = str(exc).replace(api_key, "[hidden]") if api_key else str(exc)
        print(f"[Gemini OCR] ERROR {stamp()} | model={model_name} | class={region_label} "
              f"| elapsed={time.perf_counter() - started:.2f}s | error={type(exc).__name__}: {message}", flush=True)
        raise RuntimeError(f"Gemini OCR failed: {message}") from exc


def region_record_key(result_id: str, model_key: str, index: int) -> str:
    return f"{result_id}:{model_key}:{index}"


def ensure_region_ocr(result_id: str, model_key: str, index: int, detection: dict,
                      bounds: tuple[int, int, int, int], crop: Image.Image, gemini_model: str | None = None) -> dict:
    """Attempt OCR once for this detection in this result set and keep its full record."""
    record_key = region_record_key(result_id, model_key, index)
    records = st.session_state.setdefault("region_records", {})
    record = records.get(record_key)
    if record is not None and record.get("ocr", {}).get("attempted"):
        return record
    x1, y1, x2, y2 = bounds
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
    model_name = gemini_model or GEMINI_OCR_MODELS[st.session_state.get("gemini_model", next(iter(GEMINI_OCR_MODELS)))]
    record = {
        "region_id": f"{model_key}-{index + 1}", "model": model_key, "index": index,
        "class": detection["label"], "source_class": detection.get("source_label", detection["label"]),
        "confidence": float(detection["conf"]), "box_xyxy_px": [x1, y1, x2, y2],
        "crop": {"width_px": crop.width, "height_px": crop.height, "format": "image/jpeg"},
        "ocr": {"attempted": True, "status": "pending", "model": model_name, "text": "", "error": None},
    }
    records[record_key] = record
    try:
        record["ocr"].update(status="complete", text=gemini_ocr(buf.getvalue(), detection["label"], model_name))
    except Exception as exc:
        record["ocr"].update(status="error", error=str(exc))
    return record


@st.dialog("Detection crop", dismissible=False)
def show_detection_crop(model_title: str, number: int, detection: dict, bounds, crop: Image.Image | None,
                        ocr_text_prefix: str, result_id: str, model_key: str, detection_index: int, name: str) -> None:
    def close() -> None:
        st.session_state["active_crop_dialog"] = None
        st.rerun()

    if bounds is None or crop is None:
        st.info("The selected detection has an invalid or empty box.")
        if st.button("Close", key=f"close_invalid_{ocr_text_prefix}"):
            close()
        return
    x1, y1, x2, y2 = bounds
    st.caption(f"{model_title} | Detection #{number} | {tr(st.session_state['lang'], detection['label'])} | "
               f"confidence {detection['conf']:.3f}")
    st.code(f"x1={x1}  y1={y1}  x2={x2}  y2={y2}   ({x2 - x1} x {y2 - y1} px)", language=None)
    original_col, upscaled_col = st.columns(2)
    original_col.image(crop, caption="Original crop", width="stretch")
    upscale_key = f"crop_upscale_{ocr_text_prefix}"
    upscale_error_key = f"crop_upscale_error_{ocr_text_prefix}"
    upscaled = st.session_state.get(upscale_key)
    if upscaled is None and upscaled_col.button("Upscale crop with ESPCN x2", key=upscale_key + "_button"):
        try:
            with st.spinner("Upscaling crop"):
                upscaled = upscale_espcn(crop.tobytes(), crop.width, crop.height)
            st.session_state[upscale_key] = upscaled
            st.session_state.pop(upscale_error_key, None)
        except Exception as exc:
            st.session_state[upscale_error_key] = str(exc)
        st.rerun()
    if upscaled is not None:
        upscaled_col.image(upscaled, caption=f"ESPCN x2 | {upscaled.width} x {upscaled.height} px", width="stretch")
    elif upscale_error := st.session_state.get(upscale_error_key):
        upscaled_col.warning(f"Could not upscale this crop: {upscale_error}")
    png = io.BytesIO()
    crop.save(png, "PNG")
    st.download_button("Download cutout (PNG)", png.getvalue(),
                       f"{name}_{model_key}_{number:02d}_{detection.get('source_label', detection['label'])}.png",
                       "image/png", key=f"cut_{ocr_text_prefix}")

    st.markdown("#### Gemini OCR")
    if not st.session_state.get("ocr_on", True):
        st.caption("OCR is switched off in the sidebar.")
    elif detection["label"] in NON_TEXT:
        st.caption("OCR is skipped for image and advertisement regions.")
    else:
        record_key = region_record_key(result_id, model_key, detection_index)
        ocr_text_key = f"ocr_text_{record_key}"
        records = st.session_state.setdefault("region_records", {})
        selected_model = GEMINI_OCR_MODELS[st.session_state["gemini_model"]]
        record = records.get(record_key)
        if record is None or not record.get("ocr", {}).get("attempted"):
            st.caption(f"Selected model: {selected_model}. Change it in the sidebar, then run OCR when ready.")
            if st.button("Run Gemini OCR", key=f"run_ocr_{record_key}", type="primary"):
                with st.spinner(f"Waiting for {selected_model}"):
                    record = ensure_region_ocr(result_id, model_key, detection_index, detection, bounds, crop, selected_model)
        if record is not None:
            if record["ocr"]["status"] == "error":
                st.error(record["ocr"]["error"])
            elif record["ocr"].get("model") != selected_model:
                st.info(f"This region was already processed with {record['ocr'].get('model')}; it will not be sent again.")
            st.session_state.setdefault(ocr_text_key, record["ocr"].get("text", ""))
            st.text_area("OCR transcription (editable)", key=ocr_text_key, height=180)
            record["ocr"]["text"] = st.session_state[ocr_text_key]
            detection["text"] = record["ocr"]["text"]
            st.caption(f"Model used: {record['ocr'].get('model')}. This region is sent to Gemini once per detection.")
        elif detection.get("text"):
            st.text_area("Text read by the page-level OCR", detection["text"], height=140, disabled=True,
                         key=f"page_ocr_{record_key}")
    if st.button("Close", key=f"close_{ocr_text_prefix}"):
        close()


def eligible_regions(results: dict) -> list[tuple[str, int, dict]]:
    return [(mk, i, d) for mk, r in results.items() for i, d in enumerate(r["dets"]) if d["label"] not in NON_TEXT]


def transcribe_regions(results: dict, page: Image.Image, result_id: str, label: str) -> tuple[int, int, int]:
    """Gemini OCR (once per region) for every text region of every model. Returns (done, errors, eligible)."""
    eligible = eligible_regions(results)
    records = st.session_state.setdefault("region_records", {})
    progress = st.progress(0, text=label)
    done = errors = 0
    for pos, (mk, i, d) in enumerate(eligible, 1):
        record = records.get(region_record_key(result_id, mk, i))
        if record is None or not record.get("ocr", {}).get("attempted"):
            bounds = crop_bounds(page, d["box"])
            if bounds is not None:
                record = ensure_region_ocr(result_id, mk, i, d, bounds, page.crop(bounds))
        if record and record["ocr"].get("status") == "complete":
            done += 1
            d["text"] = record["ocr"]["text"]
        elif record and record["ocr"].get("status") == "error":
            errors += 1
        progress.progress(pos / max(len(eligible), 1), text=f"Processed {pos} of {len(eligible)} text regions")
    progress.empty()
    return done, errors, len(eligible)


def full_export(results: dict, page: Image.Image, result_id: str, image_name: str, lang: str) -> str:
    """Every region of every model with coordinates, classes, a base64 crop and its Gemini OCR record."""
    records = st.session_state.setdefault("region_records", {})
    models = {}
    for mk, result in results.items():
        regions = []
        for i, d in enumerate(result["dets"]):
            bounds = crop_bounds(page, d["box"])
            record = records.get(region_record_key(result_id, mk, i))
            x1, y1, x2, y2 = bounds or (0, 0, 0, 0)
            if record is None:
                crop_size = {"width_px": max(0, x2 - x1), "height_px": max(0, y2 - y1), "format": "image/jpeg"}
                ocr = {"attempted": False, "status": "skipped", "model": None, "text": d.get("text"), "error": None}
                if d["label"] in NON_TEXT:
                    ocr["reason"] = "OCR skipped for image and advertisement regions"
                elif bounds is None:
                    ocr["reason"] = "The detected region has no valid crop bounds"
            else:
                crop_size, ocr = record["crop"], record["ocr"]
            crop_data = dict(crop_size)
            if bounds is not None:
                crop_data["image_data_uri"] = f"data:image/jpeg;base64,{b64_jpeg(page.crop(bounds), 82)}"
            regions.append({
                "region_id": f"{mk}-{i + 1}", "class": d["label"], "source_class": d.get("source_label", d["label"]),
                "class_local": tr(lang, d["label"]), "confidence": float(d["conf"]),
                "box_xyxy_px": [round(float(v), 1) for v in d["box"]], "box_xyxy_px_int": [x1, y1, x2, y2],
                "box_xywh_normalized": [round(x1 / page.width, 6), round(y1 / page.height, 6),
                                        round((x2 - x1) / page.width, 6), round((y2 - y1) / page.height, 6)],
                "crop": crop_data, "ocr": ocr,
            })
        models[mk] = {"name": title_of(mk), "elapsed_seconds": result["elapsed"], "regions": regions}
    return json.dumps({
        "schema_version": 1,
        "source": {"filename": image_name, "width_px": page.width, "height_px": page.height},
        "coordinate_system": "pixel xyxy, origin at top-left; normalized xywh is relative to page dimensions",
        "models": models}, indent=2, ensure_ascii=False)


def list_examples() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in EXAMPLE_DIRS:
        if d.exists():
            for p in sorted([*d.glob("*.png"), *d.glob("*.jpg")]):
                out[p.stem] = p
    return out


def file_manager_panel() -> tuple[bytes | None, str | None]:
    """Keep uploaded pages and their completion flags in this Streamlit session."""
    st.html("""<style>
      .st-key-file_manager_list { position: sticky; top: .75rem; z-index: 5;
        border: 1px solid rgba(124, 58, 237, .35); border-radius: 14px;
        background: color-mix(in srgb, var(--secondary-background-color) 94%, #7c3aed 6%);
        box-shadow: 0 8px 24px rgba(30, 20, 70, .18); }
      .st-key-file_manager_list [data-testid="stMarkdownContainer"] { font-size: .78rem; line-height: 1.2; }
      .st-key-file_manager_list [data-testid="stButton"] button,
      .st-key-file_manager_list [data-testid="stCheckbox"] label { font-size: .72rem; }
    </style>""")
    st.markdown("### 🗂️ File manager")
    st.caption("Add batches; open a page to work on it.")
    uploads = st.file_uploader("Add image files", type=["png", "jpg", "jpeg", "bmp", "webp"],
                               accept_multiple_files=True, key="file_manager_uploads", label_visibility="collapsed")
    items = st.session_state.setdefault("file_manager_items", [])
    known_ids = {item["id"] for item in items}
    processed_uploads = st.session_state.setdefault("processed_upload_file_ids", set())
    added_ids = []
    for uploaded in uploads or []:
        upload_token = getattr(uploaded, "file_id", None)
        if upload_token and upload_token in processed_uploads:
            continue
        data = uploaded.getvalue()
        digest = hashlib.sha256(uploaded.name.encode("utf-8") + b"\0" + data).hexdigest()[:16]
        if digest not in known_ids:
            thumbnail_data = None
            try:
                with Image.open(io.BytesIO(data)) as im:
                    thumb = im.convert("RGB")
                    thumb.thumbnail((64, 64))
                    thumb_buffer = io.BytesIO()
                    thumb.save(thumb_buffer, format="JPEG", quality=75)
                    thumbnail_data = thumb_buffer.getvalue()
            except Exception:
                pass
            items.append({"id": digest, "name": uploaded.name, "data": data, "thumbnail": thumbnail_data})
            known_ids.add(digest)
            added_ids.append(digest)
        if upload_token:
            processed_uploads.add(upload_token)
    if added_ids and not st.session_state.get("selected_manager_file_id"):
        st.session_state["selected_manager_file_id"] = added_ids[0]

    if not items:
        st.info("No files uploaded yet.")
        return None, None

    selected_id = st.session_state.get("selected_manager_file_id")
    if selected_id not in known_ids:
        selected_id = items[0]["id"]
        st.session_state["selected_manager_file_id"] = selected_id

    with st.container(height=620, border=True, key="file_manager_list"):
        for index, item in enumerate(items, 1):
            thumb_col, name_col, done_col = st.columns([0.8, 2.4, 0.8], vertical_alignment="center")
            if item.get("thumbnail"):
                thumb_col.image(Image.open(io.BytesIO(item["thumbnail"])), width=54)
            else:
                thumb_col.caption("No preview")
            name_col.markdown(f"**{index}. {escape(item['name'])}**")
            if name_col.button("Open", key=f"open_file_{item['id']}", width="stretch"):
                st.session_state["selected_manager_file_id"] = item["id"]
                selected_id = item["id"]
            done_key = f"file_done_{item['id']}"
            st.session_state.setdefault(done_key, False)
            done_col.checkbox("Done", key=done_key, label_visibility="collapsed", help="Mark this page finished")

    selected = next((item for item in items if item["id"] == selected_id), None)
    if selected is None:
        return None, None
    return selected["data"], selected["name"]


WORKSPACE_KEYS = ("sig", "result", "shown_img", "result_id", "region_records", "annotation_click_ids")


def activate_file_workspace(file_id: str | None) -> None:
    """Keep each uploaded page's detections and OCR in this session when switching pages."""
    active_id = st.session_state.get("active_manager_file_id")
    if active_id == file_id:
        return
    workspaces = st.session_state.setdefault("file_manager_workspaces", {})
    if active_id and st.session_state.get("sig", (None,))[0] == "File manager":
        workspaces[active_id] = {k: st.session_state[k] for k in WORKSPACE_KEYS if k in st.session_state}
    for key in (*WORKSPACE_KEYS, "prepared_full_json", "active_crop_dialog"):
        st.session_state.pop(key, None)
    if file_id and file_id in workspaces:
        st.session_state.update(workspaces[file_id])
    st.session_state["active_manager_file_id"] = file_id


def coord_rows(dets: list[dict], lang: str) -> list[dict]:
    rows = []
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [round(v) for v in d["box"]]
        rows.append({"#": i, "Region": tr(lang, d["label"]), "Confidence": round(d["conf"], 3),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2, "Width": x2 - x1, "Height": y2 - y1, **({"Text": d.get("text", "")} if "text" in d else {})})
    return rows


def detections_json(dets: list[dict], size: tuple[int, int], lang: str, source: str) -> dict:
    """Exported annotations. `bbox_xyxy` is [x1, y1, x2, y2] in pixels of the original page, origin top-left."""
    w, h = size
    out = []
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [round(v, 1) for v in d["box"]]
        out.append(dict(id=i, label=d["label"], name_local=tr(lang, d["label"]), class_id=d["cls"], confidence=round(d["conf"], 4),
                        bbox_xyxy=[x1, y1, x2, y2], bbox_xywh=[x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)],
                        bbox_normalized=[round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)],
                        **({"text": d["text"]} if "text" in d else {})))
    return dict(image=source, width=w, height=h, coordinate_system="pixels, origin top-left, x to the right, y downwards",
                count=len(out), detections=out)


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
    result_id = st.session_state.get("result_id", "")
    ocr_on_now = st.session_state.get("ocr_on", True)
    if ocr_on_now:
        st.markdown("### 🧾 Gemini transcription")
        st.caption(f"Model: {GEMINI_OCR_MODELS[st.session_state['gemini_model']]}. Every text region is sent once; "
                   "images and advertisements are skipped.")
        if st.button("Transcribe all regions", key=f"transcribe_all_{result_id}", type="primary"):
            done, errors, total = transcribe_regions(results, page, result_id, "Preparing OCR")
            st.success(f"Transcription finished: {done} completed, {errors} errors, {total} eligible regions.")
    tabs = st.tabs([f"{FUSION.icon if k == 'fusion' else MODELS[k].icon} {title_of(k)}" for k in keys])
    context_key = hashlib.sha1(repr((st.session_state.get("sig"), result_id)).encode("utf-8")).hexdigest()[:12]
    click_events = st.session_state.setdefault("annotation_click_ids", {})
    page_image_b64 = b64_jpeg(page, 85)
    for tab, key in zip(tabs, keys):
        r = results[key]
        with tab:
            engine = st.session_state.get("ocr_engine", "gemini")
            ocr_on = st.session_state.get("ocr_on", True)
            if ocr_on and key == "fusion" and not r.get("auto_ocr"):  # Ensemble Fusion is read automatically, once
                r["auto_ocr"] = True
                use = engine if engine != "gemini" or gemini_key() else "easyocr"
                with st.spinner(f"Reading text in the fused regions ({'Gemini' if use == 'gemini' else 'EasyOCR'})…"):
                    try:
                        ocr_page(page, r["dets"], device_name(), None, use)
                    except Exception as e:
                        st.warning(f"Automatic OCR failed: {e}")
                        for d in r["dets"]:
                            d.pop("text", None)
            has_text = any("text" in d for d in r["dets"])
            nice = "Gemini" if engine == "gemini" else "EasyOCR"
            if ocr_on and st.button(f"📝 {'Re-run' if has_text else 'Extract text'} with {nice} (Bengali + English)", key=f"ocr_{key}"):
                bar = st.progress(0.0, text=f"Starting {nice}…")
                try:
                    ocr_page(page, r["dets"], device_name(), lambda p, t: bar.progress(p, text=t), engine)
                except (PermissionError, RuntimeError) as e:
                    st.error(str(e))
                except ImportError:
                    st.error("OCR needs EasyOCR: pip install easyocr")
                bar.empty()
                has_text = any("text" in d for d in r["dets"])
            st.caption("💡 Click any box to open its cutout: coordinates, PNG download, ESPCN x2 upscale"
                       + (" and Gemini OCR." if ocr_on else ". OCR is switched off in the sidebar."))
            component_key = f"annotations_{context_key}_{key}"
            click = annotation_view(
                image_b64=page_image_b64, image_width=page.width, image_height=page.height,
                detections=[{
                    "box": [float(v) for v in d["box"]], "label": d["label"], "name": tr(lang, d["label"]),
                    "conf": float(d["conf"]), "color": color_of(d),
                    "transcribed": bool(d.get("text")) or bool(
                        (rec := st.session_state.get("region_records", {}).get(region_record_key(result_id, key, n)))
                        and rec.get("ocr", {}).get("status") == "complete"),
                } for n, d in enumerate(r["dets"])],
                key=component_key, default=None)
            if isinstance(click, dict) and click.get("nonce") != click_events.get(component_key):
                index = click.get("index")
                if isinstance(index, int) and 0 <= index < len(r["dets"]):
                    click_events[component_key] = click["nonce"]
                    bounds = crop_bounds(page, r["dets"][index]["box"])
                    st.session_state["active_crop_dialog"] = {
                        "model_title": title_of(key), "number": index + 1, "detection": r["dets"][index],
                        "bounds": bounds, "crop": page.crop(bounds) if bounds else None,
                        "ocr_text_prefix": f"{context_key}_{key}_{index}", "result_id": result_id,
                        "model_key": key, "detection_index": index, "name": image_name,
                    }
            st.html(legend_html(lang, {d["label"] for d in r["dets"]}))
            buf = io.BytesIO()
            annotate_png(page, r["dets"]).save(buf, "PNG")
            c1, c2, c3 = st.columns(3)
            c1.download_button("⬇ Annotated image (PNG)", buf.getvalue(), f"{image_name}_{key}.png", "image/png", key=f"dp_{key}")
            if has_text:
                txt = "\n\n".join(f"[#{i} {d['label']}]\n{d['text']}" for i, d in enumerate(r["dets"], 1) if d.get("text"))
                c3.download_button("⬇ Extracted text (TXT)", txt, f"{image_name}_{key}.txt", "text/plain", key=f"dt_{key}")
            payload = json.dumps(detections_json(r["dets"], page.size, lang, image_name), indent=2, ensure_ascii=False)
            c2.download_button("⬇ Detections (JSON)", payload, f"{image_name}_{key}.json", "application/json", key=f"dj_{key}")
            if key == "fusion" and has_text:
                masthead = next((d["text"] for d in r["dets"] if d.get("source_label", d["label"]) == "flag" and d.get("text")), "")
                st.download_button("📰 Download newspaper as HTML (news site, from Ensemble Fusion)",
                                   news_html([{**d, "label": d.get("source_label", d["label"])} for d in r["dets"]], image_name, masthead), f"{image_name}_news.html", "text/html",
                                   key="dn_fusion", type="primary")
            with st.expander(f"📐 Coordinates of all {len(r['dets'])} boxes (pixels, top-left origin)"):
                st.dataframe(coord_rows(r["dets"], lang), width="stretch", hide_index=True)


    active_dialog = st.session_state.get("active_crop_dialog")
    if active_dialog:
        show_detection_crop(**active_dialog)

    st.markdown("### 📦 Full page cutout data")
    st.caption("Coordinates, source and grouped classes, confidence, a crop of every region and its Gemini OCR record."
               + (" Preparing it runs OCR once for each text region not already processed." if ocr_on_now else ""))
    if st.button("Prepare full JSON export", key="prepare_full_json"):
        if ocr_on_now:
            transcribe_regions(results, page, result_id, "Preparing OCR and cutout data")
        st.session_state["prepared_full_json"] = full_export(results, page, result_id, image_name, lang)
    if st.session_state.get("prepared_full_json"):
        st.download_button("⬇ Download full JSON", st.session_state["prepared_full_json"],
                           f"{image_name}_cutouts.json", "application/json", key="download_full_json")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.html(CSS)
    st.html('<div class="hero"><h1>IndicDLP Live Layout Studio</h1>'
            '<p>Upload pages in the file manager and watch YOLOv10x and DocLayout-YOLO read each layout, region by region, '
            'in English and 11 Indic languages.</p></div>')
    cfg = sidebar()
    lang = st.session_state["lang"]

    workspace_col, manager_col = st.columns([3.5, 1.2], gap="large")
    with manager_col:
        selected_file_data, selected_file_name = file_manager_panel()

    with workspace_col:
        examples = list_examples()
        src = st.radio("Source", ["File manager", "Sample gallery"], horizontal=True, label_visibility="collapsed")
        activate_file_workspace(st.session_state.get("selected_manager_file_id") if src == "File manager" else None)
        image: Image.Image | None = None
        name = "document"
        if src == "File manager":
            if selected_file_data is not None:
                try:
                    image = Image.open(io.BytesIO(selected_file_data)).convert("RGB")
                    name = Path(selected_file_name or "document").stem
                except Exception:
                    st.error("That file couldn't be read as an image.")
            else:
                st.info("⬆️ Upload one or more pages in the File manager on the right.")
        elif examples:
            pick = st.selectbox("Sample page", list(examples), index=0)
            image = Image.open(examples[pick]).convert("RGB")
            name = pick
        else:
            st.warning("No sample images found.")

        if image is None:
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
                     + (" · **multi-scale**" if cfg["boost"] else "")
                     + (" · **6 simplified classes**" if cfg["simplify"] else ""))
            go = st.button("🚀 Scan & detect", type="primary", width="stretch")

        sig = (src, name, image.size, image_fingerprint(image), tuple(cfg["models"]), cfg["conf"], cfg["iou"],
               cfg["imgsz"], cfg["boost"], cfg["fuse"], cfg["simplify"])
        stale = ("shown_img", "sig", "annotation_click_ids", "result_id", "active_crop_dialog", "region_records",
                 "prepared_full_json")
        if st.session_state.get("sig") != sig:
            st.session_state["result"] = None
            for k in stale:
                st.session_state.pop(k, None)
        if go:
            st.session_state["result"] = None
            for k in stale:
                st.session_state.pop(k, None)
            st.session_state["sig"] = sig
            st.divider()
            analyse(image, cfg, lang)
            st.balloons()
        if st.session_state.get("sig") == sig:
            summary(name, lang)


main()
