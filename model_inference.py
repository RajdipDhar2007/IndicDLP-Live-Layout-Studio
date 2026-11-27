"""
========================================================================================
Project: National Indic Document Layout Parsing (IndicDLP) | 22 Scheduled Indian Languages
Component: Headless Model Inference & Structural Parser Engine
Institution: IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata / DST
Supervision: Dr. Sujoy Kumar Biswas, Dr. Ritam Ghosh, Prof. Dipti Prasad Mukherjee
Author: Research Intern Team
Timestamp: 2026-10-28 22:25:00 IST
Branch: feature/information-theoretic-reading-order

Context:
--------
Decouples core YOLOv10/DocLayout-YOLO model loading, inference, and layout prediction
from the Streamlit web dashboard. Allows headless, warning-free execution inside
Jupyter Notebooks, automated batch pipelines, and evaluation scripts.
========================================================================================
"""

from __future__ import annotations
import os
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"

@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    blurb: str
    file: str
    kind: str
    map: str
    icon: str
    size: int

MODELS = {
    "yolov10x": ModelSpec("yolov10x", "YOLOv10x", "Best single-model accuracy, pretrained on UED-mini then fine-tuned on IndicDLP.",
                          "md2_PT_indicdlp_FT.pt", "yolo", "57.7", "⚡", 64280114),
    "doclayout": ModelSpec("doclayout", "DocLayout-YOLO", "Document-specialised YOLOv10 variant with global-to-local perception.",
                           "doclayout_yolo_indicdlp.pt", "doclayout", "54.5", "🧭", 161475350),
}

CLASS_NAMES = [
    "paragraph", "title", "sub-title", "heading", "headline", "header", "footer", "abstract",
    "table", "figure", "caption", "ordered-list", "unordered-list", "citation", "advertisement",
    "sidebar", "formula", "page-number", "footnote", "stamp", "signature", "table-of-contents",
    "index", "author", "affiliation", "date", "flag", "dateline", "jumpline", "options",
    "first-level-question", "second-level-question", "third-level-question", "answer",
    "contact-info", "placeholder-text", "website-link", "key-value", "folio", "editorial",
    "supplement", "letter",
]

def model_status(spec: ModelSpec) -> tuple[bool, str]:
    path = WEIGHTS / spec.file
    if not path.exists():
        return False, "Weights not downloaded."
    if path.stat().st_size < spec.size:
        return False, "Weights incomplete (download still running or interrupted)."
    return True, ""

@lru_cache(maxsize=4)
def load_model(key: str, device: str = "cpu"):
    """Loads and caches the model without any Streamlit dependency."""
    spec = MODELS[key]
    path = str(WEIGHTS / spec.file)
    if spec.kind == "yolo":
        from ultralytics import YOLO
        return YOLO(path)
    from doclayout_yolo import YOLOv10
    return YOLOv10(path)

def predict_once(key: str, image: Image.Image, conf: float = 0.25, iou: float = 0.45, imgsz: int = 1024, device: str = "cpu") -> list[dict]:
    """Runs a single inference pass and returns unified list of detection dictionaries."""
    model = load_model(key, device)
    bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])
    res = list(model.predict(bgr, imgsz=imgsz, conf=conf, iou=iou,
                             device=0 if device.startswith("cuda") else "cpu", verbose=False))[0]
    names = res.names
    return [dict(box=(x1, y1, x2, y2), conf=float(c), cls=int(k), label=names[int(k)])
            for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]

def reading_order(dets: list[dict]) -> list[dict]:
    """Information-Theoretic reading order reconstruction."""
    try:
        from article_stitcher import information_theoretic_reading_order
        return information_theoretic_reading_order(dets)
    except Exception:
        return sorted(dets, key=lambda d: (round(d["box"][1] / 40), d["box"][0]))
