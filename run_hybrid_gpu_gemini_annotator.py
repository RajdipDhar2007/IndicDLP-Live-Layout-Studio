# Project: National Indic Document Layout Parsing (IndicDLP) | IDEAS-TIH, ISI Kolkata
# Pipeline: Hybrid GPU Layout Detection (RTX 2050 CUDA) + Gemini 3.5 Flash Lite Bengali OCR
# Target: 10 Historical Broadsheet Scans
# Output: Label Studio Pre-annotated Tasks JSON (Matching Dr. Ritam Ghosh classes)

import os
import sys
import json
import time
import uuid
import torch
from PIL import Image
from google import genai

# Configure Paths
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.environ.get("INPUT_SCANS_DIR", os.path.join(REPO_DIR, "scans"))
OUTPUT_JSON = os.environ.get("OUTPUT_JSON_PATH", os.path.join(REPO_DIR, "label_studio_preannotated.json"))

sys.path.append(REPO_DIR)
from model_inference import load_model, predict_once

# 1. Setup Official google-genai Client
API_KEY = os.environ.get('GEMINI_API_KEY', '')
client = genai.Client(api_key=API_KEY)
OCR_MODEL = "gemini-3.5-flash-lite"

# 2. Check Device for YOLO
device = "cuda:0" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if "cuda" in device else "CPU"
print(f"🚀 Layout Detector running on: {device} ({gpu_name})")

# Load YOLO model onto RTX 2050 GPU
print("📦 Loading YOLOv10x layout model into RTX 2050 GPU VRAM...")
model = load_model("yolov10x", device=device)
print("✓ YOLOv10x loaded successfully onto RTX 2050!")

# Class mapping from YOLO labels to Dr. Ritam's 5 Label Studio classes:
CLASS_MAP = {
    "headline": "Headline",
    "title": "Headline",
    "flag": "Headline",
    "header": "Headline",
    "sub-title": "Subtitle",
    "sub-headline": "Subtitle",
    "heading": "Subtitle",
    "section-title": "Subtitle",
    "sub-section-title": "Subtitle",
    "paragraph": "BodyText",
    "ordered-list": "BodyText",
    "unordered-list": "BodyText",
    "table": "BodyText",
    "table-of-contents": "BodyText",
    "index": "BodyText",
    "sidebar": "BodyText",
    "advertisement": "BodyText",
    "caption": "Caption",
    "table-caption": "Caption",
    "figure-caption": "Caption",
    "figure": "Image",
}

def transcribe_crop_with_gemini(crop_img: Image.Image, label: str) -> str:
    """Uses Gemini 3.5 Flash Lite to transcribe Bengali text inside a cropped bounding box."""
    if label == "Image":
        return ""
    
    prompt = "Transcribe the Bengali text in this cropped historical newspaper snippet verbatim. Do not add commentary or notes. Output only the exact transcribed text:"
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=OCR_MODEL,
                contents=[crop_img, prompt]
            )
            text = response.text.strip() if response and response.text else ""
            return text
        except Exception as e:
            time.sleep(1.5)
    return ""

def process_scans():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n📂 Found {len(image_files)} test newspaper scans in: {INPUT_DIR}")
    
    tasks = []

    for idx, fname in enumerate(image_files):
        fpath = os.path.join(INPUT_DIR, fname)
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {fname}", flush=True)
        
        t0 = time.time()
        orig_img = Image.open(fpath).convert("RGB")
        w_orig, h_orig = orig_img.size
        
        # 1. GPU Layout Inference (RTX 2050)
        t_gpu_start = time.time()
        raw_dets = predict_once("yolov10x", orig_img, conf=0.22, iou=0.45, imgsz=1024, device=device)
        gpu_time = time.time() - t_gpu_start
        print(f"  • GPU Layout Detection ({gpu_time:.2f}s): Found {len(raw_dets)} regions.", flush=True)

        # Filter and map labels
        mapped_regions = []
        for d in raw_dets:
            lbl = d["label"].lower()
            mapped_lbl = CLASS_MAP.get(lbl, "BodyText")
            mapped_regions.append({
                "box": d["box"],
                "label": mapped_lbl,
                "conf": d["conf"]
            })

        # 2. Transcribe with Gemini (Focus on top prominent elements: Headlines, Subtitles, Top Columns)
        def region_priority(r):
            x1, y1, x2, y2 = r["box"]
            area = (x2 - x1) * (y2 - y1)
            priority_score = 0
            if r["label"] == "Headline": priority_score = 3
            elif r["label"] == "Subtitle": priority_score = 2
            elif r["label"] == "Caption": priority_score = 1
            return (priority_score, area)

        mapped_regions = sorted(mapped_regions, key=region_priority, reverse=True)

        results = []
        ocr_budget = min(10, len(mapped_regions))
        print(f"  • Running Gemini 3.5 Flash Lite Bengali OCR on {ocr_budget} key regions...", flush=True)

        for r_idx, r in enumerate(mapped_regions):
            x1, y1, x2, y2 = r["box"]
            bw = x2 - x1
            bh = y2 - y1
            
            x_pct = (x1 / w_orig) * 100.0
            y_pct = (y1 / h_orig) * 100.0
            w_pct = (bw / w_orig) * 100.0
            h_pct = (bh / h_orig) * 100.0

            region_id = str(uuid.uuid4())[:8]

            # Run OCR on top priority regions
            transcription = ""
            if r_idx < ocr_budget and r["label"] != "Image":
                crop = orig_img.crop((int(x1), int(y1), int(x2), int(y2)))
                transcription = transcribe_crop_with_gemini(crop, r["label"])
                if transcription:
                    snippet = transcription[:45].replace('\n', ' ')
                    print(f"    [{r['label']:<9}] -> {snippet}...", flush=True)
                time.sleep(0.4)

            # Bounding box result
            bbox_result = {
                "id": region_id,
                "type": "rectanglelabels",
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": [r["label"]]
                },
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0,
                "original_width": w_orig,
                "original_height": h_orig
            }
            results.append(bbox_result)

            # Per-region transcription result (matching Dr. Ritam's right sidebar)
            if transcription:
                text_result = {
                    "id": region_id,
                    "type": "textarea",
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rotation": 0,
                        "text": [transcription]
                    },
                    "to_name": "image",
                    "from_name": "transcription"
                }
                results.append(text_result)

        # Build Label Studio Task object
        task = {
            "data": {
                "image": f"/data/local-files/?d={fname}"
            },
            "predictions": [
                {
                    "model_version": "gemini-ocr-1",
                    "result": results
                }
            ]
        }
        tasks.append(task)
        print(f"  ✓ Finished {fname} in {time.time() - t0:.2f}s (Total regions: {len(mapped_regions)})", flush=True)

    # Save output JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"🎉 SUCCESS! Pre-annotated all {len(tasks)} tasks.", flush=True)
    print(f"📄 Output saved to: {OUTPUT_JSON}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    process_scans()
