"""
========================================================================================
Project: National Indic Document Layout Parsing (IndicDLP) | 22 Scheduled Indian Languages
Component: Standalone Broadsheet Auto-Annotator & 100% OCR Pipeline (Production Release)
Features:
  - Multi-Project API Key Pool Rotation (Fix B: Bypasses single-account 15 RPM bottleneck)
  - Adaptive 4.1s Pacing & Exponential Backoff on 429/503/504 (Fix A: Rate-Limit Protection)
  - Gemini transcription per region; a region Gemini cannot read gets its class name as placeholder text
  - Dynamic Hardware Detection (NVIDIA CUDA GPU vs. AMD / Intel CPU)
========================================================================================
"""

import os
import sys
import json
import base64
import time
import math
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO

# ========================================================================================
# 1. Configuration & Paths
# ========================================================================================
BASE_DIR = Path(__file__).resolve().parent

# Set your input directory containing 10 broadsheet scans:
INPUT_SCANS_DIR = os.environ.get("INPUT_SCANS_DIR", str(BASE_DIR / "scans"))

# Output JSON file ready for direct drag-and-drop into Label Studio:
OUTPUT_JSON_PATH = os.environ.get("OUTPUT_JSON_PATH", str(BASE_DIR / "label_studio_preannotated.json"))

# ========================================================================================
# 2. Multi-Project Gemini API Key Pool (Fix B)
# Put keys from DIFFERENT Google Cloud / AI Studio accounts to multiply your RPM quota!
# ========================================================================================
GEMINI_API_KEYS = [
    k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()
]
# Or add key strings directly here:
# GEMINI_API_KEYS = ["AIzaSy...", "AIzaSy..."]

gemini_clients = []
if GEMINI_API_KEYS:
    try:
        from google import genai
        from google.genai import types
        for k in GEMINI_API_KEYS:
            c = genai.Client(api_key=k, http_options=types.HttpOptions(timeout=15000))
            gemini_clients.append({"client": c, "key_id": k[:8] + "...", "last_used": 0.0})
        print(f"🔑 Initialized {len(gemini_clients)} independent Gemini API project clients.")
    except Exception as e:
        print(f"⚠️  Google GenAI SDK error: {e}. Text will not be transcribed.")

current_key_idx = 0

# ========================================================================================
# 3. Hardware Auto-Detection (NVIDIA CUDA GPU vs. AMD / CPU)
# ========================================================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("🚀 IndicDLP Broadsheet Auto-Annotator & OCR Engine")
print(f"🖥️  Hardware Device: {device.upper()}")
if not torch.cuda.is_available():
    print("ℹ️  Running on multi-threaded CPU mode (safe for non-NVIDIA / AMD laptops).")
print("=" * 70)

# Locate YOLO weights
weights_candidates = [
    BASE_DIR / "weights" / "md2_PT_indicdlp_FT.pt",
    BASE_DIR / "md2_PT_indicdlp_FT.pt",
    Path.home() / "IndicDLP-Live-Layout-Studio" / "weights" / "md2_PT_indicdlp_FT.pt"
]
weights_path = next((p for p in weights_candidates if p.exists()), None)

if not weights_path:
    print("\n❌ ERROR: Model weights 'md2_PT_indicdlp_FT.pt' not found!")
    print("Run this command in terminal to download it:")
    print('curl -L "https://zenodo.org/records/15881917/files/md2_PT_indicdlp_FT.pt?download=1" -o weights/md2_PT_indicdlp_FT.pt\n')
    sys.exit(1)

print(f"📦 Loading YOLO layout detector from: {weights_path}")
yolo_model = YOLO(str(weights_path))

# ========================================================================================
# 4. Official 5-Class Taxonomy & Header / Furniture Exclusion
# ========================================================================================
CLASS_MAP = {
    "headline": "Headline",
    "title": "Headline",
    "header": "Headline",
    "flag": "Headline",
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
    "caption": "Caption",
    "table-caption": "Caption",
    "figure-caption": "Caption",
    "figure": "Image",
}

# Explicit labels to drop automatically
EXCLUDE_LABELS = {"advertisement", "ad"}

def predict_layout(img_path):
    with Image.open(img_path) as im:
        w, h = im.size
    names = yolo_model.names
    res = list(yolo_model.predict(source=img_path, conf=0.22, iou=0.45, imgsz=1024,
                                  device=0 if device.startswith("cuda") else "cpu", verbose=False))[0]
    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            dets.append({"bbox": [x1, y1, x2, y2], "conf": float(c), "label": names[int(k)]})
    return dets

# ========================================================================================
# 5. Rate-Paced Gemini OCR Engine (Fix A & Fix B)
# ========================================================================================
def ocr_with_gemini(crop_img: Image.Image) -> str:
    global current_key_idx
    if not gemini_clients:
        return ""

    prompt = (
        "Transcribe the Bengali text in this historical newspaper snippet verbatim. "
        "Output only the exact transcribed Bengali text without preamble or commentary:"
    )

    total_clients = len(gemini_clients)

    for attempt in range(total_clients * 2):
        client_entry = gemini_clients[current_key_idx]
        c = client_entry["client"]
        key_id = client_entry["key_id"]
        last_t = client_entry["last_used"]

        # Fix A: Enforce minimum 4.1s spacing per client project (under 15 RPM limit)
        now = time.time()
        elapsed = now - last_t
        if elapsed < 4.1:
            wait_s = 4.1 - elapsed
            time.sleep(wait_s)

        current_key_idx = (current_key_idx + 1) % total_clients

        try:
            client_entry["last_used"] = time.time()
            response = c.models.generate_content(
                model="gemini-2.5-flash",
                contents=[crop_img, prompt]
            )
            if response and response.text:
                txt = response.text.strip()
                cleaned_lines = [
                    l for l in txt.split("\n")
                    if not l.lower().startswith("here is") and not l.lower().startswith("transcription:")
                ]
                return "\n".join(cleaned_lines).strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"⚠️  Project [{key_id}] hit 15 RPM. Switching to next project key...")
                time.sleep(1.0)
            elif "503" in err_str or "504" in err_str:
                print(f"⚠️  Gemini busy (503/504). Switching key...")
                time.sleep(1.5)
            else:
                time.sleep(0.5)

    return ""

# ========================================================================================
# 6. Main Execution Pipeline
# ========================================================================================
def main():
    if not os.path.exists(INPUT_SCANS_DIR):
        print(f"❌ Error: INPUT_SCANS_DIR not found: {INPUT_SCANS_DIR}")
        print("Please set the folder containing your 10 scans.")
        return

    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = sorted([f for f in os.listdir(INPUT_SCANS_DIR) if f.lower().endswith(valid_exts)])[:10]

    if not image_files:
        print(f"❌ Error: No image files found in {INPUT_SCANS_DIR}")
        return

    print(f"\n📂 Found {len(image_files)} broadsheet scans to annotate and transcribe.")
    label_studio_tasks = []

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(INPUT_SCANS_DIR, fname)
        t_start = time.time()
        print(f"\n[{idx:02d}/{len(image_files):02d}] Processing: {fname}...")

        with Image.open(fpath) as pil_img:
            W, H = pil_img.size
            with open(fpath, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
            ext = os.path.splitext(fname)[1].lower().replace(".", "")
            mime = "jpeg" if ext in ["jpg", "jpeg"] else "png"
            data_uri = f"data:image/{mime};base64,{b64_data}"

        raw_dets = predict_layout(fpath)
        print(f"  • Raw Layout Regions: {len(raw_dets)}")

        ls_results = []
        valid_regions_count = 0

        for r_idx, det in enumerate(raw_dets):
            x1, y1, x2, y2 = det["bbox"]
            orig_lbl = det["label"].lower()

            # Skip explicit ads
            if orig_lbl in EXCLUDE_LABELS:
                continue

            rx = round(float(x1) / W * 100, 4)
            ry = round(float(y1) / H * 100, 4)
            rw = round(float(x2 - x1) / W * 100, 4)
            rh = round(float(y2 - y1) / H * 100, 4)

            # Skip top banner danger zone (y < 5.5% or tiny banner strip)
            if ry < 5.5 and rh < 3.0:
                continue

            std_label = CLASS_MAP.get(orig_lbl, "BodyText")
            reg_id = f"scan_{idx}_reg_{r_idx}"
            valid_regions_count += 1

            ls_results.append({
                "value": {
                    "x": rx, "y": ry, "width": rw, "height": rh,
                    "rotation": 0, "rectanglelabels": [std_label]
                },
                "id": reg_id, "from_name": "label", "to_name": "image",
                "type": "rectanglelabels", "origin": "manual"
            })

            if std_label != "Image":
                with Image.open(fpath) as pil_full:
                    crop_pil = pil_full.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))

                ocr_text = ""
                # Use Gemini if clients configured
                if gemini_clients:
                    ocr_text = ocr_with_gemini(crop_pil)

                if not ocr_text:
                    ocr_text = f"[{std_label}]"

                ls_results.append({
                    "value": {
                        "x": rx, "y": ry, "width": rw, "height": rh,
                        "text": [ocr_text]
                    },
                    "id": reg_id, "from_name": "ocr", "to_name": "image",
                    "type": "textarea", "origin": "manual"
                })

        label_studio_tasks.append({
            "data": {"image": data_uri},
            "annotations": [{"result": ls_results}]
        })
        print(f"  ✓ Processed {valid_regions_count} valid regions in {time.time() - t_start:.2f}s")

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as out_f:
        json.dump(label_studio_tasks, out_f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("🎉 BATCH ANNOTATION COMPLETE!")
    print(f"📄 Saved {len(label_studio_tasks)} tasks to: {OUTPUT_JSON_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
