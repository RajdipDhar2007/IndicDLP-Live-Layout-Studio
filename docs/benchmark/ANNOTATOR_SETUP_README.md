# Standalone IndicDLP Broadsheet Auto-Annotator & OCR Setup Guide

This guide walks you through setting up and running the automated layout annotation and Bengali OCR pipeline (`annotator_standalone.py` or `run_hybrid_gpu_gemini_annotator.py`).

---

## ⚡ 1. Automatic Hardware Detection (GPU & CPU)

The pipeline **automatically detects your available hardware**:
* **NVIDIA GPU (CUDA):** If an NVIDIA GPU is detected, PyTorch loads YOLO and EasyOCR directly into GPU VRAM for ultra-fast processing (~1.2s per broadsheet).
* **CPU / AMD Graphics:** If no CUDA GPU is detected (e.g. Intel/AMD laptops), the pipeline **automatically falls back to multi-threaded CPU mode**:
  ```python
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  use_gpu_ocr = True if torch.cuda.is_available() else False
  ```
  It dynamically adjusts data loaders, disables pinned memory, and executes inference cleanly without crashing.

---

## 📦 2. Environment Setup & Dependencies

Ensure you have Python 3.10+ installed.

```bash
# 1. Create and activate a virtual environment
python -m venv OCRvenv
# On Linux/WSL:
source OCRvenv/bin/activate
# On Windows PowerShell:
.\OCRvenv\Scripts\Activate.ps1

# 2. Install PyTorch matching your hardware
# For NVIDIA CUDA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# For CPU only:
pip install torch torchvision

# 3. Install core libraries
pip install ultralytics easyocr google-genai Pillow numpy scikit-learn matplotlib
```

---

## 📥 3. Download the YOLO Model Weights

The custom fine-tuned IndicDLP layout detector (`md2_PT_indicdlp_FT.pt`) is hosted on Zenodo (Record 15881917).

Run this command in your terminal to download it into the `weights/` folder:

```bash
mkdir -p weights
curl -L "https://zenodo.org/records/15881917/files/md2_PT_indicdlp_FT.pt?download=1" -o weights/md2_PT_indicdlp_FT.pt
```

*(Note: If running `annotator_standalone.py`, place `md2_PT_indicdlp_FT.pt` in the same directory as the script or inside `~/IndicDLP-Live-Layout-Studio/weights/`)*.

---

## 🚀 4. Running the Pipeline

### Step 1: Configure Paths
Open `run_hybrid_gpu_gemini_annotator.py` (or `annotator_standalone.py`) and set:
```python
INPUT_DIR = r"path/to/your/10_newspaper_scans_folder"
OUTPUT_JSON = r"path/to/save/label_studio_preannotated.json"
```

### Step 2: Trigger the Script
```bash
python run_hybrid_gpu_gemini_annotator.py
```

### Step 3: What to Expect in Terminal Output
```text
🖥️  Device Mode: CUDA (or CPU fallback)
📦 Loading YOLOv10x layout model...
✓ YOLOv10x loaded successfully!
📂 Found 10 newspaper scans to process.
[01/10] Processing: 1946-01-01_page_001.jpg
  • GPU Layout Detection (1.18s): Found 69 regions.
  • Transcribing key Bengali regions...
  ✓ Finished page in 4.2s (Total regions: 69)
...
🎉 SUCCESS! Pre-annotated all 10 tasks into output JSON.
```

---

## 🎨 5. Drag & Drop into Label Studio

1. Open Label Studio (`http://localhost:8080`).
2. Create a project and paste the official 5-class XML template:
   * `Headline`, `Subtitle`, `BodyText`, `Caption`, `Image` with per-region `TextArea`.
3. Import the generated `OUTPUT_JSON` directly.
4. **All 10 pages will appear instantly with all bounding boxes and Bengali OCR pre-filled!**
