"""Headless run of the IndicDLP Live Layout Studio ensemble (same functions/settings as the Streamlit app).

Usage: python run_ensemble.py <out_dir> <image> [<image> ...]
Writes <out_dir>/<stem>.json with pixel boxes (xyxy) of the Ensemble Fusion output, in the app's reading order.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app as A  # noqa: E402  (importing runs only module-level setup; main() is guarded)
from PIL import Image  # noqa: E402

CFG = dict(conf=0.25, iou=0.70, imgsz=1024, boost=True, fuse=True)  # the app's defaults, as shown in its sidebar

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
device = A.device_name()
paths = [Path(a) for a in sys.argv[2:]]
if len(paths) == 1 and paths[0].is_dir():
    paths = sorted(paths[0].glob("*.jpg"))
for p in paths:
    dst = out_dir / (p.stem + ".json")
    if dst.exists():
        continue
    t0 = time.time()
    img = Image.open(p).convert("RGB")
    per_model = [A.run_model(k, img, CFG, device)[0] for k in ("yolov10x", "doclayout")]
    fused = A.reading_order(A.fuse(per_model, img.size, CFG["conf"], 0.55))
    rows = [dict(n=i, label=d["label"], conf=round(d["conf"], 4), box=[round(v, 1) for v in d["box"]])
            for i, d in enumerate(fused, 1)]
    dst.write_text(json.dumps(dict(image=p.name, width=img.size[0], height=img.size[1], count=len(rows), detections=rows)),
                   encoding="utf-8")
    print(f"{p.name}: {len(rows)} regions in {time.time() - t0:.1f}s", flush=True)
