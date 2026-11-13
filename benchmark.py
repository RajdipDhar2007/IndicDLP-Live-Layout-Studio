"""Benchmark of the models used by IndicDLP Live Layout Studio.

Run:  python benchmark.py            -> writes benchmark_results.json and prints Markdown tables.

Measured on the sample pages in docs/images (no ground truth is available for them, so accuracy
is NOT measured here; published mAP is quoted from the IndicDLP paper). Measured: model size,
parameters, latency, detections, confidence, and agreement between models.
"""
import json, statistics as st, time, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
IMGS = sorted(p for p in (ROOT / "docs/images").glob("*") if p.suffix.lower() in {".png", ".jpg"} and "output" not in p.name
              and p.name != "IndicDLP_instances.png")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONF, IOU, SIZE, REPS = 0.25, 0.70, 1024, 3
SPECS = {"YOLOv10x": ("md2_PT_indicdlp_FT.pt", "yolo"), "DocLayout-YOLO": ("doclayout_yolo_indicdlp.pt", "doclayout")}


def load(kind, f):
    if kind == "yolo":
        from ultralytics import YOLO
        return YOLO(str(ROOT / "weights" / f))
    from doclayout_yolo import YOLOv10
    return YOLOv10(str(ROOT / "weights" / f))


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def predict(m, img, size):
    bgr = np.ascontiguousarray(np.array(img.convert("RGB"))[:, :, ::-1])
    r = list(m.predict(bgr, imgsz=size, conf=CONF, iou=IOU, device=0 if DEV == "cuda" else "cpu", verbose=False))[0]
    return [dict(box=tuple(b), conf=c, cls=int(k)) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


def wbf(lists, size, thr):
    from ensemble_boxes import weighted_boxes_fusion
    w, h = size
    B, S, L = [], [], []
    for d in lists:
        B.append([[min(max(v / s, 0), 1) for v, s in zip(x["box"], (w, h, w, h))] for x in d] or [[0, 0, 0, 0]])
        S.append([x["conf"] for x in d] or [0.0]); L.append([x["cls"] for x in d] or [0])
    fb, fs, fl = weighted_boxes_fusion(B, S, L, iou_thr=thr, skip_box_thr=CONF, conf_type="max")
    return [dict(box=(a * w, b * h, c * w, d * h), conf=float(s), cls=int(k)) for (a, b, c, d), s, k in zip(fb, fs, fl)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


def agreement(A, B, thr=0.5):
    """Fraction of boxes in A that have a same-class box in B with IoU >= thr (greedy)."""
    used, hit = set(), 0
    for a in A:
        best, bj = 0, None
        for j, b in enumerate(B):
            if j in used or a["cls"] != b["cls"]:
                continue
            v = iou(a["box"], b["box"])
            if v > best:
                best, bj = v, j
        if best >= thr:
            hit += 1; used.add(bj)
    return hit / len(A) if A else 0


def timed(fn):
    fn(); sync()  # warm-up
    ts = []
    for _ in range(REPS):
        sync(); t = time.perf_counter(); out = fn(); sync(); ts.append(time.perf_counter() - t)
    return out, st.mean(ts)


def main():
    imgs = [(p.name, Image.open(p).convert("RGB")) for p in IMGS]
    res = {"device": DEV, "images": [n for n, _ in imgs], "conf": CONF, "iou": IOU, "imgsz": SIZE, "reps": REPS, "models": {}}
    single, boosted = {}, {}
    for name, (f, kind) in SPECS.items():
        m = load(kind, f)
        params = sum(p.numel() for p in m.model.parameters()) / 1e6
        size_mb = (ROOT / "weights" / f).stat().st_size / 1e6
        row = dict(params_m=round(params, 1), size_mb=round(size_mb, 1), t1=[], tb=[], n=[], conf=[], ncls=[])
        for n, im in imgs:
            d, t = timed(lambda: predict(m, im, SIZE))
            single.setdefault(n, {})[name] = d
            row["t1"].append(t); row["n"].append(len(d)); row["conf"].append(st.mean([x["conf"] for x in d]) if d else 0)
            row["ncls"].append(len({x["cls"] for x in d}))
            def two():
                return wbf([predict(m, im, SIZE), predict(m, im, 1280)], im.size, 0.6)
            d2, t2 = timed(two)
            boosted.setdefault(n, {})[name] = d2; row["tb"].append(t2)
        res["models"][name] = row
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    # ensemble = both models, each boosted, fused (as in the app)
    ens = dict(t=[], n=[], conf=[], ncls=[])
    for n, im in imgs:
        t = res["models"]["YOLOv10x"]["tb"][len(ens["t"])] + res["models"]["DocLayout-YOLO"]["tb"][len(ens["t"])]
        t0 = time.perf_counter()
        d = wbf([boosted[n]["YOLOv10x"], boosted[n]["DocLayout-YOLO"]], im.size, 0.55)
        t += time.perf_counter() - t0
        boosted[n]["Ensemble"] = d
        ens["t"].append(t); ens["n"].append(len(d)); ens["conf"].append(st.mean([x["conf"] for x in d]) if d else 0)
        ens["ncls"].append(len({x["cls"] for x in d}))
    res["models"]["Ensemble Fusion"] = ens
    res["agreement"] = {
        "yolo_in_doclayout": st.mean(agreement(single[n]["YOLOv10x"], single[n]["DocLayout-YOLO"]) for n in single),
        "doclayout_in_yolo": st.mean(agreement(single[n]["DocLayout-YOLO"], single[n]["YOLOv10x"]) for n in single),
        "yolo_in_ens": st.mean(agreement(boosted[n]["YOLOv10x"], boosted[n]["Ensemble"]) for n in single),
        "doclayout_in_ens": st.mean(agreement(boosted[n]["DocLayout-YOLO"], boosted[n]["Ensemble"]) for n in single),
    }
    # OCR (EasyOCR) speed on the ensemble regions
    try:
        import easyocr
        rd = easyocr.Reader(["bn", "en"], gpu=DEV == "cuda", verbose=False)
        ts, cnt = [], 0
        for n, im in imgs[:3]:
            crops = [np.array(im.crop(tuple(int(v) for v in d["box"]))) for d in boosted[n]["Ensemble"]
                     if d["box"][2] - d["box"][0] > 8 and d["box"][3] - d["box"][1] > 8]
            t = time.perf_counter()
            for c in crops:
                rd.readtext(c, detail=0, paragraph=True)
            ts.append(time.perf_counter() - t); cnt += len(crops)
        res["easyocr"] = dict(pages=len(ts), regions=cnt, sec_per_page=st.mean(ts), ms_per_region=1000 * sum(ts) / max(cnt, 1))
    except Exception as e:
        res["easyocr"] = {"error": f"{type(e).__name__}: {e}"}
    (ROOT / "benchmark_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
