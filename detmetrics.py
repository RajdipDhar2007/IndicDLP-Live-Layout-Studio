"""Detection metrics: per-class COCO-style average precision. Pure NumPy, no model or Streamlit needed.

Boxes are (x1, y1, x2, y2). A prediction is {"box", "conf", "cls"}; a ground-truth box is {"box", "cls"}.
`evaluate(preds, gts, n_classes)` takes one list per image (aligned by index).
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)  # COCO: 0.50, 0.55, ..., 0.95


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n, 4) and every box in b (m, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO)."""
    if len(recall) == 0:
        return 0.0
    prec = np.maximum.accumulate(precision[::-1])[::-1]  # make precision monotone non-increasing
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    return float(np.mean(np.array([prec[i] if i < len(prec) else 0.0 for i in idx], dtype=float)))


def _class_ap(preds: list[tuple[int, float, np.ndarray]], gts: dict[int, np.ndarray], n_gt: int, thr: float) -> float:
    """AP of one class at one IoU threshold. preds: (image index, conf, box); gts: image index -> (k, 4) boxes."""
    if n_gt == 0:
        return float("nan")  # class absent from the ground truth: excluded from the mean
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda p: -p[1])
    used = {i: np.zeros(len(g), bool) for i, g in gts.items()}
    tp = np.zeros(len(preds))
    for n, (img, _, box) in enumerate(preds):
        g = gts.get(img)
        if g is None or len(g) == 0:
            continue
        ious = iou_matrix(box[None, :], g)[0]
        ious[used[img]] = -1  # a ground-truth box can be matched only once
        j = int(np.argmax(ious))
        if ious[j] >= thr:
            tp[n] = 1
            used[img][j] = True
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    return average_precision(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-12))


def evaluate(preds: list[list[dict]], gts: list[list[dict]], n_classes: int) -> dict:
    """Per-class AP50 and AP50:95 plus the means over classes that occur in the ground truth."""
    ap = np.full((n_classes, len(IOU_THRESHOLDS)), np.nan)
    n_gt_per_class = np.zeros(n_classes, int)
    for c in range(n_classes):
        gt_c = {i: np.array([g["box"] for g in img if g["cls"] == c], float).reshape(-1, 4) for i, img in enumerate(gts)}
        gt_c = {i: g for i, g in gt_c.items() if len(g)}
        n_gt = sum(len(g) for g in gt_c.values())
        n_gt_per_class[c] = n_gt
        pr_c = [(i, float(p["conf"]), np.asarray(p["box"], float)) for i, img in enumerate(preds) for p in img if p["cls"] == c]
        for t, thr in enumerate(IOU_THRESHOLDS):
            ap[c, t] = _class_ap(pr_c, gt_c, n_gt, float(thr))
    present = n_gt_per_class > 0
    return {
        "ap50": ap[:, 0].tolist(),
        "ap50_95": np.nanmean(ap, axis=1).tolist() if present.any() else [float("nan")] * n_classes,
        "n_gt": n_gt_per_class.tolist(),
        "map50": float(np.nanmean(ap[present, 0])) if present.any() else 0.0,
        "map50_95": float(np.nanmean(ap[present])) if present.any() else 0.0,
    }
