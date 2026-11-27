"""Reading order and order-aware text assembly for detected layout regions.

Replaces the old "sort by 40 px band, then left edge" heuristic with a recursive XY-cut: the page is split at
the widest empty horizontal band (stacked sections), else at the widest empty vertical band (columns), and each
part is ordered the same way. A full-width headline is therefore read before the columns under it, and each column
is read top to bottom before the next one.

This is a deterministic geometric method. It is NOT a learned reading-order model.
"""
from __future__ import annotations

# A cut must be at least this fraction of the page side, so ragged box edges do not create false cuts.
MIN_GAP_Y = 0.004
MIN_GAP_X = 0.006


def _split(items: list[dict], axis: int, min_gap: float) -> tuple[list[list[dict]], float]:
    """Group boxes into bands along `axis` (0 = x, 1 = y). Bands are separated by empty space >= min_gap.
    Returns the bands and the widest gap between them."""
    lo, hi = axis, axis + 2
    ordered = sorted(items, key=lambda d: d["box"][lo])
    groups: list[list[dict]] = [[ordered[0]]]
    reach = ordered[0]["box"][hi]
    widest = 0.0
    for d in ordered[1:]:
        gap = d["box"][lo] - reach
        if gap >= min_gap:
            groups.append([d])
            widest = max(widest, gap)
        else:
            groups[-1].append(d)
        reach = max(reach, d["box"][hi])
    return groups, widest


def _xy_cut(items: list[dict], w: float, h: float) -> list[dict]:
    if len(items) <= 1:
        return list(items)
    rows, gap_y = _split(items, 1, MIN_GAP_Y * h)
    cols, gap_x = _split(items, 0, MIN_GAP_X * w)
    # Cut where the empty band is widest relative to the page: aligned row gaps inside two columns are narrower than
    # the gutter between them, so the columns are separated first and each is then read top to bottom.
    candidates = [(gap_y / h, 1, rows)] if len(rows) > 1 else []
    candidates += [(gap_x / w, 0, cols)] if len(cols) > 1 else []
    if candidates:
        groups = max(candidates, key=lambda c: (c[0], c[1]))[2]
        out: list[dict] = []
        for g in groups:
            out.extend(_xy_cut(g, w, h))
        return out
    # No clean cut (overlapping or interleaved boxes): top to bottom, then left to right.
    return sorted(items, key=lambda d: (d["box"][1], d["box"][0]))


def reading_order(dets: list[dict], size: tuple[int, int] | None = None) -> list[dict]:
    """Detections in reading order. `size` is (width, height) of the page; it is estimated from the boxes if absent."""
    if len(dets) <= 1:
        return list(dets)
    if size is None:
        size = (max(d["box"][2] for d in dets), max(d["box"][3] for d in dets))
    return _xy_cut(list(dets), max(float(size[0]), 1.0), max(float(size[1]), 1.0))


def assemble_text(dets: list[dict], size: tuple[int, int] | None = None, sep: str = "\n\n", translation: bool = True) -> str:
    """The OCR text of all regions joined in reading order. Regions without text are skipped. When a region has an
    English translation it is written on the lines right below its text (unless `translation` is False)."""
    blocks = []
    for d in reading_order(dets, size):
        text = (d.get("text") or "").strip()
        if not text:
            continue
        english = (d.get("translation") or "").strip() if translation else ""
        blocks.append(f"{text}\n{english}" if english else text)
    return sep.join(blocks)
