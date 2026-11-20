"""Tests for reading order, exports, PDF input and AP metrics. Run: python -m pytest tests -q"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detmetrics  # noqa: E402
import exports  # noqa: E402
import layout_order  # noqa: E402

NAMES = ["headline", "paragraph", "figure"]


def det(box, label="paragraph", text="", conf=0.9):
    d = {"box": box, "conf": conf, "cls": NAMES.index(label), "label": label}
    if text:
        d["text"] = text
    return d


# --- reading order ---------------------------------------------------------
def test_two_columns_are_read_column_by_column():
    # Left column boxes sit lower than the right column's first box: the old band heuristic interleaves them.
    dets = [det((600, 110, 900, 400), text="R1"), det((600, 420, 900, 700), text="R2"),
            det((50, 130, 350, 400), text="L1"), det((50, 420, 350, 700), text="L2")]
    assert [d["text"] for d in layout_order.reading_order(dets, (1000, 1000))] == ["L1", "L2", "R1", "R2"]


def test_full_width_headline_comes_before_columns():
    dets = [det((600, 300, 950, 900), text="R"), det((50, 300, 400, 900), text="L"),
            det((50, 50, 950, 200), "headline", text="H")]
    assert [d["text"] for d in layout_order.reading_order(dets, (1000, 1000))] == ["H", "L", "R"]


def test_reading_order_handles_empty_single_and_overlap():
    assert layout_order.reading_order([]) == []
    one = [det((0, 0, 10, 10))]
    assert layout_order.reading_order(one) == one
    overlap = [det((0, 100, 500, 400), text="b"), det((0, 0, 500, 300), text="a")]
    assert [d["text"] for d in layout_order.reading_order(overlap, (500, 500))] == ["a", "b"]


def test_assemble_text_skips_empty_and_follows_order():
    dets = [det((600, 100, 900, 400), text="right"), det((50, 100, 350, 400), text="left"), det((50, 500, 350, 600))]
    assert layout_order.assemble_text(dets, (1000, 1000)) == "left\n\nright"


# --- exports ---------------------------------------------------------------
PAGE = {"name": "p1.png", "width": 1000, "height": 1000, "image": None, "dets": [
    det((50, 300, 400, 900), text="body & <text>"), det((50, 50, 950, 200), "headline", text="Title"),
    det((600, 300, 950, 600), "figure")]}


def test_coco_json_is_valid_and_ordered():
    c = json.loads(exports.coco_json([PAGE], NAMES))
    assert [x["name"] for x in c["categories"]] == NAMES and c["categories"][0]["id"] == 1
    anns = c["annotations"]
    assert len(anns) == 3 and anns[0]["category_id"] == 1  # headline first
    assert anns[0]["bbox"] == [50.0, 50.0, 900.0, 150.0] and anns[0]["area"] == 135000.0
    assert [a["reading_order"] for a in anns] == [1, 2, 3]


def test_coco_uses_source_label_when_classes_are_simplified():
    page = {**PAGE, "dets": [{**det((0, 0, 10, 10), "paragraph"), "label": "image", "source_label": "figure"}]}
    assert json.loads(exports.coco_json([page], NAMES))["annotations"][0]["category_id"] == 3


def test_page_xml_parses_and_escapes():
    root = ET.fromstring(exports.page_xml(PAGE))
    ns = {"p": exports.PAGE_NS}
    page = root.find("p:Page", ns)
    assert page is not None and page.get("imageWidth") == "1000"
    refs = root.findall(".//p:RegionRefIndexed", ns)
    assert [r.get("regionRef") for r in refs] == ["r1", "r2", "r3"]
    assert root.find(".//p:ImageRegion", ns) is not None
    texts = [u.text for u in root.findall(".//p:Unicode", ns)]
    assert "body & <text>" in texts and "Title" in texts
    region = root.find(".//p:TextRegion", ns)
    assert region is not None and region.get("type") == "heading"


def test_html_document_escapes_and_orders():
    h = exports.html_document([PAGE])
    assert "<h1" in h and "body &amp; &lt;text&gt;" in h and "<script" not in h
    assert h.index("Title") < h.index("body &amp;")
    assert "[figure]" in h  # no page image supplied, so a placeholder


# --- AP metrics ------------------------------------------------------------
def test_perfect_prediction_gives_ap_one():
    gt = [[{"box": (0, 0, 100, 100), "cls": 0}, {"box": (200, 200, 300, 300), "cls": 1}]]
    pr = [[{"box": (0, 0, 100, 100), "conf": 0.9, "cls": 0}, {"box": (200, 200, 300, 300), "conf": 0.8, "cls": 1}]]
    r = detmetrics.evaluate(pr, gt, 3)
    assert r["map50"] == pytest.approx(1.0) and r["map50_95"] == pytest.approx(1.0)
    assert r["n_gt"] == [1, 1, 0]  # class 2 is absent and is not averaged in


def test_no_and_wrong_predictions_give_zero():
    gt = [[{"box": (0, 0, 100, 100), "cls": 0}]]
    assert detmetrics.evaluate([[]], gt, 1)["map50"] == 0.0
    wrong = [[{"box": (500, 500, 600, 600), "conf": 0.9, "cls": 0}]]
    assert detmetrics.evaluate(wrong, gt, 1)["map50"] == 0.0


def test_loose_box_counts_at_iou_50_but_not_95():
    gt = [[{"box": (0, 0, 100, 100), "cls": 0}]]
    pr = [[{"box": (0, 0, 100, 70), "conf": 0.9, "cls": 0}]]  # IoU 0.7
    r = detmetrics.evaluate(pr, gt, 1)
    assert r["map50"] == pytest.approx(1.0) and 0 < r["map50_95"] < 1


def test_duplicate_prediction_is_a_false_positive():
    gt = [[{"box": (0, 0, 100, 100), "cls": 0}]]
    pr = [[{"box": (0, 0, 100, 100), "conf": 0.9, "cls": 0}, {"box": (1, 1, 100, 100), "conf": 0.5, "cls": 0}]]
    assert detmetrics.evaluate(pr, gt, 1)["map50"] == pytest.approx(1.0)  # the extra box ranks below the true positive
    pr2 = [[{"box": (1, 1, 100, 100), "conf": 0.95, "cls": 0}, {"box": (0, 0, 100, 100), "conf": 0.9, "cls": 0}]]
    assert detmetrics.evaluate(pr2, gt, 1)["map50"] == pytest.approx(1.0)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    assert detmetrics.iou_matrix(a, np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float))[0].tolist() == \
        pytest.approx([1.0, 1 / 3, 0.0])


# --- PDF input -------------------------------------------------------------
def test_pdf_pages_become_pngs():
    pymupdf = pytest.importorskip("pymupdf")
    import pdfio
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page(width=200, height=300).insert_text((20, 50), f"page {i + 1}")
    pages = pdfio.pdf_to_pngs(doc.tobytes(), "scan.pdf", dpi=72)
    assert [n for n, _ in pages] == ["scan_p001.png", "scan_p002.png", "scan_p003.png"]
    assert all(b[:8] == b"\x89PNG\r\n\x1a\n" for _, b in pages)
    assert pdfio.page_count(doc.tobytes()) == 3


# --- images and advertisements are left out of exports ------------------------
def test_without_media_drops_figures_ads_and_simplified_images():
    dets = [det((0, 0, 10, 10), "paragraph"), det((0, 0, 10, 10), "figure"),
            {**det((0, 0, 10, 10), "paragraph"), "label": "image", "source_label": "figure"},
            {**det((0, 0, 10, 10), "paragraph"), "label": "advertisement", "source_label": "advertisement"}]
    kept = exports.without_media(dets)
    assert len(kept) == 1 and kept[0]["label"] == "paragraph"
