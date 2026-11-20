"""Multi-page PDF input: renders each page of a PDF to a PNG so it can be handled like any uploaded image."""
from __future__ import annotations

from pathlib import Path

MAX_PAGES = 200  # safety limit for one PDF


def pdf_to_pngs(data: bytes, stem: str, dpi: int = 150) -> list[tuple[str, bytes]]:
    """Returns [(f"{stem}_p001.png", png_bytes), ...], one entry per page (at most MAX_PAGES)."""
    try:
        import pymupdf  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PDF input needs PyMuPDF: pip install pymupdf") from exc
    pages: list[tuple[str, bytes]] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        total = min(doc.page_count, MAX_PAGES)
        digits = max(3, len(str(total)))
        for i in range(total):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            pages.append((f"{Path(stem).stem}_p{i + 1:0{digits}d}.png", pix.tobytes("png")))
    return pages


def page_count(data: bytes) -> int:
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc.page_count
