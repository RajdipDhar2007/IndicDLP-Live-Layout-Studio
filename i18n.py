"""Localised names for the 42 IndicDLP region classes (English + Bengali).

Translations are written for UI display only; they are concise, conventional document-layout
terms and should be reviewed by a native speaker before any formal use.
"""
from __future__ import annotations

LANGS = {"en": "English", "bn": "বাংলা (Bengali)"}

# base term -> Bengali
_BASE_BN = {
    "advertisement": "বিজ্ঞাপন",
    "answer": "উত্তর",
    "author": "লেখক",
    "chapter-title": "অধ্যায়ের শিরোনাম",
    "contact-info": "যোগাযোগের তথ্য",
    "dateline": "তারিখ-স্থান রেখা",
    "figure": "চিত্র",
    "caption": "ক্যাপশন",
    "first-level-question": "প্রথম স্তরের প্রশ্ন",
    "second-level-question": "দ্বিতীয় স্তরের প্রশ্ন",
    "third-level-question": "তৃতীয় স্তরের প্রশ্ন",
    "flag": "মাস্টহেড",
    "folio": "ফোলিও",
    "footer": "ফুটার",
    "footnote": "পাদটীকা",
    "formula": "সূত্র",
    "header": "হেডার",
    "headline": "প্রধান শিরোনাম",
    "index": "নির্ঘণ্ট",
    "jumpline": "পরবর্তী পৃষ্ঠার ইঙ্গিত",
    "options": "বিকল্প",
    "ordered-list": "ক্রমিক তালিকা",
    "unordered-list": "অক্রমিক তালিকা",
    "page-number": "পৃষ্ঠা নম্বর",
    "paragraph": "অনুচ্ছেদ",
    "placeholder-text": "স্থানধারক লেখা",
    "quote": "উদ্ধৃতি",
    "reference": "তথ্যসূত্র",
    "section-title": "বিভাগের শিরোনাম",
    "sidebar": "সাইডবার",
    "table": "সারণি",
    "table-of-contents": "সূচিপত্র",
    "website-link": "ওয়েবসাইট লিংক",
}

# "sub-" and "sub-sub-" prefixes in Bengali
_SUB_BN = ("উপ-", "উপ-উপ-")

TRANSLATIONS: dict[str, dict[str, str]] = {lang: {} for lang in LANGS}


def _build() -> None:
    for lang in LANGS:
        t = TRANSLATIONS[lang]
        for base in _BASE_BN:
            if base == "caption":
                continue
            t[base] = base.replace("-", " ") if lang == "en" else _BASE_BN[base]
        caption = "caption" if lang == "en" else _BASE_BN["caption"]
        for derived, core in {"figure-caption": "figure", "table-caption": "table"}.items():
            t[derived] = f"{t[core]} {caption}"
        for level, idx in (("sub", 0), ("subsub", 1)):
            for core in ("headline", "section-title", "ordered-list", "unordered-list"):
                name = f"{level}-{core}"
                if lang == "en":
                    t[name] = ("sub-" if level == "sub" else "sub-sub-") + core.replace("-", " ")
                else:
                    t[name] = _SUB_BN[idx] + t[core]


_build()


def label(lang: str, cls_name: str) -> str:
    """Localised label for an IndicDLP class name (falls back to the English name)."""
    return TRANSLATIONS.get(lang, {}).get(cls_name, cls_name)
