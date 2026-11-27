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


# ----------------------------------------------------------------------------
# Interface text (English -> Bengali). The English text is the key, so an entry that is
# missing simply shows in English. `{name}` slots are filled by ui(). Product and format names
# (YOLOv10x, DocLayout-YOLO, Gemini, EasyOCR, JSON, COCO, PNG, mAP ...) stay in English.
# Like the class names above, these should be read by a native speaker before formal use.
# ----------------------------------------------------------------------------
UI_BN: dict[str, str] = {
    # hero and model cards
    "Upload pages in the file manager and watch YOLOv10x and DocLayout-YOLO read each layout, region by region, in English and Bengali.":
        "ফাইল ম্যানেজারে পৃষ্ঠা আপলোড করুন এবং দেখুন YOLOv10x ও DocLayout-YOLO ইংরেজি ও বাংলায় প্রতিটি লেআউট অঞ্চল ধরে ধরে পড়ছে।",
    "Best single-model accuracy, pretrained on UED-mini then fine-tuned on IndicDLP.":
        "একক মডেলের মধ্যে সর্বোচ্চ নির্ভুলতা; প্রথমে UED-mini-তে প্রি-ট্রেন করে পরে IndicDLP-তে ফাইন-টিউন করা।",
    "Document-specialised YOLOv10 variant with global-to-local perception.":
        "নথির জন্য বিশেষায়িত YOLOv10 রূপ, যা সামগ্রিক থেকে স্থানীয় স্তরে পৃষ্ঠা বিশ্লেষণ করে।",
    "All models voting together: boxes from every model merged with weighted box fusion.":
        "সব মডেল একসঙ্গে ভোট দেয়: প্রতিটি মডেলের বাক্স ওজনযুক্ত বক্স ফিউশনে একত্র করা হয়।",
    "loading…": "লোড হচ্ছে…",
    "detecting": "শনাক্ত হচ্ছে",
    "failed": "ব্যর্থ",
    "done · {n} regions": "সম্পন্ন · {n}টি অঞ্চল",
    "{title} · ANALYSING": "{title} · বিশ্লেষণ চলছে",
    "SCANNING DOCUMENT": "নথি স্ক্যান হচ্ছে",
    "{title} failed: {err}": "{title} ব্যর্থ হয়েছে: {err}",
    "No regions above the confidence threshold. Try lowering it.":
        "আস্থার সীমার ওপরে কোনো অঞ্চল পাওয়া যায়নি। সীমাটি কমিয়ে দেখুন।",
    "Region {i} of {n}: {name}": "অঞ্চল {i}/{n}: {name}",
    "Complete ✓": "সম্পূর্ণ ✓",
    # stat tiles
    "Regions": "অঞ্চল", "Classes": "শ্রেণি", "Avg conf": "গড় আস্থা", "Inference": "অনুমান-সময়", "Device": "ডিভাইস",
    # sidebar
    "### 🎛️ Control room": "### 🎛️ নিয়ন্ত্রণ কক্ষ",
    "Language of the whole interface, and of the region names in the feed, chips, legend, tables and the final annotated page.":
        "পুরো ইন্টারফেসের ভাষা, এবং ফিড, চিপ, নির্দেশিকা, সারণি ও চূড়ান্ত চিহ্নিত পৃষ্ঠার অঞ্চলের নামের ভাষা।",
    "Weights not downloaded.": "ওয়েটস ডাউনলোড করা হয়নি।",
    "Weights incomplete (download still running or interrupted).": "ওয়েটস অসম্পূর্ণ (ডাউনলোড চলছে বা মাঝপথে থেমে গেছে)।",
    "⬇ Download {title} weights": "⬇ {title} ওয়েটস ডাউনলোড করুন",
    "Downloading {file}…": "{file} ডাউনলোড হচ্ছে…",
    "Downloading {file} · {done:.0f}/{total:.0f} MB": "{file} ডাউনলোড হচ্ছে · {done:.0f}/{total:.0f} MB",
    "##### 🌐 Regions and exports": "##### 🌐 অঞ্চল ও এক্সপোর্ট",
    "Leave out images and ads in exports": "এক্সপোর্টে ছবি ও বিজ্ঞাপন বাদ দিন",
    "JSON, COCO, PAGE-XML, HTML, text, cutouts and batch ZIPs skip figure and advertisement regions. The annotated PNG and the on-screen boxes still show them.":
        "JSON, COCO, PAGE-XML, HTML, টেক্সট, কাটআউট ও ব্যাচ ZIP-এ চিত্র ও বিজ্ঞাপন অঞ্চল বাদ যায়। চিহ্নিত PNG ও পর্দার বাক্সগুলোতে সেগুলো তবুও দেখা যায়।",
    "Simplified classes": "সরলীকৃত শ্রেণি",
    "Folds the 42 classes into 6 families: headline, subtitle, paragraph, image, caption, advertisement.":
        "৪২টি শ্রেণিকে ৬টি পরিবারে একত্র করে: শিরোনাম, উপশিরোনাম, অনুচ্ছেদ, ছবি, ক্যাপশন, বিজ্ঞাপন।",
    "##### 🎯 Accuracy": "##### 🎯 নির্ভুলতা",
    "Multi-scale boost": "মাল্টি-স্কেল বুস্ট",
    "Runs every model at two image sizes and fuses the boxes, recovering small or faint regions.":
        "প্রতিটি মডেল দুটি ছবির আকারে চালিয়ে বাক্সগুলো একত্র করে, ফলে ছোট বা অস্পষ্ট অঞ্চলও ধরা পড়ে।",
    "Ensemble fusion": "এনসেম্বল ফিউশন",
    "Merges the boxes of all enabled models into one more complete result.":
        "চালু থাকা সব মডেলের বাক্স মিলিয়ে একটি পূর্ণাঙ্গ ফলাফল তৈরি করে।",
    "Confidence threshold": "আস্থার সীমা",
    "Inference size": "অনুমানের আকার",
    "Reveal speed": "প্রদর্শনের গতি",
    "Cinematic": "সিনেমাটিক", "Normal": "স্বাভাবিক", "Fast": "দ্রুত", "Instant": "তাৎক্ষণিক",
    "##### 📝 Text extraction (OCR)": "##### 📝 লেখা পড়া (OCR)",
    "Enable OCR": "OCR চালু করুন",
    "Off: only the layout annotations are produced, no text is read and no Gemini requests are made.":
        "বন্ধ থাকলে: শুধু লেআউট চিহ্নিতকরণ হবে, কোনো লেখা পড়া হবে না এবং Gemini-তে কোনো অনুরোধ যাবে না।",
    "Engine": "ইঞ্জিন",
    "Gemini: cloud, most accurate. EasyOCR: local, works offline.": "Gemini: ক্লাউডে চলে, সবচেয়ে নির্ভুল। EasyOCR: লোকাল, অফলাইনে চলে।",
    "Gemini API key": "Gemini API কী",
    "or set GEMINI_API_KEY": "অথবা GEMINI_API_KEY সেট করুন",
    "Kept only in this browser session. Leave empty if GEMINI_API_KEY is set.":
        "শুধু এই ব্রাউজার সেশনে থাকে। GEMINI_API_KEY সেট করা থাকলে ফাঁকা রাখুন।",
    "Gemini model for per-region OCR": "অঞ্চলভিত্তিক OCR-এর জন্য Gemini মডেল",
    "Used when you click a region (or press Transcribe All). Each region is sent only once.":
        "কোনো অঞ্চলে ক্লিক করলে (বা “সব অঞ্চল লিখে নিন” চাপলে) ব্যবহৃত হয়। প্রতিটি অঞ্চল মাত্র একবার পাঠানো হয়।",
    "Gemini sends each text region's image to Google. Use EasyOCR to stay offline.":
        "Gemini প্রতিটি লেখার অঞ্চলের ছবি Google-এ পাঠায়। অফলাইনে থাকতে EasyOCR ব্যবহার করুন।",
    "Device: **{device}**": "ডিভাইস: **{device}**",
    "🔄 Reset to defaults": "🔄 ডিফল্টে ফিরুন",
    # source / workspace
    "File manager": "ফাইল ম্যানেজার", "Sample gallery": "নমুনা গ্যালারি",
    "Sample page": "নমুনা পৃষ্ঠা",
    "That file couldn't be read as an image.": "ফাইলটি ছবি হিসেবে পড়া যায়নি।",
    "⬆️ Upload one or more pages in the File manager on the right.": "⬆️ ডানদিকের ফাইল ম্যানেজারে এক বা একাধিক পৃষ্ঠা আপলোড করুন।",
    "No sample images found.": "কোনো নমুনা ছবি পাওয়া যায়নি।",
    "Enable at least one model in the sidebar.": "সাইডবারে অন্তত একটি মডেল চালু করুন।",
    "#### Ready to analyse": "#### বিশ্লেষণের জন্য প্রস্তুত",
    "**Models:** {models}": "**মডেল:** {models}",
    "**Confidence ≥** {conf:.2f} · **IoU** {iou:.2f} · **Size** {size}px": "**আস্থা ≥** {conf:.2f} · **IoU** {iou:.2f} · **আকার** {size}px",
    " · **multi-scale**": " · **মাল্টি-স্কেল**",
    " · **6 simplified classes**": " · **৬টি সরলীকৃত শ্রেণি**",
    "🚀 Scan & detect": "🚀 স্ক্যান ও শনাক্ত করুন",
    # file manager
    "### 🗂️ File manager": "### 🗂️ ফাইল ম্যানেজার",
    "Add batches; open a page to work on it.": "একসঙ্গে অনেক ফাইল যোগ করুন; কাজ করতে একটি পৃষ্ঠা খুলুন।",
    "Add image or PDF files": "ছবি বা PDF ফাইল যোগ করুন",
    "{name}: could not be read as a PDF ({err})": "{name}: PDF হিসেবে পড়া যায়নি ({err})",
    "No files uploaded yet.": "এখনও কোনো ফাইল আপলোড করা হয়নি।",
    "No preview": "প্রিভিউ নেই",
    "Open": "খুলুন",
    "Done": "সম্পন্ন",
    "Mark this page finished": "এই পৃষ্ঠাটি শেষ বলে চিহ্নিত করুন",
    # batch
    "📚 Batch: process all {n} pages at once": "📚 ব্যাচ: সব {n}টি পৃষ্ঠা একসঙ্গে প্রক্রিয়া করুন",
    "Runs detection on every page in the file manager with the current sidebar settings (no animation) and builds one ZIP: per-page JSON, PAGE-XML and text, plus COCO JSON and HTML for the whole batch. Uses Ensemble Fusion when two models are on, otherwise the single model.":
        "সাইডবারের বর্তমান সেটিং অনুযায়ী ফাইল ম্যানেজারের প্রতিটি পৃষ্ঠায় শনাক্তকরণ চালায় (অ্যানিমেশন ছাড়া) এবং একটি ZIP তৈরি করে: প্রতি পৃষ্ঠার JSON, PAGE-XML ও টেক্সট, এবং পুরো ব্যাচের COCO JSON ও HTML। দুটি মডেল চালু থাকলে Ensemble Fusion, নইলে একক মডেল ব্যবহার হয়।",
    "Read the text of every region (OCR)": "প্রতিটি অঞ্চলের লেখা পড়ুন (OCR)",
    "Slower: each page is OCR'd with the engine chosen in the sidebar.": "ধীর: সাইডবারে বাছাই করা ইঞ্জিনে প্রতিটি পৃষ্ঠায় OCR চলে।",
    "▶ Run batch": "▶ ব্যাচ চালান",
    "Starting…": "শুরু হচ্ছে…",
    "Page {n}/{total}: {name}": "পৃষ্ঠা {n}/{total}: {name}",
    "{pages} pages, {regions} regions.": "{pages}টি পৃষ্ঠা, {regions}টি অঞ্চল।",
    "⬇ Download batch ZIP": "⬇ ব্যাচ ZIP ডাউনলোড করুন",
    # crop dialog
    "Detection crop": "শনাক্ত অঞ্চলের কাটআউট",
    "The selected detection has an invalid or empty box.": "নির্বাচিত শনাক্তকরণের বাক্সটি অবৈধ বা ফাঁকা।",
    "Close": "বন্ধ করুন",
    "{model} | Detection #{number} | {cls} | confidence {conf:.3f}": "{model} | শনাক্তকরণ #{number} | {cls} | আস্থা {conf:.3f}",
    "Original crop": "মূল কাটআউট",
    "Upscale crop with ESPCN x2": "ESPCN x2 দিয়ে বড় করুন",
    "Upscaling crop": "কাটআউট বড় করা হচ্ছে",
    "Could not upscale this crop: {err}": "এই কাটআউট বড় করা যায়নি: {err}",
    "Download cutout (PNG)": "কাটআউট ডাউনলোড করুন (PNG)",
    "OCR is switched off in the sidebar.": "সাইডবারে OCR বন্ধ করা আছে।",
    "OCR is skipped for image and advertisement regions.": "ছবি ও বিজ্ঞাপন অঞ্চলে OCR করা হয় না।",
    "Selected model: {model}. Change it in the sidebar, then run OCR when ready.":
        "নির্বাচিত মডেল: {model}। সাইডবারে বদলানো যাবে; প্রস্তুত হলে OCR চালান।",
    "Run Gemini OCR": "Gemini OCR চালান",
    "Waiting for {model}": "{model}-এর উত্তরের অপেক্ষা",
    "This region was already processed with {model}; it will not be sent again.":
        "এই অঞ্চলটি ইতিমধ্যে {model} দিয়ে প্রক্রিয়া করা হয়েছে; আবার পাঠানো হবে না।",
    "OCR transcription (editable)": "OCR লিখিত রূপ (সম্পাদনাযোগ্য)",
    "Model used: {model}. This region is sent to Gemini once per detection.":
        "ব্যবহৃত মডেল: {model}। প্রতিটি শনাক্তকরণের অঞ্চল Gemini-তে একবারই পাঠানো হয়।",
    "Text read by the page-level OCR": "পৃষ্ঠা-স্তরের OCR যে লেখা পড়েছে",
    "Processed {pos} of {n} text regions": "{n}টি লেখার অঞ্চলের মধ্যে {pos}টি প্রক্রিয়া হয়েছে",
    # results tables
    "## 📊 Results": "## 📊 ফলাফল",
    "Model": "মডেল", "Time (ms)": "সময় (ms)",
    "Region": "অঞ্চল", "Confidence": "আস্থা", "Width": "প্রস্থ", "Height": "উচ্চতা", "Text": "লেখা",
    "#### Model benchmark": "#### মডেল বেঞ্চমার্ক",
    "Published mAP (IndicDLP paper) and measured speed on the sample pages from `benchmark.py` (median over pages, 1024 px, confidence 0.25).":
        "প্রকাশিত mAP (IndicDLP গবেষণাপত্র) এবং `benchmark.py`-এর নমুনা পৃষ্ঠায় মাপা গতি (পৃষ্ঠাগুলোর মধ্যমা, ১০২৪ px, আস্থা ০.২৫)।",
    "mAP (published)": "mAP (প্রকাশিত)",
    "Params (M)": "প্যারামিটার (M)",
    "Weights (MB)": "ওয়েটস (MB)",
    "Latency / page (ms)": "প্রতি পৃষ্ঠায় সময় (ms)",
    "With multi-scale (ms)": "মাল্টি-স্কেল সহ (ms)",
    "Avg regions / page": "গড় অঞ্চল / পৃষ্ঠা",
    "Avg confidence": "গড় আস্থা",
    "#### OCR benchmark": "#### OCR বেঞ্চমার্ক",
    "EasyOCR figures are from `benchmark.py`; Gemini rows fill in as you run OCR on regions in this session.":
        "EasyOCR-এর সংখ্যা `benchmark.py` থেকে; এই সেশনে অঞ্চলে OCR চালালে Gemini-র সারিগুলো যুক্ত হবে।",
    "No OCR timings yet. Run Gemini OCR on a region to see its speed here.":
        "এখনও কোনো OCR সময় নেই। গতি দেখতে কোনো অঞ্চলে Gemini OCR চালান।",
    "Regions read": "পড়া অঞ্চল",
    "Avg time / region (ms)": "গড় সময় / অঞ্চল (ms)",
    "Avg time / page (s)": "গড় সময় / পৃষ্ঠা (s)",
    "Errors": "ত্রুটি",
    "Source": "উৎস",
    "EasyOCR (local, Bengali + English)": "EasyOCR (লোকাল, বাংলা + ইংরেজি)",
    "benchmark.py, {n} sample pages": "benchmark.py, {n}টি নমুনা পৃষ্ঠা",
    "Gemini {model} (cloud, one request per region)": "Gemini {model} (ক্লাউড, প্রতি অঞ্চলে একটি অনুরোধ)",
    "measured in this session": "এই সেশনে মাপা",
    "Regions per class": "শ্রেণি অনুযায়ী অঞ্চল",
    # transcription / exports
    "### 🧾 Gemini transcription": "### 🧾 Gemini দিয়ে লেখা পড়া",
    "Model: {model}. Every text region is sent once; images and advertisements are skipped.":
        "মডেল: {model}। প্রতিটি লেখার অঞ্চল একবার পাঠানো হয়; ছবি ও বিজ্ঞাপন বাদ যায়।",
    "Transcribe all regions": "সব অঞ্চল লিখে নিন",
    "Preparing OCR": "OCR প্রস্তুত হচ্ছে",
    "Transcription finished: {done} completed, {errors} errors, {total} eligible regions.":
        "লেখা পড়া শেষ: {total}টি যোগ্য অঞ্চলের মধ্যে {done}টি সম্পন্ন, {errors}টি ত্রুটি।",
    "Reading text in the fused regions ({engine})…": "ফিউজ করা অঞ্চলের লেখা পড়া হচ্ছে ({engine})…",
    "Automatic OCR failed: {err}": "স্বয়ংক্রিয় OCR ব্যর্থ হয়েছে: {err}",
    "📝 Re-run with {engine} (Bengali + English)": "📝 {engine} দিয়ে আবার চালান (বাংলা + ইংরেজি)",
    "📝 Extract text with {engine} (Bengali + English)": "📝 {engine} দিয়ে লেখা বের করুন (বাংলা + ইংরেজি)",
    "Starting {engine}…": "{engine} শুরু হচ্ছে…",
    "OCR needs EasyOCR: pip install easyocr": "OCR-এর জন্য EasyOCR দরকার: pip install easyocr",
    "💡 Click any box to open its cutout: coordinates, PNG download, ESPCN x2 upscale and Gemini OCR.":
        "💡 যেকোনো বাক্সে ক্লিক করে তার কাটআউট খুলুন: স্থানাঙ্ক, PNG ডাউনলোড, ESPCN x2 বড় করা এবং Gemini OCR।",
    "💡 Click any box to open its cutout: coordinates, PNG download, ESPCN x2 upscale. OCR is switched off in the sidebar.":
        "💡 যেকোনো বাক্সে ক্লিক করে তার কাটআউট খুলুন: স্থানাঙ্ক, PNG ডাউনলোড, ESPCN x2 বড় করা। সাইডবারে OCR বন্ধ আছে।",
    "⬇ Annotated image (PNG)": "⬇ চিহ্নিত ছবি (PNG)",
    "⬇ Extracted text (TXT, reading order)": "⬇ বের করা লেখা (TXT, পড়ার ক্রমে)",
    "⬇ HTML (reading order)": "⬇ HTML (পড়ার ক্রমে)",
    "⬇ Detections (JSON)": "⬇ শনাক্তকরণ (JSON)",
    "📰 Download newspaper as HTML (news site, from Ensemble Fusion)": "📰 সংবাদপত্র HTML হিসেবে ডাউনলোড করুন (নিউজ সাইট, Ensemble Fusion থেকে)",
    "📐 Coordinates of all {n} boxes (pixels, top-left origin)": "📐 সব {n}টি বাক্সের স্থানাঙ্ক (পিক্সেল, উপরে-বামে মূলবিন্দু)",
    "### 📦 Full page cutout data": "### 📦 সম্পূর্ণ পৃষ্ঠার কাটআউট ডেটা",
    "Coordinates, source and grouped classes, confidence, a crop of every region and its Gemini OCR record.":
        "স্থানাঙ্ক, মূল ও গোষ্ঠীবদ্ধ শ্রেণি, আস্থা, প্রতিটি অঞ্চলের কাটআউট এবং তার Gemini OCR রেকর্ড।",
    "Preparing it runs OCR once for each text region not already processed.":
        "এটি প্রস্তুত করতে, যে লেখার অঞ্চলগুলোতে আগে OCR হয়নি সেগুলোতে একবার করে OCR চলবে।",
    "Prepare full JSON export": "সম্পূর্ণ JSON এক্সপোর্ট প্রস্তুত করুন",
    "Preparing OCR and cutout data": "OCR ও কাটআউট ডেটা প্রস্তুত হচ্ছে",
    "⬇ Download full JSON": "⬇ সম্পূর্ণ JSON ডাউনলোড করুন",
    # errors raised in OCR code
    "No Gemini API key. Enter it in the sidebar or set GEMINI_API_KEY.": "Gemini API কী নেই। সাইডবারে দিন অথবা GEMINI_API_KEY সেট করুন।",
    "No Gemini API key. Enter it in the sidebar, or set GEMINI_API_KEY / GOOGLE_GEMINI_API_KEY in .env.":
        "Gemini API কী নেই। সাইডবারে দিন, অথবা .env-এ GEMINI_API_KEY / GOOGLE_GEMINI_API_KEY সেট করুন।",
    "Gemini rejected the request (HTTP {code}). Check the API key.": "Gemini অনুরোধ প্রত্যাখ্যান করেছে (HTTP {code})। API কী যাচাই করুন।",
    "Gemini could not read the page ({last}). Daily free-tier limits may be used up; try EasyOCR.":
        "Gemini পৃষ্ঠাটি পড়তে পারেনি ({last})। দৈনিক বিনামূল্যের সীমা শেষ হয়ে থাকতে পারে; EasyOCR চেষ্টা করুন।",
    "Gemini OCR failed: {message}": "Gemini OCR ব্যর্থ হয়েছে: {message}",
    # legend groups
    "Headlines": "প্রধান শিরোনাম", "Titles": "শিরোনাম ও অধ্যায়", "Body text": "মূল লেখা", "Lists": "তালিকা",
    "Figures & tables": "চিত্র ও সারণি", "Page furniture": "পৃষ্ঠার কাঠামো", "Exam parts": "পরীক্ষার অংশ", "Other": "অন্যান্য",
}


def ui(lang: str, text: str, **kw) -> str:
    """Interface text in `lang` ("en" or "bn"); falls back to the English text. `{name}` slots are filled from kw."""
    out = UI_BN.get(text, text) if lang == "bn" else text
    return out.format(**kw) if kw else out
