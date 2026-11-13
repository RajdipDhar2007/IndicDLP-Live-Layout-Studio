"""Turns OCR'd layout regions into a news-website style HTML page (text only, images ignored)."""
from __future__ import annotations

import datetime
import re
import statistics
from html import escape

HEAD = {"headline"}
DECK = {"sub-headline", "subsub-headline"}
BYLINE = {"author", "dateline"}
BODY = {"paragraph", "quote", "sidebar", "ordered-list", "sub-ordered-list", "subsub-ordered-list", "unordered-list",
        "sub-unordered-list", "subsub-unordered-list", "section-title", "sub-section-title", "subsub-section-title",
        "jumpline", "reference", "footnote", "contact-info", "table-caption"}
SKIP_NOTE = "Images, ads and page furniture were left out."


def _clean(t: str) -> str:
    return re.sub(r"[ \t]+", " ", t or "").strip()


def _one_line(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def _geom(d):
    x1, y1, x2, y2 = d["box"]
    return x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2


def build_stories(dets: list[dict]) -> list[dict]:
    """Group text regions into stories: every headline starts one, body regions join the headline above them."""
    texty = [d for d in dets if _clean(d.get("text", ""))]
    heads = sorted((d for d in texty if d["label"] in HEAD), key=lambda d: (round(d["box"][1] / 40), d["box"][0]))
    stories = [dict(head=h, deck=[], by=[], body=[]) for h in heads]
    orphan = dict(head=None, deck=[], by=[], body=[])
    for d in texty:
        lab = d["label"]
        if lab in HEAD:
            continue
        if lab not in DECK | BYLINE | BODY:
            continue
        x1, y1, x2, y2, cx, cy = _geom(d)
        best, best_cost = None, None
        for s in stories:
            hx1, hy1, hx2, hy2, _, _ = _geom(s["head"])
            if cy < hy1:  # text above a headline cannot belong to it
                continue
            overlap = min(x2, hx2) - max(x1, hx1)
            if overlap <= 0 and not (hx1 <= cx <= hx2):
                continue
            cost = max(0.0, y1 - hy2)
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        if best is None:  # no headline directly above: take the closest one by centre distance
            for s in stories:
                hx1, hy1, hx2, hy2, hcx, hcy = _geom(s["head"])
                cost = abs(cx - hcx) + abs(cy - hcy) * 2
                if best_cost is None or cost < best_cost:
                    best, best_cost = s, cost
        target = best or orphan
        key = "deck" if lab in DECK else "by" if lab in BYLINE else "body"
        target[key].append(d)
    if orphan["body"] or orphan["deck"] or orphan["by"]:
        stories.append(orphan)

    out = []
    for s in stories:
        body = s["body"]
        if body:  # read column by column (left to right), top to bottom inside a column
            widths = [b["box"][2] - b["box"][0] for b in body]
            med = statistics.median(widths)
            cols: list[float] = []
            for b in sorted(body, key=lambda b: b["box"][0]):
                if b["box"][2] - b["box"][0] > 1.6 * med:
                    continue
                if not cols or b["box"][0] - cols[-1] > 0.5 * med:
                    cols.append(b["box"][0])

            def col_of(b):
                if b["box"][2] - b["box"][0] > 1.6 * med or not cols:
                    return -1
                return min(range(len(cols)), key=lambda i: abs(cols[i] - b["box"][0]))
            body = sorted(body, key=lambda b: (col_of(b), b["box"][1]))
        paras = []
        for b in body:
            for chunk in re.split(r"\n\s*\n", b["text"]):
                chunk = _one_line(chunk)
                if chunk:
                    paras.append((b["label"], chunk))
        head = s["head"]
        out.append(dict(
            title=_one_line(head["text"]) if head else "More from this page",
            size=(head["box"][3] - head["box"][1]) * (head["box"][2] - head["box"][0]) if head else 0,
            deck=[_one_line(d["text"]) for d in sorted(s["deck"], key=lambda d: d["box"][1])],
            by=[_one_line(d["text"]) for d in sorted(s["by"], key=lambda d: d["box"][1])],
            paras=paras))
    return [s for s in out if s["paras"] or s["deck"] or s["title"] != "More from this page"]


def _article(n: int, s: dict) -> str:
    body = []
    for lab, p in s["paras"]:
        e = escape(p)
        if lab == "quote":
            body.append(f"<blockquote>{e}</blockquote>")
        elif lab in ("section-title", "sub-section-title", "subsub-section-title"):
            body.append(f"<h3>{e}</h3>")
        elif lab in ("unordered-list", "sub-unordered-list", "subsub-unordered-list", "ordered-list", "sub-ordered-list", "subsub-ordered-list"):
            body.append(f"<p class='li'>• {e}</p>")
        else:
            body.append(f"<p>{e}</p>")
    deck = "".join(f"<p class='deck'>{escape(x)}</p>" for x in s["deck"])
    by = f"<p class='by'>{escape(' · '.join(s['by']))}</p>" if s["by"] else ""
    return (f"<article class='story' id='s{n}'><div class='sheet'><a class='close' href='#top'>✕ Close</a>"
            f"<h1>{escape(s['title'])}</h1>{deck}{by}<div class='text'>{''.join(body) or '<p><em>No body text was recognised.</em></p>'}</div>"
            f"</div></article>")


def _card(n: int, s: dict, lead: bool = False) -> str:
    first = s["paras"][0][1] if s["paras"] else (s["deck"][0] if s["deck"] else "")
    limit = 330 if lead else 190
    ex = first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "…"
    by = f"<div class='meta'>{escape(' · '.join(s['by']))}</div>" if s["by"] else ""
    words = sum(len(p.split()) for _, p in s["paras"])
    mins = f"<span>{max(1, round(words / 200))} min read</span>" if words else ""
    return (f"<a class='card{' lead' if lead else ''}' href='#s{n}'><h2>{escape(s['title'])}</h2>{by}"
            f"<p>{escape(ex)}</p>{mins}</a>")


def news_html(dets: list[dict], paper: str = "Digitised Newspaper", masthead: str = "") -> str:
    stories = build_stories(dets)
    today = datetime.date.today().strftime("%A, %d %B %Y")
    order = list(range(len(stories)))
    lead_i = max(order, key=lambda i: stories[i]["size"]) if order else None
    cards = ""
    if lead_i is not None:
        cards += _card(lead_i, stories[lead_i], True)
        cards += "".join(_card(i, stories[i]) for i in order if i != lead_i)
    articles = "".join(_article(i, s) for i, s in enumerate(stories))
    name = escape(_one_line(masthead) or paper)
    return f"""<!doctype html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600;800&family=Noto+Serif+Bengali:wght@400;700;800&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{{--bg:#f6f6f4;--card:#fff;--ink:#14161a;--mut:#667;--line:#e4e4df;--acc:#d6001c}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e0f12;--card:#17191e;--ink:#eceef2;--mut:#98a0ad;--line:#2a2d35;--acc:#ff4d5e}}}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,'Noto Sans Bengali',system-ui,sans-serif;line-height:1.6}}
header.top{{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:3px solid var(--acc)}}
.bar{{max-width:1180px;margin:auto;padding:12px 20px;display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap}}
.logo{{font:800 clamp(1.4rem,3vw,2rem)/1.1 'Noto Serif Bengali',Georgia,serif;margin:0;letter-spacing:-.01em}}
.logo b{{color:var(--acc)}}.date{{color:var(--mut);font-size:.85rem}}
main{{max-width:1180px;margin:auto;padding:24px 20px 60px}}
.label{{font:700 .75rem Inter,sans-serif;text-transform:uppercase;letter-spacing:.12em;color:var(--acc);margin:0 0 14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px}}
.card{{display:flex;flex-direction:column;gap:6px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;color:inherit;text-decoration:none;transition:transform .15s,box-shadow .15s}}
.card:hover{{transform:translateY(-3px);box-shadow:0 10px 28px rgba(0,0,0,.12)}}
.card h2{{font:700 1.2rem/1.35 'Noto Serif Bengali',Georgia,serif;margin:0}}
.card p{{margin:0;color:var(--mut);font-size:.95rem}}.card span{{margin-top:auto;padding-top:8px;font-size:.78rem;color:var(--acc);font-weight:600}}
.meta{{font-size:.78rem;color:var(--mut);font-weight:600}}
.card.lead{{grid-column:1/-1;border-top:4px solid var(--acc);padding:28px}}
.card.lead h2{{font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.25}}.card.lead p{{font-size:1.1rem;max-width:80ch}}
.story{{display:none;position:fixed;inset:0;z-index:20;background:rgba(0,0,0,.6);overflow:auto;padding:4vh 12px}}
.story:target{{display:block}}
.sheet{{max-width:760px;margin:auto;background:var(--card);border-radius:14px;padding:clamp(20px,5vw,48px)}}
.close{{display:inline-block;margin-bottom:18px;color:var(--acc);font-weight:700;font-size:.85rem;text-decoration:none}}
.sheet h1{{font:800 clamp(1.7rem,4vw,2.4rem)/1.25 'Noto Serif Bengali',Georgia,serif;margin:0 0 10px}}
.deck{{font-size:1.15rem;color:var(--mut);margin:0 0 10px;font-weight:600}}.by{{font-size:.85rem;color:var(--acc);font-weight:700;margin:0 0 18px}}
.text p{{font:400 1.08rem/1.9 'Noto Serif Bengali',Georgia,serif;margin:0 0 1.1em}}
.text h3{{font:700 1.2rem 'Noto Serif Bengali',serif;margin:1.4em 0 .5em}}
blockquote{{margin:1.2em 0;padding:4px 0 4px 18px;border-left:4px solid var(--acc);font:italic 1.15rem/1.7 'Noto Serif Bengali',serif}}
footer{{max-width:1180px;margin:auto;padding:0 20px 40px;color:var(--mut);font-size:.8rem}}
</style></head><body id="top">
<header class="top"><div class="bar"><h1 class="logo">{name}</h1><span class="date">{today} · {len(stories)} stories</span></div></header>
<main><p class="label">Top stories</p><div class="grid">{cards or '<p>No stories were recognised on this page.</p>'}</div></main>
{articles}
<footer>Digitised automatically from a scanned page with IndicDLP layout detection and OCR. {SKIP_NOTE} Text may contain recognition errors.</footer>
</body></html>"""
