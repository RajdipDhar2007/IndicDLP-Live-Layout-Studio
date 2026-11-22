"""Build one self-contained Label Studio import file with every annotated page (image + boxes + OCR text).

Usage: python make_labelstudio_import.py [out_dir]     (default: labelstudio_import next to this script)

Output:
    labelstudio_import/import_annotated_pages.json   tasks with the page images embedded (base64) and the annotations
    labelstudio_import/label_config.xml              labelling interface to paste into the new project
The Label Studio DB is opened read-only.
"""
import base64
import json
import mimetypes
import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LS_DIR = Path(os.path.expanduser(r"~\AppData\Local\label-studio\label-studio"))
DB = LS_DIR / "label_studio.sqlite3"
MEDIA = LS_DIR / "media"

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "labelstudio_import"
out_dir.mkdir(parents=True, exist_ok=True)

db = sqlite3.connect("file:" + str(DB).replace("\\", "/") + "?mode=ro", uri=True)
config = db.execute("select label_config from project where id = 2").fetchone()[0]
(out_dir / "label_config.xml").write_text(config, encoding="utf-8")

rows = db.execute(
    "select t.id, t.data, a.result from task t join task_completion a on a.task_id = t.id "
    "where a.was_cancelled = 0 order by t.id, a.id").fetchall()

tasks = {}
for task_id, data, result in rows:
    if task_id not in tasks:
        rel = next(iter(json.loads(data).values()))
        path = MEDIA / rel.replace("/data/", "", 1).lstrip("/")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        uri = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        tasks[task_id] = {"data": {"image": uri, "page": path.name.split("-", 1)[-1]}, "annotations": []}
    tasks[task_id]["annotations"].append({"result": json.loads(result)})
    print(task_id, tasks[task_id]["data"]["page"], flush=True)

target = out_dir / "import_annotated_pages.json"
target.write_text(json.dumps(list(tasks.values()), ensure_ascii=False), encoding="utf-8")
print(f"\n{len(tasks)} tasks -> {target} ({target.stat().st_size / 1e6:.0f} MB)")
