"""Consistent online backup of the Label Studio sqlite DB (read-only on the source)."""
import os
import sqlite3

src_path = os.path.expanduser(r"~\AppData\Local\label-studio\label-studio\label_studio.sqlite3")
src = sqlite3.connect("file:" + src_path.replace("\\", "/") + "?mode=ro", uri=True)
import datetime
dst = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup",
                                   f"label_studio_{datetime.date.today().isoformat()}_annotations.sqlite3"))
src.backup(dst)
print("annotations:", dst.execute("select count(*) from task_completion").fetchone()[0],
      "tasks:", dst.execute("select count(*) from task").fetchone()[0])
dst.close()
src.close()
