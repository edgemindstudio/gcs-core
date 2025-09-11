
from __future__ import annotations
import json
from pathlib import Path
import datetime as _dt

def now_iso():
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def write_json_line(obj, jsonl_path):
    p = Path(jsonl_path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
