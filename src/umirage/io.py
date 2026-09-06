from __future__ import annotations
import hashlib, json
from pathlib import Path
from .schema import Episode

def load_jsonl(path: str | Path) -> list[Episode]:
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x=json.loads(line)
            x["images"]=tuple(x["images"])
            rows.append(Episode(**x))
    return rows

def dump_jsonl(path: str | Path, rows: list[dict]) -> None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True)+"\n")

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()
