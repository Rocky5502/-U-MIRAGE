from __future__ import annotations
import json, hashlib
from pathlib import Path
from .core import Episode

REGISTRY={
 "mioh":{"role":"primary multi-image hallucination","backend":"external"},
 "vischainbench":{"role":"progressive multi-image reasoning","hf_id":"eyehole/VisChainBench"},
 "mihbench":{"role":"controlled multi-image diagnostic","hf_id":"MM-Hallu/MIHBench"},
 "rest_plus":{"role":"cross-modal inconsistency","backend":"external"},
 "hallusionbench":{"role":"generalization","backend":"external"},
 "mmmu":{"role":"utility/generalization","backend":"external"},
 "vlm_uqbench":{"role":"uncertainty stress test","backend":"external"},
}

def load_jsonl(path):
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            x=json.loads(line); rows.append(Episode(x["id"],tuple(x["images"]),x["question"],x["answer"],x.get("source_label","clean"),x.get("metadata",{})))
    return rows

def dump_jsonl(path, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows),encoding="utf-8")

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
