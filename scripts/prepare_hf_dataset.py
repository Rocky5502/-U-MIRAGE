from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

HF_DATASETS={
    "vischainbench":"eyehole/VisChainBench",
    "mihbench":"MM-Hallu/MIHBench",
}

def main():
    p=argparse.ArgumentParser(description="Download an allowed HF benchmark and write a provenance manifest.")
    p.add_argument("dataset",choices=sorted(HF_DATASETS));p.add_argument("--revision",default=None);p.add_argument("--cache-dir",default="data/cache")
    a=p.parse_args()
    try: from datasets import load_dataset
    except ImportError as exc: raise SystemExit("Install `pip install -e .[vlm]` first") from exc
    hf_id=HF_DATASETS[a.dataset]
    ds=load_dataset(hf_id,revision=a.revision,cache_dir=a.cache_dir)
    summary={"id":a.dataset,"hf_id":hf_id,"requested_revision":a.revision,"splits":{k:len(v) for k,v in ds.items()}}
    canonical=json.dumps(summary,sort_keys=True).encode();summary["manifest_sha256"]=hashlib.sha256(canonical).hexdigest()
    out=Path("data/manifests")/f"{a.dataset}.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(out)
if __name__=="__main__":main()
