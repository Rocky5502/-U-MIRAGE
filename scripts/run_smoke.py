from __future__ import annotations
import csv, json
from pathlib import Path
from umirage.io import load_jsonl, dump_jsonl
from umirage.adapters.mock import MockMultiImageAdapter
from umirage.runner import run_episodes
from umirage.evaluation import accuracy,brier,source_accuracy,risk_coverage

def main():
    src=Path("data/examples/mirage_demo.jsonl")
    if not src.exists():
        from make_demo_data import main as make_demo; make_demo()
    rows=run_episodes(load_jsonl(src),MockMultiImageAdapter())
    out=Path("artifacts/smoke"); out.mkdir(parents=True,exist_ok=True)
    dump_jsonl(out/"predictions.jsonl",rows)
    metrics={"synthetic_only":True,"accuracy":accuracy(rows),"brier":brier(rows),"source_accuracy":source_accuracy(rows)}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with (out/"risk_coverage.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["coverage","selective_risk","n"]); w.writeheader(); w.writerows(risk_coverage(rows))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
