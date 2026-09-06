from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import Episode, UMiragePipeline, accuracy, brier
from .adapters import MockMultiImageAdapter
from .data import REGISTRY, dump_jsonl
from .provenance import write

def demo_episodes(n=16):
    labels=("clean","visual","integration","reasoning"); rows=[]
    for i in range(n):
        s=labels[i%4]; rows.append(Episode(f"demo-{i}",("img-a","img-b","img-c"),"Which label is supported?","alpha",s,{"synthetic":{"risk":.12 if s=="clean" else .72,"wrong_answer":"beta"}}))
    return rows

def run_smoke(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); pipe=UMiragePipeline(MockMultiImageAdapter()); rows=[]
    for ep in demo_episodes():
        o,s=pipe.analyze(ep); rows.append({"id":ep.id,"gold":ep.answer,"prediction":o.answer,"correct":o.answer==ep.answer,"source_label":ep.source_label,"predicted_source":max(s.source_posterior,key=s.source_posterior.get),**s.to_dict()})
    dump_jsonl(out/"predictions.jsonl",rows); metrics={"synthetic_only":True,"n":len(rows),"accuracy":accuracy(rows),"brier":brier(rows)}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); write(out/"provenance.json",{"synthetic_only":True}); return metrics

def main():
    p=argparse.ArgumentParser(prog="umirage"); sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("registry"); s=sub.add_parser("smoke"); s.add_argument("--output",default="artifacts/smoke")
    a=p.parse_args()
    if a.cmd=="registry": print(json.dumps(REGISTRY,indent=2))
    else: print(json.dumps(run_smoke(a.output),indent=2))
if __name__=="__main__":main()
