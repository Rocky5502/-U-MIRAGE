from __future__ import annotations
import json
from pathlib import Path

def main():
    out=Path("data/examples/mirage_demo.jsonl"); out.parent.mkdir(parents=True,exist_ok=True)
    sources=["clean","visual","integration","reasoning"]
    rows=[]
    for i in range(16):
        source=sources[i%4]
        rows.append({"id":f"demo-{i:03d}","images":[f"virtual://{i}/a",f"virtual://{i}/b",f"virtual://{i}/c"],
                     "question":"Which synthetic label is supported by the evidence?","answer":"alpha","source_label":source,
                     "metadata":{"synthetic":{"confidence":0.92,"risk":{"clean":0.05,"visual":0.72,"integration":0.78,"reasoning":0.70}[source],"bad_image":0,"wrong_answer":"beta"}}})
    with out.open("w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r)+"\n")
    print(f"wrote {len(rows)} synthetic episodes -> {out}")
if __name__=="__main__": main()
