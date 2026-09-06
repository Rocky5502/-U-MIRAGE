from __future__ import annotations

def _failures(rows): return [1-int(bool(r["correct"])) for r in rows]
def brier(rows):
    y=_failures(rows); return sum((float(r["risk"])-t)**2 for r,t in zip(rows,y))/len(rows) if rows else float("nan")
def ece(rows,bins=10):
    if not rows:return float("nan")
    total=0.0;n=len(rows)
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins; xs=[r for r in rows if lo<=float(r["risk"])<hi or (b==bins-1 and float(r["risk"])==1)]
        if xs:
            pred=sum(float(r["risk"]) for r in xs)/len(xs);err=sum(int(not r["correct"]) for r in xs)/len(xs);total+=len(xs)/n*abs(pred-err)
    return total
def auroc(rows):
    pos=[float(r["risk"]) for r in rows if not r["correct"]];neg=[float(r["risk"]) for r in rows if r["correct"]]
    if not pos or not neg:return float("nan")
    return sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))
def aurc(rows):
    if not rows:return float("nan")
    ordered=sorted(rows,key=lambda r:float(r["risk"]));errors=0;vals=[]
    for i,r in enumerate(ordered,1):errors+=int(not r["correct"]);vals.append(errors/i)
    return sum(vals)/len(vals)
def source_macro_f1(rows,labels=("visual","integration","reasoning")):
    scores=[]
    for label in labels:
        tp=sum(r.get("source_label")==label and r.get("predicted_source")==label for r in rows);fp=sum(r.get("source_label")!=label and r.get("predicted_source")==label for r in rows);fn=sum(r.get("source_label")==label and r.get("predicted_source")!=label for r in rows)
        p=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.;scores.append(2*p*rec/(p+rec) if p+rec else 0.)
    return sum(scores)/len(scores)
def summarize(rows):
    return {"n":len(rows),"accuracy":sum(bool(r["correct"]) for r in rows)/len(rows) if rows else float("nan"),"auroc":auroc(rows),"ece":ece(rows),"brier":brier(rows),"aurc":aurc(rows),"source_macro_f1":source_macro_f1(rows)}
