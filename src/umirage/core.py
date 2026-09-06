from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import math

SOURCES=("clean","visual","integration","reasoning")
ACTIONS=("reinspect_image","compare_images","verify_reasoning","abstain")

def _clip(x: float) -> float: return max(0.0,min(1.0,float(x)))
def _norm(s: str) -> str: return " ".join(s.lower().strip().split())
def _disagree(a: str,b: str) -> float: return float(_norm(a)!=_norm(b))

@dataclass(frozen=True)
class Episode:
    id: str; images: tuple[str,...]; question: str; answer: str
    source_label: str="clean"; metadata: dict[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        if self.source_label not in SOURCES: raise ValueError(f"invalid source_label={self.source_label}")
        if not self.images: raise ValueError("episode needs >=1 image")

@dataclass(frozen=True)
class ModelOutput:
    answer: str; confidence: float; token_entropy: float
    evidence_scores: tuple[float,...]=(); metadata: dict[str,Any]=field(default_factory=dict)

@dataclass(frozen=True)
class UncertaintyState:
    visual: float; integration: float; reasoning: float; risk: float
    source_posterior: dict[str,float]; action: str
    def to_dict(self): return asdict(self)

def visual_uncertainty(o: ModelOutput) -> float:
    support=sum(o.evidence_scores)/len(o.evidence_scores) if o.evidence_scores else o.confidence
    return _clip(.55*o.token_entropy+.30*(1-o.confidence)+.15*(1-support))

def integration_uncertainty(base: ModelOutput, variants: list[ModelOutput]) -> float:
    if not variants: return 0.0
    d=sum(_disagree(base.answer,v.answer) for v in variants)/len(variants)
    c=sum(abs(base.confidence-v.confidence) for v in variants)/len(variants)
    return _clip(.7*d+.3*c)

def reasoning_uncertainty(base: ModelOutput, variants: list[ModelOutput]) -> float:
    if not variants: return _clip(base.token_entropy)
    d=sum(_disagree(base.answer,v.answer) for v in variants)/len(variants)
    c=sum(abs(base.confidence-v.confidence) for v in variants)/len(variants)
    return _clip(.5*base.token_entropy+.35*d+.15*c)

def propagate(v: float,i: float,r: float) -> tuple[float,dict[str,float]]:
    interaction=max(v*i,i*r,v*r)
    risk=_clip((.30*v+.40*i+.30*r+.20*interaction)/1.20)
    raw={"visual":v,"integration":i,"reasoning":r}; ex={k:math.exp(4*x) for k,x in raw.items()}; z=sum(ex.values())
    return risk,{k:x/z for k,x in ex.items()}

def recommend_action(risk: float, posterior: dict[str,float]) -> str:
    if risk>=.80: return "abstain"
    source=max(posterior,key=posterior.get)
    return {"visual":"reinspect_image","integration":"compare_images","reasoning":"verify_reasoning"}[source]

class UMiragePipeline:
    def __init__(self, adapter): self.adapter=adapter
    def analyze(self, ep: Episode):
        base=self.adapter.answer(ep)
        integration=[]
        if len(ep.images)>1:
            for i in range(len(ep.images)):
                integration.append(self.adapter.answer(ep,image_indices=tuple(j for j in range(len(ep.images)) if j!=i)))
            integration.append(self.adapter.answer(ep,order=tuple(reversed(range(len(ep.images))))))
        traces=[self.adapter.answer(ep,prompt_variant=j) for j in (1,2)]
        v=visual_uncertainty(base); i=integration_uncertainty(base,integration); r=reasoning_uncertainty(base,traces)
        risk,post=propagate(v,i,r); action=recommend_action(risk,post)
        return base,UncertaintyState(v,i,r,risk,post,action)

def brier(rows: list[dict]) -> float:
    if not rows: return float("nan")
    return sum((float(x["risk"])-float(not x["correct"]))**2 for x in rows)/len(rows)

def accuracy(rows: list[dict]) -> float:
    return sum(bool(x["correct"]) for x in rows)/len(rows) if rows else float("nan")
