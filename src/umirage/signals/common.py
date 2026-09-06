from __future__ import annotations
import math

def clamp01(x: float) -> float:
    return max(0.0,min(1.0,float(x)))

def disagreement(a: str, b: str) -> float:
    a=" ".join(a.lower().strip().split()); b=" ".join(b.lower().strip().split())
    return 0.0 if a==b else 1.0

def normalized_entropy(ps: list[float]) -> float:
    ps=[max(1e-12,float(x)) for x in ps]
    s=sum(ps); ps=[x/s for x in ps]
    if len(ps)<=1: return 0.0
    return clamp01(-sum(p*math.log(p) for p in ps)/math.log(len(ps)))
