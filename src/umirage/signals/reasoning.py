from __future__ import annotations
from .common import clamp01, disagreement
from ..schema import ModelOutput

def reasoning_uncertainty(base: ModelOutput, variants: list[ModelOutput]) -> float:
    if not variants: return clamp01(base.token_entropy)
    ans=sum(disagreement(base.answer,v.answer) for v in variants)/len(variants)
    conf=sum(abs(base.confidence-v.confidence) for v in variants)/len(variants)
    return clamp01(0.5*base.token_entropy + 0.35*ans + 0.15*conf)
