from __future__ import annotations
from .common import clamp01, disagreement
from ..schema import ModelOutput

def integration_uncertainty(full: ModelOutput, leave_one_out: list[ModelOutput], order_variants: list[ModelOutput]) -> float:
    variants=leave_one_out+order_variants
    if not variants: return 0.0
    ans=sum(disagreement(full.answer,v.answer) for v in variants)/len(variants)
    conf=sum(abs(full.confidence-v.confidence) for v in variants)/len(variants)
    return clamp01(0.7*ans + 0.3*conf)
