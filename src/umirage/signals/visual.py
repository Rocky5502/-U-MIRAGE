from __future__ import annotations

from ..schema import ModelOutput
from .common import clamp01

def visual_uncertainty(output: ModelOutput) -> float:
    evidence_gap = 1.0 - (sum(output.evidence_scores)/len(output.evidence_scores) if output.evidence_scores else output.confidence)
    return clamp01(0.55*output.token_entropy + 0.30*(1-output.confidence) + 0.15*evidence_gap)
