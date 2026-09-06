from __future__ import annotations
import math
from dataclasses import dataclass
from .signals.common import clamp01

@dataclass(frozen=True)
class PropagationWeights:
    visual: float=0.30
    integration: float=0.40
    reasoning: float=0.30
    interaction: float=0.20

class SourceAwarePropagator:
    def __init__(self, weights: PropagationWeights | None=None): self.w=weights or PropagationWeights()
    def propagate(self, visual: float, integration: float, reasoning: float) -> tuple[float,dict[str,float]]:
        interaction=max(visual*integration, integration*reasoning, visual*reasoning)
        z=(self.w.visual*visual + self.w.integration*integration + self.w.reasoning*reasoning + self.w.interaction*interaction)
        risk=clamp01(z/(self.w.visual+self.w.integration+self.w.reasoning+self.w.interaction))
        vals={"visual":visual,"integration":integration,"reasoning":reasoning}
        ex={k:math.exp(4*v) for k,v in vals.items()}; s=sum(ex.values())
        return risk,{k:v/s for k,v in ex.items()}
