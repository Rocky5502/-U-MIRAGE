from __future__ import annotations

import math
from dataclasses import dataclass

from .signals.common import clamp01


@dataclass(frozen=True)
class PropagationWeights:
    visual: float = 0.30
    integration: float = 0.40
    reasoning: float = 0.30
    interaction: float = 0.20
    maximum: float = 0.20

    def __post_init__(self) -> None:
        vals = (self.visual, self.integration, self.reasoning, self.interaction, self.maximum)
        if any(v < 0 for v in vals):
            raise ValueError("Propagation weights must be non-negative")
        if sum(vals) <= 0:
            raise ValueError("At least one propagation weight must be positive")


class SourceAwarePropagator:
    """Transparent monotonic source-aware propagator matching the manuscript."""

    def __init__(self, weights: PropagationWeights | None = None, beta: float = 4.0):
        self.w = weights or PropagationWeights()
        self.beta = float(beta)

    def propagate(self, visual: float, integration: float, reasoning: float) -> tuple[float, dict[str, float]]:
        interaction = max(visual * integration, integration * reasoning, visual * reasoning)
        maximum = max(visual, integration, reasoning)
        z = (
            self.w.visual * visual
            + self.w.integration * integration
            + self.w.reasoning * reasoning
            + self.w.interaction * interaction
            + self.w.maximum * maximum
        )
        den = self.w.visual + self.w.integration + self.w.reasoning + self.w.interaction + self.w.maximum
        risk = clamp01(z / den)
        vals = {"visual": visual, "integration": integration, "reasoning": reasoning}
        ex = {k: math.exp(self.beta * v) for k, v in vals.items()}
        total = sum(ex.values())
        return risk, {k: v / total for k, v in ex.items()}
