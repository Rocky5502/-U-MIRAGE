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


class SourceAwarePropagator:
    def __init__(self, weights: PropagationWeights | None = None):
        self.w = weights or PropagationWeights()

    def propagate(
        self,
        visual: float,
        integration: float,
        reasoning: float,
    ) -> tuple[float, dict[str, float]]:
        interaction = max(
            visual * integration,
            integration * reasoning,
            visual * reasoning,
        )
        numerator = (
            self.w.visual * visual
            + self.w.integration * integration
            + self.w.reasoning * reasoning
            + self.w.interaction * interaction
        )
        denominator = (
            self.w.visual
            + self.w.integration
            + self.w.reasoning
            + self.w.interaction
        )
        risk = clamp01(numerator / denominator)

        values = {
            "visual": visual,
            "integration": integration,
            "reasoning": reasoning,
        }
        exponentials = {key: math.exp(4 * value) for key, value in values.items()}
        normalizer = sum(exponentials.values())
        posterior = {
            key: value / normalizer
            for key, value in exponentials.items()
        }
        return risk, posterior
