from __future__ import annotations

import math
from dataclasses import dataclass

from .signals.common import clamp01


@dataclass(frozen=True)
class PropagationWeights:
    """Weights for the deterministic pilot risk propagator.

    The pilot deliberately uses a transparent monotonic aggregator before we
    fit any learned propagation module.  This makes the first source-separation
    experiments auditable and gives us a strong non-learned baseline for the
    paper.
    """

    visual: float = 0.30
    integration: float = 0.40
    reasoning: float = 0.30
    interaction: float = 0.20
    dominant: float = 0.10
    posterior_scale: float = 4.0

    def __post_init__(self) -> None:
        weighted_terms = (
            self.visual,
            self.integration,
            self.reasoning,
            self.interaction,
            self.dominant,
        )
        if any(value < 0.0 for value in weighted_terms):
            raise ValueError("propagation weights must be non-negative")
        if sum(weighted_terms) <= 0.0:
            raise ValueError("at least one propagation weight must be positive")
        if self.posterior_scale <= 0.0:
            raise ValueError("posterior_scale must be positive")


class SourceAwarePropagator:
    """Transparent source-aware uncertainty aggregation for the pilot stage.

    Inputs are clamped to [0, 1].  The risk score is monotonic in each source
    score and additionally includes (i) the strongest pairwise interaction and
    (ii) the dominant source.  The source posterior is a stable softmax over
    the three source scores and is intended for attribution/routing, not as a
    calibrated probability of failure.
    """

    SOURCES = ("visual", "integration", "reasoning")

    def __init__(self, weights: PropagationWeights | None = None):
        self.w = weights or PropagationWeights()

    def _posterior(self, values: dict[str, float]) -> dict[str, float]:
        logits = {key: self.w.posterior_scale * value for key, value in values.items()}
        offset = max(logits.values())
        exp_scores = {key: math.exp(value - offset) for key, value in logits.items()}
        normalizer = sum(exp_scores.values())
        return {key: value / normalizer for key, value in exp_scores.items()}

    def propagate(
        self,
        visual: float,
        integration: float,
        reasoning: float,
    ) -> tuple[float, dict[str, float]]:
        values = {
            "visual": clamp01(float(visual)),
            "integration": clamp01(float(integration)),
            "reasoning": clamp01(float(reasoning)),
        }

        v = values["visual"]
        i = values["integration"]
        r = values["reasoning"]
        interaction = max(v * i, i * r, v * r)
        dominant = max(values.values())

        numerator = (
            self.w.visual * v
            + self.w.integration * i
            + self.w.reasoning * r
            + self.w.interaction * interaction
            + self.w.dominant * dominant
        )
        denominator = (
            self.w.visual
            + self.w.integration
            + self.w.reasoning
            + self.w.interaction
            + self.w.dominant
        )
        risk = clamp01(numerator / denominator)
        return risk, self._posterior(values)
