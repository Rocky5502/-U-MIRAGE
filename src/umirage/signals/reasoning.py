from __future__ import annotations

from ..schema import ModelOutput
from .common import clamp01, disagreement


def reasoning_uncertainty(base: ModelOutput, variants: list[ModelOutput]) -> float:
    if not variants:
        return clamp01(base.token_entropy)

    answer_disagreement = sum(disagreement(base.answer, variant.answer) for variant in variants)
    answer_disagreement /= len(variants)
    confidence_shift = sum(abs(base.confidence - variant.confidence) for variant in variants)
    confidence_shift /= len(variants)
    return clamp01(
        0.5 * base.token_entropy + 0.35 * answer_disagreement + 0.15 * confidence_shift
    )
