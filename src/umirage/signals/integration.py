from __future__ import annotations

from ..schema import ModelOutput
from .common import clamp01, disagreement


def integration_uncertainty(
    full: ModelOutput,
    leave_one_out: list[ModelOutput],
    order_variants: list[ModelOutput],
) -> float:
    variants = leave_one_out + order_variants
    if not variants:
        return 0.0

    answer_disagreement = sum(disagreement(full.answer, variant.answer) for variant in variants)
    answer_disagreement /= len(variants)
    confidence_shift = sum(abs(full.confidence - variant.confidence) for variant in variants)
    confidence_shift /= len(variants)
    return clamp01(0.7 * answer_disagreement + 0.3 * confidence_shift)
