from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds for the deterministic pilot intervention policy."""

    answer_threshold: float = 0.35
    abstain_threshold: float = 0.82
    min_source_posterior: float = 0.45

    def __post_init__(self) -> None:
        for name, value in (
            ("answer_threshold", self.answer_threshold),
            ("abstain_threshold", self.abstain_threshold),
            ("min_source_posterior", self.min_source_posterior),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.answer_threshold >= self.abstain_threshold:
            raise ValueError("answer_threshold must be smaller than abstain_threshold")


def recommend_action(
    risk: float,
    posterior: dict[str, float],
    config: PolicyConfig | None = None,
) -> str:
    """Map calibrated risk and source attribution to an auditable action.

    The policy is intentionally deterministic for the pilot.  Low-risk cases
    are answered directly; very high-risk cases abstain.  Intermediate-risk
    cases route to the source-specific intervention.  If attribution is too
    diffuse, we use the generic reasoning verification path rather than
    pretending that a source has been localized confidently.
    """

    cfg = config or PolicyConfig()
    risk = min(1.0, max(0.0, float(risk)))

    if risk <= cfg.answer_threshold:
        return "answer"
    if risk >= cfg.abstain_threshold:
        return "abstain"

    expected_sources = {"visual", "integration", "reasoning"}
    if set(posterior) != expected_sources:
        raise ValueError(f"posterior must contain exactly {sorted(expected_sources)}")

    total = sum(float(value) for value in posterior.values())
    if total <= 0.0:
        raise ValueError("posterior mass must be positive")
    normalized = {key: max(0.0, float(value)) / total for key, value in posterior.items()}
    source = max(normalized, key=normalized.get)

    if normalized[source] < cfg.min_source_posterior:
        return "verify_reasoning"

    return {
        "visual": "reinspect_image",
        "integration": "compare_images",
        "reasoning": "verify_reasoning",
    }[source]
