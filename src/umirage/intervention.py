from __future__ import annotations


def recommend_action(
    risk: float,
    posterior: dict[str, float],
    abstain_threshold: float = 0.82,
) -> str:
    if risk >= abstain_threshold:
        return "abstain"

    source = max(posterior, key=posterior.get)
    return {
        "visual": "reinspect_image",
        "integration": "compare_images",
        "reasoning": "verify_reasoning",
    }[source]
