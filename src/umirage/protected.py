from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("visual", "integration", "reasoning")
LABELS = ("clean",) + SOURCES


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class SourceScores:
    visual: float
    integration: float
    reasoning: float

    def __post_init__(self) -> None:
        for name in SOURCES:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1], got {value}")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SOURCES}

    def argmax(self) -> str:
        d = self.as_dict()
        return max(SOURCES, key=lambda k: (d[k], -SOURCES.index(k)))


@dataclass(frozen=True)
class PropagationWeights:
    visual: float = 1.0
    integration: float = 1.0
    reasoning: float = 1.0
    interaction: float = 1.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        vals = (
            self.visual,
            self.integration,
            self.reasoning,
            self.interaction,
            self.maximum,
        )
        if any(float(x) < 0 for x in vals):
            raise ValueError("Propagation weights must be non-negative")
        if sum(float(x) for x in vals) <= 0:
            raise ValueError("At least one propagation weight must be positive")


def no_propagation_risk(
    scores: SourceScores,
    weights: Mapping[str, float] | None = None,
) -> float:
    w = dict(weights or {"visual": 1.0, "integration": 1.0, "reasoning": 1.0})
    if set(w) != set(SOURCES):
        raise ValueError(f"no-propagation weights must be exactly {SOURCES}")
    if any(float(v) < 0 for v in w.values()) or sum(float(v) for v in w.values()) <= 0:
        raise ValueError("No-propagation weights must be non-negative with positive sum")
    d = scores.as_dict()
    return clamp01(
        sum(float(w[k]) * d[k] for k in SOURCES) / sum(float(w[k]) for k in SOURCES)
    )


def propagation_terms(scores: SourceScores) -> dict[str, float]:
    v, i, r = scores.visual, scores.integration, scores.reasoning
    return {
        "interaction": max(v * i, i * r, v * r),
        "maximum": max(v, i, r),
    }


def propagated_risk(
    scores: SourceScores,
    weights: PropagationWeights | None = None,
) -> float:
    w = weights or PropagationWeights()
    t = propagation_terms(scores)
    num = (
        w.visual * scores.visual
        + w.integration * scores.integration
        + w.reasoning * scores.reasoning
        + w.interaction * t["interaction"]
        + w.maximum * t["maximum"]
    )
    den = w.visual + w.integration + w.reasoning + w.interaction + w.maximum
    return clamp01(num / den)


def source_posterior(scores: SourceScores, beta: float = 4.0) -> dict[str, float]:
    d = scores.as_dict()
    ex = {k: math.exp(float(beta) * d[k]) for k in SOURCES}
    z = sum(ex.values())
    return {k: ex[k] / z for k in SOURCES}


def selectivity(
    control: SourceScores,
    intervention: SourceScores,
    source: str,
) -> tuple[float, dict[str, float]]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    c, x = control.as_dict(), intervention.as_dict()
    delta = {k: x[k] - c[k] for k in SOURCES}
    off = max(delta[k] for k in SOURCES if k != source)
    return delta[source] - off, delta


def source_action(source: str) -> str:
    return {
        "visual": "reinspect_image",
        "integration": "compare_images",
        "reasoning": "verify_reasoning",
    }[source]


def source_action_prompt(source: str, question: str) -> str:
    prompts = {
        "visual": (
            "Reinspect the image evidence relevant to the question. Identify the specific image(s) "
            "that support the answer, check the visual facts again, and then answer. "
            "Do not rely on the previous answer."
        ),
        "integration": (
            "Re-compare all provided images. Extract the needed fact from each image independently, "
            "then explicitly bind/aggregate those facts before answering. Do not rely on the previous answer."
        ),
        "reasoning": (
            "Verify the composition step using the visual facts. Recompute the comparison/count/logical "
            "operation carefully, then answer. Do not rely on the previous answer."
        ),
    }
    return f"{question}\n\n{prompts[source]}"


def generic_verify_prompt(question: str) -> str:
    return (
        f"{question}\n\nVerify your answer carefully using the provided images. "
        "Check the evidence and reasoning again, then give the final answer."
    )


def generic_requery_prompt(question: str) -> str:
    return f"{question}\n\nAnswer the question again from scratch using only the provided images."


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def require_fields(row: Mapping[str, object], fields: Iterable[str]) -> None:
    missing = [k for k in fields if k not in row]
    if missing:
        raise KeyError(f"missing required fields: {missing}")


def scores_from_row(row: Mapping[str, object]) -> SourceScores:
    if isinstance(row.get("scores"), Mapping):
        d = row["scores"]
        return SourceScores(float(d["visual"]), float(d["integration"]), float(d["reasoning"]))
    return SourceScores(
        float(row["visual"]),
        float(row["integration"]),
        float(row["reasoning"]),
    )


def group_prefix(group_id: str) -> str:
    parts = str(group_id).split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else str(group_id)


def flatten_groups(
    rows: Sequence[Mapping[str, object]],
    group_key: str = "group_id",
) -> dict[str, list[Mapping[str, object]]]:
    out: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        key = str(row[group_key])
        out.setdefault(key, []).append(row)
    return out
