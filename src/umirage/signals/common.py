from __future__ import annotations

import math


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def disagreement(a: str, b: str) -> float:
    a_norm = " ".join(a.lower().strip().split())
    b_norm = " ".join(b.lower().strip().split())
    return 0.0 if a_norm == b_norm else 1.0


def normalized_entropy(ps: list[float]) -> float:
    values = [max(1e-12, float(x)) for x in ps]
    total = sum(values)
    values = [x / total for x in values]
    if len(values) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in values) / math.log(len(values))
    return clamp01(entropy)
