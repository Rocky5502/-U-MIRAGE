from __future__ import annotations

import math
import random
from collections.abc import Callable


Metric = Callable[[list[dict]], float]


def align_paired_rows(
    left: list[dict], right: list[dict], id_key: str = "id"
) -> tuple[list[dict], list[dict]]:
    """Align two prediction sets by episode id and reject incomplete pairing."""
    left_map = {str(row[id_key]): row for row in left}
    right_map = {str(row[id_key]): row for row in right}
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise ValueError(f"duplicate {id_key!r} values are not allowed")
    if set(left_map) != set(right_map):
        missing_left = sorted(set(right_map) - set(left_map))[:5]
        missing_right = sorted(set(left_map) - set(right_map))[:5]
        raise ValueError(
            "paired runs must contain identical episode ids; "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )
    ids = sorted(left_map)
    return [left_map[i] for i in ids], [right_map[i] for i in ids]


def paired_bootstrap_delta(
    left: list[dict],
    right: list[dict],
    metric: Metric,
    *,
    samples: int = 5000,
    seed: int = 27,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Paired percentile bootstrap for metric(left) - metric(right)."""
    if samples <= 0:
        raise ValueError("samples must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    left, right = align_paired_rows(left, right)
    n = len(left)
    if n == 0:
        raise ValueError("paired bootstrap requires at least one episode")

    observed = float(metric(left) - metric(right))
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        left_sample = [left[i] for i in indices]
        right_sample = [right[i] for i in indices]
        delta = float(metric(left_sample) - metric(right_sample))
        if math.isfinite(delta):
            deltas.append(delta)

    if not deltas:
        raise ValueError("metric produced no finite bootstrap samples")
    deltas.sort()

    def percentile(p: float) -> float:
        if len(deltas) == 1:
            return deltas[0]
        position = p * (len(deltas) - 1)
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        if lo == hi:
            return deltas[lo]
        weight = position - lo
        return deltas[lo] * (1 - weight) + deltas[hi] * weight

    lower = percentile(alpha / 2)
    upper = percentile(1 - alpha / 2)
    return {
        "n": n,
        "samples": len(deltas),
        "seed": seed,
        "alpha": alpha,
        "delta": observed,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def paired_accuracy_counts(left: list[dict], right: list[dict]) -> dict[str, int]:
    """Return paired win/loss/tie counts using correctness as the outcome."""
    left, right = align_paired_rows(left, right)
    wins = losses = ties = 0
    for a, b in zip(left, right):
        a_ok = bool(a["correct"])
        b_ok = bool(b["correct"])
        if a_ok and not b_ok:
            wins += 1
        elif b_ok and not a_ok:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "n": len(left)}


def mcnemar_exact_p(left: list[dict], right: list[dict]) -> float:
    """Two-sided exact McNemar p-value using only discordant pairs."""
    counts = paired_accuracy_counts(left, right)
    b = counts["wins"]
    c = counts["losses"]
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def holm_bonferroni(
    pvalues: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, float | bool]]:
    """Return Holm-adjusted p-values and reject decisions."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    for name, value in pvalues.items():
        if not 0 <= value <= 1:
            raise ValueError(f"invalid p-value for {name}: {value}")

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (m - index) * pvalue)
        running = max(running, candidate)
        adjusted[name] = running

    return {
        name: {
            "raw_p": float(pvalues[name]),
            "adjusted_p": float(adjusted[name]),
            "reject": bool(adjusted[name] <= alpha),
        }
        for name in pvalues
    }
