from __future__ import annotations

import math


def accuracy(rows: list[dict]) -> float:
    return sum(bool(r["correct"]) for r in rows) / len(rows) if rows else float("nan")


def _failures(rows: list[dict]) -> list[int]:
    return [1 - int(bool(r["correct"])) for r in rows]


def brier(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    y = _failures(rows)
    return sum((float(r["risk"]) - t) ** 2 for r, t in zip(rows, y)) / len(rows)


def ece(rows: list[dict], bins: int = 10) -> float:
    if not rows:
        return float("nan")
    total = 0.0
    n = len(rows)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        xs = [
            r
            for r in rows
            if lo <= float(r["risk"]) < hi or (b == bins - 1 and float(r["risk"]) == 1.0)
        ]
        if xs:
            pred = sum(float(r["risk"]) for r in xs) / len(xs)
            err = sum(int(not r["correct"]) for r in xs) / len(xs)
            total += len(xs) / n * abs(pred - err)
    return total


def auroc(rows: list[dict]) -> float:
    pos = [float(r["risk"]) for r in rows if not r["correct"]]
    neg = [float(r["risk"]) for r in rows if r["correct"]]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auprc(rows: list[dict]) -> float:
    """Average precision for failure detection using risk as the score."""
    ordered = sorted(rows, key=lambda r: float(r["risk"]), reverse=True)
    positives = sum(int(not r["correct"]) for r in ordered)
    if positives == 0:
        return float("nan")
    hit = 0
    precision_sum = 0.0
    for rank, row in enumerate(ordered, 1):
        if not row["correct"]:
            hit += 1
            precision_sum += hit / rank
    return precision_sum / positives


def risk_coverage(rows: list[dict], coverages=(0.5, 0.7, 0.8, 0.9)) -> list[dict]:
    ranked = sorted(rows, key=lambda r: float(r["risk"]))
    out = []
    for coverage in coverages:
        n = max(1, int(round(coverage * len(ranked)))) if ranked else 0
        kept = ranked[:n]
        selective_risk = 1.0 - accuracy(kept) if kept else float("nan")
        out.append({"coverage": coverage, "selective_risk": selective_risk, "n": n})
    return out


def aurc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    ordered = sorted(rows, key=lambda r: float(r["risk"]))
    errors = 0
    values = []
    for i, row in enumerate(ordered, 1):
        errors += int(not row["correct"])
        values.append(errors / i)
    return sum(values) / len(values)


def source_accuracy(rows: list[dict]) -> float:
    valid = [r for r in rows if r.get("source_label") in {"visual", "integration", "reasoning"}]
    if not valid:
        return float("nan")
    return sum(r["predicted_source"] == r["source_label"] for r in valid) / len(valid)


def source_macro_f1(rows: list[dict], labels=("visual", "integration", "reasoning")) -> float:
    valid = [r for r in rows if r.get("source_label") in set(labels)]
    if not valid:
        return float("nan")
    scores = []
    for label in labels:
        tp = sum(r.get("source_label") == label and r.get("predicted_source") == label for r in valid)
        fp = sum(r.get("source_label") != label and r.get("predicted_source") == label for r in valid)
        fn = sum(r.get("source_label") == label and r.get("predicted_source") != label for r in valid)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def summarize(rows: list[dict]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "auroc": auroc(rows),
        "auprc": auprc(rows),
        "ece": ece(rows),
        "brier": brier(rows),
        "aurc": aurc(rows),
        "source_accuracy": source_accuracy(rows),
        "source_macro_f1": source_macro_f1(rows),
    }
