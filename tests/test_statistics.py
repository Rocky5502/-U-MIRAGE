from umirage.evaluation import accuracy
from umirage.statistics import (
    align_paired_rows,
    holm_bonferroni,
    mcnemar_exact_p,
    paired_accuracy_counts,
    paired_bootstrap_delta,
)


def _rows(values):
    return [
        {"id": f"e{index}", "correct": value, "risk": 0.2 if value else 0.8}
        for index, value in enumerate(values)
    ]


def test_align_paired_rows_is_id_stable():
    left = _rows([True, False, True])
    right = list(reversed(_rows([True, True, False])))
    a, b = align_paired_rows(left, right)
    assert [row["id"] for row in a] == ["e0", "e1", "e2"]
    assert [row["id"] for row in b] == ["e0", "e1", "e2"]


def test_paired_bootstrap_delta_is_deterministic():
    left = _rows([True, True, True, False])
    right = _rows([True, False, False, False])
    result = paired_bootstrap_delta(left, right, accuracy, samples=500, seed=9)
    assert result["delta"] == 0.5
    assert result["ci_lower"] <= result["delta"] <= result["ci_upper"]
    assert result == paired_bootstrap_delta(left, right, accuracy, samples=500, seed=9)


def test_mcnemar_and_counts_use_discordant_pairs():
    left = _rows([True, True, True, False])
    right = _rows([True, False, False, False])
    counts = paired_accuracy_counts(left, right)
    assert counts == {"wins": 2, "losses": 0, "ties": 2, "n": 4}
    assert mcnemar_exact_p(left, right) == 0.5


def test_holm_bonferroni_monotonic_adjustment():
    result = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.04})
    assert result["a"]["adjusted_p"] <= result["b"]["adjusted_p"]
    assert result["b"]["adjusted_p"] <= result["c"]["adjusted_p"]
    assert result["a"]["reject"] is True
