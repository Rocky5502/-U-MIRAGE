import math

from umirage.evaluation import auprc, auroc, ece, source_macro_f1, summarize


def rows():
    return [
        {"correct": True, "risk": 0.10, "source_label": "visual", "predicted_source": "visual"},
        {"correct": True, "risk": 0.20, "source_label": "reasoning", "predicted_source": "reasoning"},
        {"correct": False, "risk": 0.80, "source_label": "integration", "predicted_source": "integration"},
        {"correct": False, "risk": 0.90, "source_label": "visual", "predicted_source": "visual"},
    ]


def test_ranking_metrics_are_perfect_when_separated():
    assert auroc(rows()) == 1.0
    assert auprc(rows()) == 1.0


def test_summary_is_bounded():
    m = summarize(rows())
    for key in ("accuracy", "auroc", "auprc", "ece", "brier", "aurc", "source_accuracy", "source_macro_f1"):
        assert not math.isnan(float(m[key]))
        assert 0.0 <= float(m[key]) <= 1.0


def test_source_macro_f1_perfect():
    assert source_macro_f1(rows()) == 1.0
