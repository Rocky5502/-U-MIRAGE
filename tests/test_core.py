from umirage.adapters.mock import MockMultiImageAdapter
from umirage.pipeline import UMiragePipeline
from umirage.schema import Episode


def ep(source="integration"):
    return Episode(
        "x",
        ("a", "b", "c"),
        "q",
        "alpha",
        source,
        {"synthetic": {"risk": 0.76, "wrong_answer": "beta"}},
    )


def test_pipeline_bounds_and_action():
    _, state = UMiragePipeline(MockMultiImageAdapter()).analyze(ep())
    assert all(0 <= x <= 1 for x in (state.visual, state.integration, state.reasoning, state.risk))
    assert abs(sum(state.source_posterior.values()) - 1) < 1e-9
    assert state.action in {
        "answer",
        "reinspect_image",
        "compare_images",
        "verify_reasoning",
        "abstain",
    }


def test_pipeline_preserves_perturbation_audit_trail():
    _, state = UMiragePipeline(MockMultiImageAdapter()).analyze(ep())
    diagnostics = state.diagnostics

    assert diagnostics["n_images"] == 3
    assert set(diagnostics["full"]) >= {
        "answer",
        "confidence",
        "token_entropy",
        "evidence_scores",
        "metadata",
    }
    assert len(diagnostics["leave_one_out"]) == 3
    assert {row["omitted_index"] for row in diagnostics["leave_one_out"]} == {0, 1, 2}
    assert len(diagnostics["order_variants"]) == 1
    assert diagnostics["order_variants"][0]["order"] == [2, 1, 0]
    assert len(diagnostics["reasoning_variants"]) == 2
    assert {row["prompt_variant"] for row in diagnostics["reasoning_variants"]} == {1, 2}


def test_deterministic():
    pipeline = UMiragePipeline(MockMultiImageAdapter())
    assert pipeline.analyze(ep())[1] == pipeline.analyze(ep())[1]
