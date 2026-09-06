from umirage.schema import Episode
from umirage.adapters.mock import MockMultiImageAdapter
from umirage.pipeline import UMiragePipeline


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


def test_deterministic():
    pipeline = UMiragePipeline(MockMultiImageAdapter())
    assert pipeline.analyze(ep())[1] == pipeline.analyze(ep())[1]
