from umirage.adapters.mock import MockMultiImageAdapter
from umirage.pipeline import UMiragePipeline
from umirage.schema import Episode


def ep(source="integration"):
    return Episode("x",("a","b","c"),"q","alpha",source,{"synthetic":{"risk":0.76,"wrong_answer":"beta"}})

def test_pipeline_bounds_and_action():
    _,s=UMiragePipeline(MockMultiImageAdapter()).analyze(ep())
    assert all(0<=x<=1 for x in (s.visual,s.integration,s.reasoning,s.risk))
    assert abs(sum(s.source_posterior.values())-1)<1e-9
    assert s.action in {"reinspect_image","compare_images","verify_reasoning","abstain"}

def test_deterministic():
    p=UMiragePipeline(MockMultiImageAdapter())
    assert p.analyze(ep())[1]==p.analyze(ep())[1]
