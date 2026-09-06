from umirage.core import Episode,UMiragePipeline
from umirage.adapters import MockMultiImageAdapter
from umirage.data import REGISTRY

def episode(src="integration"):
 return Episode("x",("a","b","c"),"q","alpha",src,{"synthetic":{"risk":.76,"wrong_answer":"beta"}})
def test_pipeline_is_bounded_and_deterministic():
 p=UMiragePipeline(MockMultiImageAdapter()); _,s=p.analyze(episode()); _,s2=p.analyze(episode())
 assert s==s2 and all(0<=x<=1 for x in (s.visual,s.integration,s.reasoning,s.risk))
 assert abs(sum(s.source_posterior.values())-1)<1e-9
def test_registry(): assert {"mioh","vischainbench","mihbench","rest_plus"}.issubset(REGISTRY)
def test_evaluation_summary():
 from umirage.evaluation import summarize
 rows=[{"correct":True,"risk":.1,"source_label":"visual","predicted_source":"visual"},{"correct":False,"risk":.9,"source_label":"integration","predicted_source":"integration"}]
 m=summarize(rows);assert 0<=m["brier"]<=1 and m["auroc"]==1.0
