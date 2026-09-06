from __future__ import annotations
from .pipeline import UMiragePipeline

def run_episodes(episodes, adapter) -> list[dict]:
    pipe=UMiragePipeline(adapter); rows=[]
    for ep in episodes:
        out,state=pipe.analyze(ep)
        pred=max(state.source_posterior,key=state.source_posterior.get)
        rows.append({
            "id":ep.id,"prediction":out.answer,"gold":ep.answer,"correct":out.answer.strip().lower()==ep.answer.strip().lower(),
            "source_label":ep.source_label,"visual":state.visual,"integration":state.integration,
            "reasoning":state.reasoning,"risk":state.risk,"predicted_source":pred,"action":state.action,
            "source_posterior":state.source_posterior,
        })
    return rows
