from __future__ import annotations
from .schema import Episode, UncertaintyState
from .signals.visual import visual_uncertainty
from .signals.integration import integration_uncertainty
from .signals.reasoning import reasoning_uncertainty
from .propagation import SourceAwarePropagator
from .intervention import recommend_action

class UMiragePipeline:
    def __init__(self, adapter, propagator=None):
        self.adapter=adapter; self.propagator=propagator or SourceAwarePropagator()

    def analyze(self, ep: Episode) -> tuple[object,UncertaintyState]:
        full=self.adapter.answer(ep)
        loo=[]
        if len(ep.images)>1:
            for i in range(len(ep.images)):
                keep=tuple(j for j in range(len(ep.images)) if j!=i)
                loo.append(self.adapter.answer(ep,image_indices=keep))
        order=[]
        if len(ep.images)>1:
            order.append(self.adapter.answer(ep,order=tuple(reversed(range(len(ep.images))))))
        reasoning=[self.adapter.answer(ep,prompt_variant=i) for i in (1,2)]
        u_v=visual_uncertainty(full)
        u_i=integration_uncertainty(full,loo,order)
        u_r=reasoning_uncertainty(full,reasoning)
        risk,post=self.propagator.propagate(u_v,u_i,u_r)
        action=recommend_action(risk,post)
        return full,UncertaintyState(u_v,u_i,u_r,risk,post,action)
