from __future__ import annotations

from .intervention import recommend_action
from .propagation import SourceAwarePropagator
from .schema import Episode, UncertaintyState
from .signals.integration import integration_uncertainty
from .signals.reasoning import reasoning_uncertainty
from .signals.visual import visual_uncertainty


class UMiragePipeline:
    def __init__(self, adapter, propagator=None):
        self.adapter = adapter
        self.propagator = propagator or SourceAwarePropagator()

    def analyze(self, ep: Episode) -> tuple[object, UncertaintyState]:
        full = self.adapter.answer(ep)

        leave_one_out = []
        if len(ep.images) > 1:
            for image_index in range(len(ep.images)):
                keep = tuple(index for index in range(len(ep.images)) if index != image_index)
                leave_one_out.append(self.adapter.answer(ep, image_indices=keep))

        order_variants = []
        if len(ep.images) > 1:
            reversed_order = tuple(reversed(range(len(ep.images))))
            order_variants.append(self.adapter.answer(ep, order=reversed_order))

        reasoning_variants = [self.adapter.answer(ep, prompt_variant=index) for index in (1, 2)]
        visual = visual_uncertainty(full)
        integration = integration_uncertainty(full, leave_one_out, order_variants)
        reasoning = reasoning_uncertainty(full, reasoning_variants)
        risk, posterior = self.propagator.propagate(visual, integration, reasoning)
        action = recommend_action(risk, posterior)

        return full, UncertaintyState(
            visual,
            integration,
            reasoning,
            risk,
            posterior,
            action,
        )
