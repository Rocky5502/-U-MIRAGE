from __future__ import annotations

from .intervention import recommend_action
from .propagation import SourceAwarePropagator
from .schema import Episode, ModelOutput, UncertaintyState
from .signals.integration import integration_uncertainty
from .signals.reasoning import reasoning_uncertainty
from .signals.visual import visual_uncertainty


def _output_record(output: ModelOutput) -> dict:
    return {
        "answer": output.answer,
        "confidence": float(output.confidence),
        "token_entropy": float(output.token_entropy),
        "evidence_scores": [float(value) for value in output.evidence_scores],
        "metadata": dict(output.metadata),
    }


class UMiragePipeline:
    def __init__(self, adapter, propagator=None):
        self.adapter = adapter
        self.propagator = propagator or SourceAwarePropagator()

    def analyze(self, ep: Episode) -> tuple[object, UncertaintyState]:
        full = self.adapter.answer(ep)

        leave_one_out = []
        leave_one_out_records = []
        if len(ep.images) > 1:
            for omitted_index in range(len(ep.images)):
                keep = tuple(index for index in range(len(ep.images)) if index != omitted_index)
                output = self.adapter.answer(ep, image_indices=keep)
                leave_one_out.append(output)
                leave_one_out_records.append(
                    {
                        "omitted_index": omitted_index,
                        "kept_indices": list(keep),
                        **_output_record(output),
                    }
                )

        order_variants = []
        order_records = []
        if len(ep.images) > 1:
            reverse_order = tuple(reversed(range(len(ep.images))))
            output = self.adapter.answer(ep, order=reverse_order)
            order_variants.append(output)
            order_records.append(
                {
                    "order": list(reverse_order),
                    **_output_record(output),
                }
            )

        reasoning_variants = []
        reasoning_records = []
        for prompt_variant in (1, 2):
            output = self.adapter.answer(ep, prompt_variant=prompt_variant)
            reasoning_variants.append(output)
            reasoning_records.append(
                {
                    "prompt_variant": prompt_variant,
                    **_output_record(output),
                }
            )

        visual = visual_uncertainty(full)
        integration = integration_uncertainty(full, leave_one_out, order_variants)
        reasoning = reasoning_uncertainty(full, reasoning_variants)
        risk, posterior = self.propagator.propagate(visual, integration, reasoning)
        action = recommend_action(risk, posterior)

        diagnostics = {
            "full": _output_record(full),
            "leave_one_out": leave_one_out_records,
            "order_variants": order_records,
            "reasoning_variants": reasoning_records,
            "n_images": len(ep.images),
        }

        return full, UncertaintyState(
            visual,
            integration,
            reasoning,
            risk,
            posterior,
            action,
            diagnostics,
        )
