from __future__ import annotations

import hashlib

from ..schema import Episode, ModelOutput


class MockMultiImageAdapter:
    name = "mock-multi-image"

    def _noise(self, key: str) -> float:
        b = hashlib.sha256(key.encode()).digest()[0]
        return (b / 255.0 - 0.5) * 0.08

    def answer(self, episode: Episode, *, image_indices=None, order=None, prompt_variant=0, prompt_override=None) -> ModelOutput:
        meta = episode.metadata.get("synthetic", {})
        base = float(meta.get("confidence", 0.85))
        source = episode.source_label
        risk = float(meta.get("risk", {"clean": 0.08, "visual": 0.72, "integration": 0.76, "reasoning": 0.70}[source]))
        indices = image_indices if image_indices is not None else tuple(range(len(episode.images)))
        if order is None:
            order = indices
        perturb = 0.0
        if source == "visual" and len(indices) < len(episode.images):
            bad = int(meta.get("bad_image", 0))
            if bad not in indices:
                perturb -= 0.30
        if source == "integration" and tuple(order) != tuple(sorted(order)):
            perturb += 0.22
        if source == "reasoning" and prompt_variant:
            perturb += 0.10 * prompt_variant
        prompt_key = hashlib.sha256((prompt_override or "").encode()).hexdigest()[:8]
        n = self._noise(f"{episode.id}|{indices}|{order}|{prompt_variant}|{prompt_key}")
        conf = min(0.99, max(0.01, base - risk * 0.45 - perturb + n))
        correct = risk + max(0, perturb) < 0.62
        ans = episode.answer if correct else str(meta.get("wrong_answer", "unknown"))
        entropy = min(1.0, max(0.0, 1.0 - conf + risk * 0.2))
        ev = tuple(max(0.0, min(1.0, conf - 0.04 * i)) for i in range(len(episode.images)))
        return ModelOutput(ans, conf, entropy, ev, {"synthetic": True, "prompt_override": bool(prompt_override), "input_tokens": 0, "output_tokens": 0})
