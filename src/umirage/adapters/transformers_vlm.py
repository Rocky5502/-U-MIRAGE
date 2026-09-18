from __future__ import annotations

import math
from pathlib import Path

from ..schema import Episode, ModelOutput


class TransformersVLMAdapter:
    """Generic Hugging Face image-text-to-text boundary."""

    def __init__(self, model_id: str, revision: str | None = None, device_map: str = "auto", max_new_tokens: int = 128, trust_remote_code: bool = False):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install `pip install -e .[vlm]` for real VLM inference") from exc
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, device_map=device_map, torch_dtype="auto", trust_remote_code=trust_remote_code)
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.name = model_id

    @staticmethod
    def _prompt(question: str, variant: int) -> str:
        suffixes = {
            0: "Answer using only evidence in the provided images.",
            1: "Identify the relevant visual evidence first, then answer concisely.",
            2: "Cross-check evidence across images before answering concisely.",
        }
        return f"{question}\n{suffixes.get(variant, 'Answer concisely and ground the answer in the images.')}"

    def _score_generation(self, scores) -> tuple[float, float]:
        confidences, entropies = [], []
        for score in scores or ():
            probs = self.torch.softmax(score[0].float(), dim=-1)
            confidences.append(float(probs.max().item()))
            raw_h = -(probs * probs.clamp_min(1e-12).log()).sum()
            entropies.append(float(raw_h.item() / math.log(max(2, probs.numel()))))
        return (
            sum(confidences) / len(confidences) if confidences else 0.5,
            sum(entropies) / len(entropies) if entropies else 0.5,
        )

    def answer(self, episode: Episode, *, image_indices=None, order=None, prompt_variant=0, prompt_override: str | None = None) -> ModelOutput:
        from PIL import Image

        idx = image_indices if image_indices is not None else tuple(range(len(episode.images)))
        if order is not None:
            idx = tuple(order)
        image_paths = [Path(episode.images[i]) for i in idx]
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
        images = [Image.open(path).convert("RGB") for path in image_paths]
        prompt = prompt_override if prompt_override is not None else self._prompt(episode.question, prompt_variant)
        messages = [{"role": "user", "content": [*[{"type": "image"} for _ in images], {"type": "text", "text": prompt}]}]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=True)
        input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated = out.sequences[:, input_len:] if input_len else out.sequences
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        confidence, entropy = self._score_generation(out.scores)
        return ModelOutput(
            answer,
            confidence,
            max(0.0, min(1.0, entropy)),
            tuple([confidence] * len(images)),
            {
                "real_model": True,
                "model_id": self.model_id,
                "revision": self.revision,
                "image_indices": tuple(idx),
                "prompt_variant": prompt_variant,
                "prompt_override": prompt_override is not None,
                "input_tokens": input_len,
                "output_tokens": int(generated.shape[-1]),
            },
        )
