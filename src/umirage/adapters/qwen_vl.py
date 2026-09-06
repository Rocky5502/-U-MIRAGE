from __future__ import annotations

from pathlib import Path

from ..schema import Episode, ModelOutput
from .transformers_vlm import TransformersVLMAdapter


class QwenVLAdapter(TransformersVLMAdapter):
    """Qwen2.5-VL / Qwen3-VL adapter with Qwen multimodal message preprocessing."""

    def answer(self, episode: Episode, *, image_indices=None, order=None, prompt_variant=0) -> ModelOutput:
        idx = image_indices if image_indices is not None else tuple(range(len(episode.images)))
        if order is not None:
            idx = tuple(order)
        paths = [str(Path(episode.images[i])) for i in idx]
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")

        prompt = self._prompt(episode.question, prompt_variant)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path} for path in paths],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Qwen adapter requires `qwen-vl-utils`; install `pip install -e .[vlm]`."
            ) from exc
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated = out.sequences[:, input_len:] if input_len else out.sequences
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        confidence, entropy = self._score_generation(out.scores)
        return ModelOutput(
            answer,
            confidence,
            entropy,
            tuple([confidence] * len(paths)),
            {
                "real_model": True,
                "model_id": self.model_id,
                "revision": self.revision,
                "image_indices": tuple(idx),
                "prompt_variant": prompt_variant,
            },
        )
