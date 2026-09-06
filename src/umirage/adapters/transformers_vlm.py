from __future__ import annotations

import math
from pathlib import Path

from ..schema import Episode, ModelOutput


class TransformersVLMAdapter:
    """Generic Hugging Face image-text-to-text boundary.

    Model-family quirks stay here; the uncertainty pipeline remains model agnostic.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device_map: str = "auto",
        max_new_tokens: int = 128,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        if load_in_4bit and load_in_8bit:
            raise ValueError("4-bit and 8-bit loading are mutually exclusive")

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install `pip install -e .[vlm]` for real VLM inference") from exc

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(dtype_map)}")

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        model_kwargs = {
            "revision": revision,
            "device_map": device_map,
            "torch_dtype": dtype_map[dtype],
            "trust_remote_code": trust_remote_code,
        }
        quantization_mode = "none"
        if load_in_4bit or load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Quantized loading requires bitsandbytes; install `pip install -e .[quant]`."
                ) from exc

            quantization_mode = "4bit" if load_in_4bit else "8bit"
            quant_kwargs = {
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": load_in_8bit,
            }
            if load_in_4bit:
                compute_dtype = dtype_map[dtype]
                if compute_dtype == "auto" or compute_dtype == torch.float32:
                    compute_dtype = torch.float16
                quant_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
                quant_kwargs["bnb_4bit_quant_type"] = "nf4"
                quant_kwargs["bnb_4bit_use_double_quant"] = True
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_mode = quantization_mode
        self.name = model_id

    @staticmethod
    def _prompt(question: str, variant: int) -> str:
        suffixes = {
            0: "Answer using only evidence in the provided images.",
            1: "Identify the relevant visual evidence first, then answer concisely.",
            2: "Cross-check evidence across images before answering concisely.",
        }
        return f"{question}\n{suffixes.get(variant, 'Answer concisely and ground the answer in the images.')}"

    def runtime_metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "quantization": self.quantization_mode,
            "device_map": self.device_map,
            "max_new_tokens": self.max_new_tokens,
        }

    def _score_generation(self, scores, generated_token_ids) -> tuple[float, float]:
        """Return sequence confidence and normalized predictive entropy.

        Confidence must correspond to what the model actually emitted. Using
        the maximum vocabulary probability at each step can overstate confidence
        whenever the generated token is not the argmax (or when generation
        behavior changes). We therefore gather the probability assigned to the
        generated token at each step and aggregate those probabilities with a
        geometric mean, which is length-normalized and numerically stable.
        """

        token_log_probs: list[float] = []
        entropies: list[float] = []
        token_ids = [int(token_id) for token_id in generated_token_ids]

        for step, score in enumerate(scores or ()):
            if step >= len(token_ids):
                break
            probs = self.torch.softmax(score[0].float(), dim=-1)
            token_id = token_ids[step]
            if token_id < 0 or token_id >= probs.numel():
                continue
            token_prob = float(probs[token_id].clamp_min(1e-12).item())
            token_log_probs.append(math.log(token_prob))

            raw_h = -(probs * probs.clamp_min(1e-12).log()).sum()
            denom = math.log(max(2, probs.numel()))
            entropies.append(float(raw_h.item() / denom))

        confidence = (
            math.exp(sum(token_log_probs) / len(token_log_probs))
            if token_log_probs
            else 0.5
        )
        entropy = sum(entropies) / len(entropies) if entropies else 0.5
        return max(0.0, min(1.0, confidence)), max(0.0, min(1.0, entropy))

    def answer(self, episode: Episode, *, image_indices=None, order=None, prompt_variant=0) -> ModelOutput:
        from PIL import Image

        idx = image_indices if image_indices is not None else tuple(range(len(episode.images)))
        if order is not None:
            idx = tuple(order)
        image_paths = [Path(episode.images[i]) for i in idx]
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
        images = [Image.open(path).convert("RGB") for path in image_paths]
        prompt = self._prompt(episode.question, prompt_variant)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
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
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        generated_token_ids = generated[0].detach().cpu().tolist()

        confidence, entropy = self._score_generation(out.scores, generated_token_ids)
        return ModelOutput(
            text,
            confidence,
            entropy,
            tuple([confidence] * len(images)),
            {
                "real_model": True,
                **self.runtime_metadata(),
                "image_indices": tuple(idx),
                "prompt_variant": prompt_variant,
                "confidence_definition": "geometric_mean_generated_token_probability",
                "entropy_definition": "mean_normalized_predictive_entropy",
            },
        )
