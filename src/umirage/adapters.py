from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib
import math
from pathlib import Path
from .core import Episode, ModelOutput

class MultiImageAdapter(ABC):
    @abstractmethod
    def answer(self, episode: Episode, image_indices=None, order=None, prompt_variant: int = 0) -> ModelOutput: ...

class MockMultiImageAdapter(MultiImageAdapter):
    """Deterministic synthetic adapter for CI; never a scientific baseline."""
    def answer(self, episode, image_indices=None, order=None, prompt_variant=0):
        meta = episode.metadata.get("synthetic", {})
        risk = float(meta.get("risk", 0.2))
        source = episode.source_label
        key = f"{episode.id}|{image_indices}|{order}|{prompt_variant}".encode()
        jitter = int(hashlib.sha256(key).hexdigest()[:4], 16) / 65535
        perturb = source == "integration" and (image_indices is not None or order is not None)
        perturb = perturb or (source == "reasoning" and prompt_variant != 0)
        wrong = meta.get("wrong_answer", "unknown")
        answer = wrong if (risk > 0.55 and (perturb or source in {"visual", "reasoning"})) else episode.answer
        confidence = max(0.05, min(0.95, 1 - risk * 0.75 + 0.05 * (jitter - 0.5)))
        entropy = max(0.01, min(1.0, risk * (0.85 if source == "reasoning" else 0.55) + 0.08 * jitter))
        evidence = tuple(max(0.02, min(0.98, 1 - risk * (0.9 if source == "visual" else 0.45))) for _ in episode.images)
        return ModelOutput(answer, confidence, entropy, evidence, {"synthetic": True})

class TransformersVLMAdapter(MultiImageAdapter):
    """Practical HF adapter for chat-template VLMs (Qwen-VL style and compatible models)."""
    def __init__(self, model_id: str, revision: str | None = None, device_map: str = "auto", dtype: str = "auto", max_new_tokens: int = 128):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise RuntimeError("Install local VLM dependencies with: pip install -e .[vlm]") from e
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, device_map=device_map, torch_dtype=dtype, trust_remote_code=True)
        self.model_id = model_id; self.revision = revision; self.max_new_tokens = max_new_tokens

    def _prompt(self, question: str, variant: int) -> str:
        suffix = {0:"Answer using only evidence in the provided images.",1:"First identify the relevant visual evidence, then answer concisely.",2:"Cross-check the evidence across images before answering concisely."}.get(variant,"Answer concisely and ground the answer in the images.")
        return f"{question}\n{suffix}"

    @staticmethod
    def _load_images(paths):
        from PIL import Image
        images=[]
        for p in paths:
            path=Path(p)
            if not path.exists(): raise FileNotFoundError(f"Image not found: {path}")
            images.append(Image.open(path).convert("RGB"))
        return images

    def answer(self, episode, image_indices=None, order=None, prompt_variant=0):
        import torch
        idx=tuple(range(len(episode.images))) if image_indices is None else tuple(image_indices)
        if order is not None: idx=tuple(order)
        images=self._load_images([episode.images[i] for i in idx]); prompt=self._prompt(episode.question,prompt_variant)
        messages=[{"role":"user","content":[{"type":"image"} for _ in images]+[{"type":"text","text":prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) if hasattr(self.processor,"apply_chat_template") else prompt
        inputs=self.processor(text=[text],images=images,padding=True,return_tensors="pt")
        if hasattr(inputs,"to"): inputs=inputs.to(self.model.device)
        out=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,return_dict_in_generate=True,output_scores=True)
        input_len=int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated=out.sequences[:,input_len:] if input_len else out.sequences
        text_out=self.processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
        entropies=[]; confidences=[]
        for score in out.scores:
            p=torch.softmax(score.float(),dim=-1); entropies.append(float((-(p*p.clamp_min(1e-12).log()).sum(-1)).mean())); confidences.append(float(p.max(-1).values.mean()))
        confidence=sum(confidences)/len(confidences) if confidences else .5; raw_entropy=sum(entropies)/len(entropies) if entropies else .5
        entropy=1-math.exp(-raw_entropy/10.0)
        return ModelOutput(text_out,confidence,entropy,tuple([confidence]*len(images)),{"model_id":self.model_id,"revision":self.revision,"image_indices":idx,"prompt_variant":prompt_variant})
