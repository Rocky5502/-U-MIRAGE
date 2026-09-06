from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib, math
from .core import Episode, ModelOutput

class MultiImageAdapter(ABC):
    @abstractmethod
    def answer(self, episode: Episode, image_indices=None, order=None, prompt_variant: int=0) -> ModelOutput: ...

class MockMultiImageAdapter(MultiImageAdapter):
    """Deterministic synthetic adapter for CI; never a scientific baseline."""
    def answer(self, episode, image_indices=None, order=None, prompt_variant=0):
        meta=episode.metadata.get("synthetic",{}); risk=float(meta.get("risk",0.2)); source=episode.source_label
        key=f"{episode.id}|{image_indices}|{order}|{prompt_variant}".encode(); jitter=int(hashlib.sha256(key).hexdigest()[:4],16)/65535
        perturb=False
        if source=="integration" and (image_indices is not None or order is not None): perturb=True
        if source=="reasoning" and prompt_variant: perturb=True
        wrong=meta.get("wrong_answer","unknown")
        answer=wrong if (risk>.55 and (perturb or source in {"visual","reasoning"})) else episode.answer
        conf=max(.05,min(.95,1-risk*.75+.05*(jitter-.5)))
        entropy=max(.01,min(1.0,risk*(.85 if source=="reasoning" else .55)+.08*jitter))
        evidence=tuple(max(.02,min(.98,1-risk*(.9 if source=="visual" else .45))) for _ in episode.images)
        return ModelOutput(answer,conf,entropy,evidence,{"synthetic":True})

class TransformersVLMAdapter(MultiImageAdapter):
    """Optional generic local-HF boundary. Processor/model API differences remain isolated here."""
    def __init__(self, model_id: str, revision: str|None=None, device_map="auto", dtype="auto"):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise RuntimeError("Install with: pip install -e .[vlm]") from e
        self.processor=AutoProcessor.from_pretrained(model_id,revision=revision)
        self.model=AutoModelForImageTextToText.from_pretrained(model_id,revision=revision,device_map=device_map,torch_dtype=dtype)
    def answer(self, episode, image_indices=None, order=None, prompt_variant=0):
        from PIL import Image
        idx=tuple(range(len(episode.images))) if image_indices is None else tuple(image_indices)
        if order is not None: idx=tuple(order)
        images=[Image.open(episode.images[i]).convert("RGB") for i in idx]
        prompt=episode.question if prompt_variant==0 else f"Variant {prompt_variant}: {episode.question}"
        inputs=self.processor(text=prompt,images=images,return_tensors="pt").to(self.model.device)
        out=self.model.generate(**inputs,max_new_tokens=256,do_sample=False,return_dict_in_generate=True,output_scores=True)
        text=self.processor.batch_decode(out.sequences,skip_special_tokens=True)[0]
        import torch
        ents=[]; conf=[]
        for score in out.scores:
            p=torch.softmax(score.float(),dim=-1); ents.append(float((-(p*p.clamp_min(1e-12).log()).sum(-1)).mean()))
            conf.append(float(p.max(-1).values.mean()))
        c=sum(conf)/len(conf) if conf else .5; h=sum(ents)/len(ents) if ents else .5
        h=1-math.exp(-h/10)
        return ModelOutput(text,c,h,tuple([c]*len(images)),{"model_id":self.model.name_or_path})
