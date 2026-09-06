from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetSpec:
    id: str
    role: str
    upstream: str
    redistribution: str

REGISTRY = {
    "mioh": DatasetSpec("mioh", "primary multi-image hallucination", "CVPR 2026 MIOH", "verify upstream license"),
    "vischainbench": DatasetSpec("vischainbench", "progressive multi-image reasoning", "eyehole/VisChainBench", "manifest-first"),
    "mihbench": DatasetSpec("mihbench", "controlled multi-image diagnostic", "MM-Hallu/MIHBench", "manifest-first"),
    "rest_plus": DatasetSpec("rest_plus", "cross-modal inconsistency", "Same-Content-Different-Answers", "manifest-first"),
    "hallusionbench": DatasetSpec("hallusionbench", "generalization", "HallusionBench", "verify per-media terms"),
    "mmmu": DatasetSpec("mmmu", "general utility", "MMMU-Benchmark/MMMU", "verify source-item rights"),
    "vlm_uqbench": DatasetSpec("vlm_uqbench", "UQ stress test", "VLM-UQBench", "preprint benchmark; verify terms"),
}

def registry_rows() -> list[dict[str, str]]:
    return [vars(REGISTRY[k]) for k in sorted(REGISTRY)]
