from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VALID_SOURCES = {"clean", "visual", "integration", "reasoning"}


@dataclass(frozen=True)
class Episode:
    id: str
    images: tuple[str, ...]
    question: str
    answer: str
    source_label: str = "clean"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_label not in VALID_SOURCES:
            raise ValueError(f"invalid source_label={self.source_label!r}")
        if not self.images:
            raise ValueError("episode must contain at least one image reference")


@dataclass(frozen=True)
class ModelOutput:
    answer: str
    confidence: float
    token_entropy: float
    evidence_scores: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyState:
    visual: float
    integration: float
    reasoning: float
    risk: float
    source_posterior: dict[str, float]
    action: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
