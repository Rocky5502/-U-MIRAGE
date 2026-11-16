from __future__ import annotations

from typing import Protocol

from ..schema import Episode, ModelOutput


class MultiImageAdapter(Protocol):
    name: str

    def answer(
        self,
        episode: Episode,
        *,
        image_indices: tuple[int, ...] | None = None,
        order: tuple[int, ...] | None = None,
        prompt_variant: int = 0,
        prompt_override: str | None = None,
    ) -> ModelOutput: ...
