from __future__ import annotations

from .pipeline import UMiragePipeline


def _normalized(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_episodes(episodes, adapter) -> list[dict]:
    pipeline = UMiragePipeline(adapter)
    rows = []

    for episode in episodes:
        output, state = pipeline.analyze(episode)
        predicted_source = max(state.source_posterior, key=state.source_posterior.get)
        rows.append(
            {
                "id": episode.id,
                "prediction": output.answer,
                "gold": episode.answer,
                "correct": _normalized(output.answer) == _normalized(episode.answer),
                "source_label": episode.source_label,
                "visual": state.visual,
                "integration": state.integration,
                "reasoning": state.reasoning,
                "risk": state.risk,
                "predicted_source": predicted_source,
                "action": state.action,
                "source_posterior": state.source_posterior,
                "model_confidence": float(output.confidence),
                "token_entropy": float(output.token_entropy),
                "evidence_scores": [float(value) for value in output.evidence_scores],
                "model_metadata": dict(output.metadata),
                "n_images": len(episode.images),
                "episode_metadata": dict(episode.metadata),
                "diagnostics": state.diagnostics,
            }
        )

    return rows
