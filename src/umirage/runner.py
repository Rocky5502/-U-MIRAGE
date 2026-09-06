from __future__ import annotations

from .pipeline import UMiragePipeline


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
                "correct": output.answer.strip().lower() == episode.answer.strip().lower(),
                "source_label": episode.source_label,
                "visual": state.visual,
                "integration": state.integration,
                "reasoning": state.reasoning,
                "risk": state.risk,
                "predicted_source": predicted_source,
                "action": state.action,
                "source_posterior": state.source_posterior,
            }
        )

    return rows
