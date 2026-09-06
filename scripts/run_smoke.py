from __future__ import annotations

import csv
import json
from pathlib import Path

from umirage.adapters.mock import MockMultiImageAdapter
from umirage.evaluation import accuracy, brier, risk_coverage, source_accuracy
from umirage.io import dump_jsonl, load_jsonl
from umirage.runner import run_episodes


def main() -> None:
    source_path = Path("data/examples/mirage_demo.jsonl")
    if not source_path.exists():
        from make_demo_data import main as make_demo

        make_demo()

    rows = run_episodes(load_jsonl(source_path), MockMultiImageAdapter())
    output_dir = Path("artifacts/smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "predictions.jsonl", rows)

    metrics = {
        "synthetic_only": True,
        "accuracy": accuracy(rows),
        "brier": brier(rows),
        "source_accuracy": source_accuracy(rows),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "risk_coverage.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["coverage", "selective_risk", "n"],
        )
        writer.writeheader()
        writer.writerows(risk_coverage(rows))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
