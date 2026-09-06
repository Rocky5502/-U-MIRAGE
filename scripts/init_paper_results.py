from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_SCHEMAS: dict[str, list[str]] = {
    "source_attribution.csv": [
        "run_id", "model", "dataset", "method", "visual_f1", "integration_f1",
        "reasoning_f1", "clean_f1", "macro_f1", "image_localization", "seed",
    ],
    "source_selectivity.csv": [
        "run_id", "model", "source", "perturbation_family", "severity",
        "delta_visual", "delta_integration", "delta_reasoning", "selectivity",
        "negative_control", "seed",
    ],
    "failure_calibration.csv": [
        "run_id", "model", "dataset", "method", "accuracy", "auroc", "auprc",
        "ece", "brier", "aurc", "coverage_at_target_risk", "seed",
    ],
    "intervention.csv": [
        "run_id", "model", "dataset", "policy", "budget_calls", "accuracy",
        "hallucination_rate", "aurc", "coverage", "extra_calls", "vision_tokens",
        "generation_tokens", "latency_ms", "seed",
    ],
    "ablation.csv": [
        "run_id", "model", "dataset", "variant", "macro_f1", "auroc", "aurc",
        "accuracy", "seed",
    ],
    "transfer.csv": [
        "run_id", "train_model", "test_model", "train_dataset", "test_dataset",
        "method", "macro_f1", "auroc", "aurc", "relative_degradation",
    ],
    "risk_coverage.csv": [
        "run_id", "model", "dataset", "method", "coverage", "selective_risk", "seed",
    ],
    "source_confusion.csv": [
        "run_id", "model", "dataset", "true_source", "predicted_source", "count", "rate",
    ],
}

JSON_TEMPLATES = {
    "statistics.json": {"schema_version": 1, "comparisons": []},
    "compute.json": {"schema_version": 1, "runs": []},
}


def initialize(output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fields in CSV_SCHEMAS.items():
        path = output_dir / name
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --force only for an empty-template reset")
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fields)

    for name, payload in JSON_TEMPLATES.items():
        path = output_dir / name
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --force only for an empty-template reset")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "empty_pre_result_template",
        "csv_files": sorted(CSV_SCHEMAS),
        "json_files": sorted(JSON_TEMPLATES),
        "rule": "No manuscript value may be populated before a frozen run artifact exists.",
    }
    (output_dir / "paper_results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create header-only U-MIRAGE CVPR paper-result artifacts with no empirical values."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    initialize(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
