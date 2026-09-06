from __future__ import annotations

import argparse
import json
from pathlib import Path

from umirage.evaluation import accuracy, aurc, auprc, auroc, brier, ece, source_macro_f1
from umirage.io import load_jsonl
from umirage.statistics import mcnemar_exact_p, paired_accuracy_counts, paired_bootstrap_delta


METRICS = {
    "accuracy": accuracy,
    "auroc": auroc,
    "auprc": auprc,
    "ece": ece,
    "brier": brier,
    "aurc": aurc,
    "source_macro_f1": source_macro_f1,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired statistical comparison of two U-MIRAGE predictions.jsonl files."
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    left = load_jsonl(args.left)
    right = load_jsonl(args.right)
    result = {
        "left": str(Path(args.left)),
        "right": str(Path(args.right)),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "alpha": args.alpha,
        "paired_accuracy_counts": paired_accuracy_counts(left, right),
        "mcnemar_exact_p": mcnemar_exact_p(left, right),
        "metrics": {},
    }

    for name, metric in METRICS.items():
        try:
            result["metrics"][name] = paired_bootstrap_delta(
                left,
                right,
                metric,
                samples=args.samples,
                seed=args.seed,
                alpha=args.alpha,
            )
        except ValueError as exc:
            result["metrics"][name] = {"status": "not_estimable", "reason": str(exc)}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
