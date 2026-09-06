from __future__ import annotations

import argparse
import json

from .adapters.mock import MockMultiImageAdapter
from .datasets import registry_rows
from .experiment import run_experiment


def main(argv=None):
    p = argparse.ArgumentParser(prog="umirage")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("registry", aliases=["inspect-registry"])

    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True, help="canonical JSONL path")
    run.add_argument("--output", required=True, help="output directory")
    args = p.parse_args(argv)

    if args.cmd in {"registry", "inspect-registry"}:
        print(json.dumps(registry_rows(), indent=2))
        return 0

    metrics = run_experiment(args.dataset, MockMultiImageAdapter(), args.output, {"cli": True})
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
