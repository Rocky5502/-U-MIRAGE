#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from umirage.protected import sha256_file


def _walk_strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk_strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk_strings(v)
    elif isinstance(x, str):
        yield x


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze protected U-MIRAGE experiment protocol")
    ap.add_argument("--config", default="configs/protected_experiments_v1.yaml")
    ap.add_argument("--output", default="provenance/protected/PROTECTED_PROTOCOL_FREEZE_v1.json")
    ap.add_argument(
        "--tracked",
        nargs="*",
        default=[
            "src/umirage/protected.py",
            "src/umirage/propagation.py",
            "src/umirage/adapters/base.py",
            "src/umirage/adapters/qwen_vl.py",
            "src/umirage/adapters/transformers_vlm.py",
            "experiments/protected/build_rq1_matched_manifest.py",
            "experiments/protected/make_rq1_annotation_sheet.py",
            "experiments/protected/score_rq1_annotations.py",
            "experiments/protected/evaluate_rq1.py",
            "experiments/protected/audit_leakage.py",
            "experiments/protected/evaluate_rq2.py",
            "experiments/protected/run_rq3_actions.py",
            "experiments/protected/evaluate_rq3.py",
        ],
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    bad = [s for s in _walk_strings(cfg) if "TBD" in s.upper()]
    if bad:
        raise SystemExit(f"STOP: unresolved TBD values in protected config: {bad[:5]}")

    if cfg["protected_partitions"]["calibration_allowed_before_freeze"]:
        raise SystemExit("STOP: calibration must remain closed before protocol freeze")
    if cfg["protected_partitions"]["holdout_allowed_before_freeze"]:
        raise SystemExit("STOP: holdout must remain closed before protocol freeze")
    if int(cfg["protected_partitions"]["holdout_runs"]) != 1:
        raise SystemExit("STOP: final holdout must be one-shot")

    tracked = [cfg_path, *(Path(x) for x in args.tracked)]
    missing = [str(p) for p in tracked if not p.exists()]
    if missing:
        raise SystemExit(f"STOP: missing tracked protocol files: {missing}")

    manifest = {
        "schema": "umirage.protected.freeze.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PRE_RESULT_FROZEN",
        "files": {str(p): sha256_file(p) for p in tracked},
        "protected_state": {
            "calibration_opened": False,
            "holdout_opened": False,
            "holdout_runs_completed": 0,
        },
        "config": cfg,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "status": manifest["status"], "files": len(tracked)}, indent=2))


if __name__ == "__main__":
    main()
