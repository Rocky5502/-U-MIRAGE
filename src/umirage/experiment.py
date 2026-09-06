from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evaluation import risk_coverage, summarize
from .io import dump_jsonl, load_jsonl, sha256_file
from .provenance import write_manifest
from .runner import run_episodes


def _json_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_experiment(
    input_jsonl: str | Path,
    adapter,
    output_dir: str | Path,
    config: dict | None = None,
    limit: int | None = None,
) -> dict:
    input_jsonl = Path(input_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})

    episodes = load_jsonl(input_jsonl)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        episodes = episodes[:limit]

    rows = run_episodes(episodes, adapter)
    dump_jsonl(output_dir / "predictions.jsonl", rows)

    metrics = summarize(rows)
    metrics["synthetic_only"] = bool(getattr(adapter, "name", "").startswith("mock"))
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    rc = risk_coverage(rows)
    (output_dir / "risk_coverage.json").write_text(
        json.dumps(rc, indent=2, sort_keys=True), encoding="utf-8"
    )

    provenance = {
        **config,
        "adapter": getattr(adapter, "name", adapter.__class__.__name__),
        "input_jsonl": str(input_jsonl),
        "input_sha256": sha256_file(input_jsonl),
        "requested_limit": limit,
        "config_sha256": _json_hash(config),
        "n_examples": len(rows),
    }
    write_manifest(output_dir / "provenance.json", provenance)
    return metrics
