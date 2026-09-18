#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from umirage.protected import SOURCES, sha256_file


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def collect_ids(paths: list[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"STOP: forbidden-ID file missing: {p}")
        if p.suffix == ".jsonl":
            for row in read_jsonl(p):
                for key in ("id", "original_id", "upstream_id"):
                    if row.get(key) is not None:
                        ids.add(str(row[key]))
                for value in row.get("upstream_ids", []) or []:
                    ids.add(str(value))
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(line.strip())
    return ids


def row_upstream_ids(row: dict) -> set[str]:
    out = {str(row[k]) for k in ("id", "original_id", "upstream_id") if row.get(k) is not None}
    out.update(str(x) for x in (row.get("upstream_ids") or []))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Select an origin-independent matched RQ1 source-selectivity manifest")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--forbidden-id-file", action="append", default=[])
    ap.add_argument("--target-per-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--allow-shortfall", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.candidates)
    forbidden = collect_ids(args.forbidden_id_file)
    eligible = []
    rejected = Counter()
    seen_ids: set[str] = set()

    for row in rows:
        source = str(row.get("source_label", ""))
        if source not in SOURCES:
            rejected["invalid_source"] += 1
            continue
        required = ("id", "group_id", "pair_id", "control_id", "origin", "task")
        if any(row.get(k) in (None, "") for k in required):
            rejected["missing_required_field"] += 1
            continue
        rid = str(row["id"])
        if rid in seen_ids:
            rejected["duplicate_id"] += 1
            continue
        ids = row_upstream_ids(row)
        if ids & forbidden:
            rejected["protected_or_dev_overlap"] += 1
            continue
        seen_ids.add(rid)
        eligible.append(row)

    # Deterministic ordering avoids selection after seeing model outcomes.
    eligible.sort(key=lambda r: (str(r["source_label"]), str(r["group_id"]), str(r["pair_id"]), str(r["id"])))
    selected = []
    counts = Counter()
    used_pairs: set[str] = set()
    for row in eligible:
        source = str(row["source_label"])
        pair_id = str(row["pair_id"])
        if counts[source] >= args.target_per_source:
            continue
        if pair_id in used_pairs:
            rejected["duplicate_pair"] += 1
            continue
        selected.append(row)
        counts[source] += 1
        used_pairs.add(pair_id)

    missing = {s: args.target_per_source - counts[s] for s in SOURCES if counts[s] < args.target_per_source}
    if missing and not args.allow_shortfall:
        raise SystemExit(f"STOP: insufficient eligible matched pairs: {missing}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "schema": "umirage.rq1.matched_manifest.v1",
        "seed": args.seed,
        "candidate_file": str(args.candidates),
        "candidate_sha256": sha256_file(args.candidates),
        "forbidden_files": {p: sha256_file(p) for p in args.forbidden_id_file},
        "target_per_source": args.target_per_source,
        "selected_counts": dict(counts),
        "selected_n": len(selected),
        "selected_groups": len({str(r["group_id"]) for r in selected}),
        "selected_origins": sorted({str(r["origin"]) for r in selected}),
        "selected_tasks": sorted({str(r["task"]) for r in selected}),
        "rejected": dict(rejected),
        "protected_overlap": 0,
        "output": str(out),
        "output_sha256": sha256_file(out),
    }
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
