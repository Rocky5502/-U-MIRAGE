#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

POLICIES = ("umirage", "verify", "requery", "random_source", "no_extra_call")
SOURCES = ("visual", "integration", "reasoning")
ACTIONS = ("reinspect_image", "compare_images", "verify_reasoning")


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(str(r["policy"]), int(r["budget"]))].append(r)
    out = {}
    for (policy, budget), rr in sorted(by.items()):
        acc = float(np.mean([bool(r["correct_after"]) for r in rr]))
        base = float(np.mean([bool(r["correct_before"]) for r in rr]))
        recovered = sum((not bool(r["correct_before"])) and bool(r["correct_after"]) for r in rr)
        eligible = sum(not bool(r["correct_before"]) for r in rr)
        tokens = sum(float(r.get("extra_tokens", 0)) for r in rr)
        latency = sum(float(r.get("extra_latency_s", 0)) for r in rr)
        out[f"{policy}|{budget}"] = {
            "n": len(rr),
            "base_accuracy": base,
            "post_accuracy": acc,
            "accuracy_gain": acc - base,
            "recovered_errors": recovered,
            "eligible_errors": eligible,
            "recovery_rate": recovered / eligible if eligible else float("nan"),
            "extra_tokens": tokens,
            "extra_latency_s": latency,
            "tokens_per_recovered_error": tokens / recovered if recovered else None,
            "latency_per_recovered_error": latency / recovered if recovered else None,
        }
    return out


def source_repair_matrix(rows: list[dict], budget: int = 1) -> dict:
    matrix = {s: {a: {"recovered": 0, "eligible": 0} for a in ACTIONS} for s in SOURCES}
    for r in rows:
        if int(r["budget"]) != budget:
            continue
        source = str(r.get("source_label"))
        action = str(r.get("executed_action"))
        if source not in SOURCES or action not in ACTIONS or bool(r["correct_before"]):
            continue
        cell = matrix[source][action]
        cell["eligible"] += 1
        cell["recovered"] += int(bool(r["correct_after"]))
    for source in SOURCES:
        for action in ACTIONS:
            cell = matrix[source][action]
            cell["recovery_rate"] = cell["recovered"] / cell["eligible"] if cell["eligible"] else None
    return matrix


def paired_group_bootstrap(rows, policy_a, policy_b, budget, n, seed):
    keep = [r for r in rows if int(r["budget"]) == budget and r["policy"] in {policy_a, policy_b}]
    keyed = defaultdict(dict)
    for r in keep:
        keyed[(str(r["group_id"]), str(r["id"]))][str(r["policy"])] = r
    pairs = [v for v in keyed.values() if policy_a in v and policy_b in v]
    groups = sorted({str(v[policy_a]["group_id"]) for v in pairs})
    by_group = defaultdict(list)
    for v in pairs:
        by_group[str(v[policy_a]["group_id"])].append(v)
    if not groups:
        return {"n_pairs": 0}
    point = float(np.mean([int(v[policy_a]["correct_after"]) - int(v[policy_b]["correct_after"]) for v in pairs]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        sample_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = [v for g in sample_groups for v in by_group[str(g)]]
        boot.append(float(np.mean([int(v[policy_a]["correct_after"]) - int(v[policy_b]["correct_after"]) for v in sample])))
    return {"n_pairs": len(pairs), "accuracy_difference": point, "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Protected RQ3 matched-budget action evaluator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=27)
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    unknown = sorted({str(r["policy"]) for r in rows} - set(POLICIES))
    if unknown:
        raise SystemExit(f"STOP: unknown policies: {unknown}")
    result = {
        "mode": "protected_matched_budget_actions",
        "summary": summarize(rows),
        "source_repair_matrix_budget1": source_repair_matrix(rows, 1),
        "paired_primary": {
            "umirage_vs_verify_b1": paired_group_bootstrap(rows, "umirage", "verify", 1, args.bootstrap, args.seed),
            "umirage_vs_requery_b1": paired_group_bootstrap(rows, "umirage", "requery", 1, args.bootstrap, args.seed),
            "umirage_vs_verify_b2": paired_group_bootstrap(rows, "umirage", "verify", 2, args.bootstrap, args.seed),
            "umirage_vs_requery_b2": paired_group_bootstrap(rows, "umirage", "requery", 2, args.bootstrap, args.seed),
        },
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
