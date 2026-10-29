#!/usr/bin/env python3
"""Read-only audit of frozen RQ3 answer-output compatibility.

This is NOT an efficacy evaluator. It never modifies raw experiment data,
frozen correctness labels, prompts, policies, or scoring thresholds.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

POLICIES = ("umirage", "verify", "requery", "random_source", "no_extra_call")
BUDGETS = (0, 1, 2)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical_norm(value: object) -> str:
    """Original frozen exact-match normalization."""
    return " ".join(str(value).strip().lower().split())


def format_diagnostic(value: object) -> str:
    """Counterfactual formatting diagnostic ONLY: not a replacement metric."""
    s = canonical_norm(value)
    s = re.sub(r"^(final\s+answer|answer|response)\s*[:=]\s*", "", s)
    return s.strip(" \t\r\n\"'.,:;!?[](){}")


def audit(rows: list[dict], episodes: list[dict] | None = None) -> dict:
    if not rows:
        raise ValueError("Empty input is not a successful audit")

    by = collections.defaultdict(list)
    keyset = set()
    ids = set()
    for r in rows:
        key = (str(r["id"]), str(r["policy"]), int(r["budget"]))
        if key in keyset:
            raise ValueError(f"Duplicate episode/policy/budget: {key}")
        keyset.add(key)
        ids.add(key[0])
        by[(key[1], key[2])].append(r)

    expected = {(p, b) for p in POLICIES for b in BUDGETS}
    complete = all(
        {(p, b) for e, p, b in keyset if e == eid} == expected for eid in ids
    )
    ep = None
    if episodes is not None:
        ep = {str(r["id"]): r for r in episodes}
        if len(ep) != len(episodes) or set(ep) != ids:
            raise ValueError("Episode IDs missing, duplicate, or mismatched")

    metrics = {}
    for (policy, budget), rr in sorted(by.items()):
        texts = [str(x.get("final_answer") or "") for x in rr]
        strict = [canonical_norm(x) for x in texts]
        relaxed = [format_diagnostic(x) for x in texts]
        correct = sum(bool(x["correct_after"]) for x in rr)
        cell = {
            "n": len(rr),
            "recorded_correct": correct,
            "recorded_accuracy": correct / len(rr),
            "empty_answer_count": sum(not x.strip() for x in texts),
            "unique_answer_count": len(set(strict)),
            "answer_prefix_count": sum(
                bool(re.match(r"^(final\s+answer|answer|response)\s*[:=]", x, re.I))
                for x in texts
            ),
            "yes_no_prefix_count": sum(
                bool(re.match(r"^\s*(yes|no)\b", x, re.I)) for x in texts
            ),
            "mean_answer_chars": sum(map(len, texts)) / len(rr),
            "mean_extra_tokens": sum(float(x.get("extra_tokens", 0)) for x in rr) / len(rr),
        }
        if ep is not None:
            gold = [str(ep[str(x["id"])]["answer"]) for x in rr]
            cell["canonical_recomputed_matches"] = sum(
                a == canonical_norm(g) for a, g in zip(strict, gold)
            )
            cell["format_only_counterfactual_matches"] = sum(
                a == format_diagnostic(g) for a, g in zip(relaxed, gold)
            )
            cell["frozen_label_reproduced"] = cell["canonical_recomputed_matches"] == correct
        metrics[f"{policy}|{budget}"] = cell

    nonzero = [f"{p}|{b}" for p in POLICIES[:4] for b in (1, 2)]
    allzero = (
        metrics.get("umirage|0", {}).get("recorded_accuracy", 0) > 0
        and all(metrics.get(k, {}).get("recorded_correct") == 0 for k in nonzero)
    )
    return {
        "schema": "umirage.rq3.output_contract_audit.v1",
        "evidence_level": "development_diagnostic_only",
        "total_rows": len(rows),
        "unique_ids": len(ids),
        "full_15_cell_grid": complete,
        "all_nonzero_budget_arms_zero": allzero,
        "efficacy_interpretation": "BLOCKED" if allzero else "REVIEW_REQUIRED",
        "frozen_correctness_labels_modified": False,
        "counterfactual_used_as_efficacy": False,
        "per_policy_budget": metrics,
        "notes": [
            "A format-only match is a diagnostic and must not replace frozen outcomes.",
            "All-zero repair arms may indicate an output/adapter/scoring contract failure.",
            "No raw generated answers are included in this report."
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--episodes", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = audit(
        read_jsonl(args.input),
        read_jsonl(args.episodes) if args.episodes else None,
    )
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.episodes:
        report["episodes_sha256"] = hashlib.sha256(args.episodes.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "rows": report["total_rows"],
        "ids": report["unique_ids"],
        "complete_grid": report["full_15_cell_grid"],
        "all_zero_repair": report["all_nonzero_budget_arms_zero"],
        "efficacy_interpretation": report["efficacy_interpretation"],
        "report": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
