#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from umirage.protected import LABELS, SOURCES, scores_from_row, selectivity


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def percentile_ci(values: list[float]) -> list[float]:
    a = np.asarray(values, dtype=float)
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def group_bootstrap_selectivity(pairs: list[dict], n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = sorted({str(p["group_id"]) for p in pairs})
    by_group = defaultdict(list)
    for p in pairs:
        by_group[str(p["group_id"])].append(p)
    boots = []
    for _ in range(n):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        vals = [x["selectivity"] for g in sampled for x in by_group[str(g)]]
        boots.append(float(np.mean(vals)))
    return {
        "mean": float(np.mean([p["selectivity"] for p in pairs])),
        "ci95": percentile_ci(boots),
    }


def controlled(rows: list[dict], nboot: int, seed: int) -> dict:
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[str(r["pair_id"])].append(r)

    per_source: dict[str, list[dict]] = {s: [] for s in SOURCES}
    rejected = []
    for pid, rr in by_pair.items():
        controls = [r for r in rr if r.get("condition") == "control"]
        interventions = [r for r in rr if r.get("condition") == "intervention"]
        if len(controls) != 1 or len(interventions) != 1:
            rejected.append(pid)
            continue
        x = interventions[0]
        source = str(x["source_label"])
        if source not in SOURCES:
            rejected.append(pid)
            continue
        sel, delta = selectivity(scores_from_row(controls[0]), scores_from_row(x), source)
        per_source[source].append({
            "pair_id": pid,
            "group_id": str(x["group_id"]),
            "selectivity": float(sel),
            "delta": delta,
        })

    out = {"mode": "controlled_selectivity", "rejected_pairs": rejected, "sources": {}}
    for source in SOURCES:
        pairs = per_source[source]
        vals = [p["selectivity"] for p in pairs]
        if not vals:
            out["sources"][source] = {"n": 0}
            continue
        out["sources"][source] = {
            "n": len(vals),
            "mean_selectivity": float(np.mean(vals)),
            "median_selectivity": float(np.median(vals)),
            "positive_rate": float(np.mean(np.asarray(vals) > 0)),
            "group_bootstrap": group_bootstrap_selectivity(pairs, nboot, seed),
        }
    return out


def natural(rows: list[dict], clean_threshold: float, nboot: int, seed: int) -> dict:
    y_true, y_pred = [], []
    valid_rows = []
    for r in rows:
        true = str(r["source_label"])
        if true not in LABELS:
            continue
        scores = scores_from_row(r)
        d = scores.as_dict()
        pred = "clean" if max(d.values()) < clean_threshold else max(SOURCES, key=lambda k: d[k])
        y_true.append(true)
        y_pred.append(pred)
        valid_rows.append(r)

    if not y_true:
        raise SystemExit("STOP: no valid natural source labels")
    labels = list(LABELS)
    base = {
        "n": len(y_true),
        "clean_threshold": clean_threshold,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {
            label: float(recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0))
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    groups = sorted({str(r["group_id"]) for r in valid_rows})
    by_group = defaultdict(list)
    for r in valid_rows:
        by_group[str(r["group_id"])].append(r)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(nboot):
        sample = [r for g in rng.choice(groups, size=len(groups), replace=True) for r in by_group[str(g)]]
        yt, yp = [], []
        for r in sample:
            scores = scores_from_row(r)
            d = scores.as_dict()
            yt.append(str(r["source_label"]))
            yp.append("clean" if max(d.values()) < clean_threshold else max(SOURCES, key=lambda k: d[k]))
        boots.append(float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)))
    base["macro_f1_ci95"] = percentile_ci(boots)
    return {"mode": "natural_attribution", **base}


def main() -> None:
    ap = argparse.ArgumentParser(description="Protected RQ1 evaluator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", choices=["controlled", "natural"], required=True)
    ap.add_argument("--clean-threshold", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.input)

    if args.mode == "controlled":
        result = controlled(rows, args.bootstrap, args.seed)
    else:
        if args.clean_threshold is None:
            raise SystemExit("STOP: --clean-threshold must be frozen before natural RQ1 evaluation")
        result = natural(rows, args.clean_threshold, args.bootstrap, args.seed)

    result["protected_claim"] = True
    result["seed"] = args.seed
    result["bootstrap_resamples"] = args.bootstrap
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
