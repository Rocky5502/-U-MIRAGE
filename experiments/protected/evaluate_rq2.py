#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from umirage.evaluation import auprc, aurc, auroc, brier, ece, risk_coverage
from umirage.protected import PropagationWeights, no_propagation_risk, propagated_risk, scores_from_row


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def scored_rows(rows: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    no_w = cfg["rq2"]["no_propagation_weights"]
    pw = PropagationWeights(**cfg["rq2"]["propagation_weights"])
    control, prop = [], []
    for r in rows:
        if "correct" not in r or "group_id" not in r:
            raise KeyError("RQ2 rows require correct and group_id")
        scores = scores_from_row(r)
        base = {"correct": bool(r["correct"]), "group_id": str(r["group_id"]), "id": r.get("id")}
        control.append({**base, "risk": no_propagation_risk(scores, no_w)})
        prop.append({**base, "risk": propagated_risk(scores, pw)})
    return control, prop


def summary(rows: list[dict], coverages) -> dict:
    return {
        "auroc": auroc(rows),
        "aurc": aurc(rows),
        "auprc": auprc(rows),
        "brier": brier(rows),
        "ece": ece(rows),
        "risk_coverage": risk_coverage(rows, coverages=tuple(coverages)),
    }


def group_bootstrap(control: list[dict], prop: list[dict], n: int, seed: int) -> dict:
    c_by = defaultdict(list)
    p_by = defaultdict(list)
    for r in control:
        c_by[r["group_id"]].append(r)
    for r in prop:
        p_by[r["group_id"]].append(r)
    groups = sorted(set(c_by) & set(p_by))
    if not groups:
        raise SystemExit("STOP: no shared groups")
    rng = np.random.default_rng(seed)
    d_auc, d_aurc = [], []
    for _ in range(n):
        sample = list(rng.choice(groups, size=len(groups), replace=True))
        c = [r for g in sample for r in c_by[str(g)]]
        p = [r for g in sample for r in p_by[str(g)]]
        ca, pa = auroc(c), auroc(p)
        cr, pr = aurc(c), aurc(p)
        if np.isfinite(ca) and np.isfinite(pa):
            d_auc.append(pa - ca)
        if np.isfinite(cr) and np.isfinite(pr):
            d_aurc.append(cr - pr)
    if not d_auc or not d_aurc:
        raise SystemExit("STOP: bootstrap lacks both classes in valid resamples")
    q = lambda xs: [float(np.quantile(xs, .025)), float(np.quantile(xs, .975))]
    return {
        "delta_auroc_prop_minus_control": {
            "point": auroc(prop) - auroc(control),
            "ci95": q(d_auc),
            "valid_resamples": len(d_auc),
        },
        "favorable_delta_aurc_control_minus_prop": {
            "point": aurc(control) - aurc(prop),
            "ci95": q(d_aurc),
            "valid_resamples": len(d_aurc),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Protected RQ2 exact-same-signal propagation evaluator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="configs/protected_experiments_v1.yaml")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    n = int(args.bootstrap or cfg["bootstrap_resamples"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    rows = read_jsonl(args.input)
    control, prop = scored_rows(rows, cfg)
    coverages = cfg["rq2"]["risk_coverages"]

    result = {
        "mode": "protected_exact_same_signal_propagation",
        "n": len(rows),
        "groups": len({r["group_id"] for r in control}),
        "control": summary(control, coverages),
        "propagation": summary(prop, coverages),
        "paired_group_bootstrap": group_bootstrap(control, prop, n, seed),
        "same_source_signals": True,
        "seed": seed,
        "bootstrap_resamples": n,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
