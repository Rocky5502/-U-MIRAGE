#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

LABELS = ("clean", "visual", "integration", "reasoning")


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def prefix(group_id: str) -> str:
    parts = str(group_id).split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else str(group_id)


def oof_macro_f1(rows: list[dict], feature_name: str, folds: int, seed: int) -> dict:
    y = np.asarray([str(r["source_label"]) for r in rows])
    groups = np.asarray([str(r["group_id"]) for r in rows])
    features = []
    for r in rows:
        meta = r.get("metadata") or {}
        if feature_name == "origin":
            features.append({"origin": str(r.get("origin", meta.get("origin", "UNKNOWN")))})
        elif feature_name == "task":
            features.append({"task": str(r.get("task", meta.get("task", "UNKNOWN")))})
        elif feature_name == "group_prefix":
            features.append({"group_prefix": prefix(str(r["group_id"]))})
        elif feature_name == "metadata":
            safe = {
                str(k): str(v)
                for k, v in meta.items()
                if k not in {"source_label", "intervention_intent", "grounded_stage"}
                and isinstance(v, (str, int, float, bool))
            }
            features.append(safe or {"metadata": "EMPTY"})
        else:
            raise ValueError(feature_name)

    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    pred = np.empty(len(rows), dtype=object)
    for train, test in splitter.split(features, y, groups):
        vec = DictVectorizer(sparse=True)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        x_train = vec.fit_transform([features[i] for i in train])
        x_test = vec.transform([features[i] for i in test])
        clf.fit(x_train, y[train])
        pred[test] = clf.predict(x_test)
    return {
        "feature": feature_name,
        "macro_f1": float(f1_score(y, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "n": len(rows),
        "groups": len(set(groups)),
        "chance_reference": 0.25,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit RQ1 source-label leakage from non-scientific metadata")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=27)
    args = ap.parse_args()
    rows = [r for r in read_jsonl(args.input) if str(r.get("source_label")) in LABELS]
    if len({str(r["group_id"]) for r in rows}) < 2:
        raise SystemExit("STOP: need at least two groups for leakage audit")
    result = {
        "mode": "rq1_non_scientific_leakage_audit",
        "warning": "Any strong metadata/origin/task-only source prediction must be resolved before protected evaluation.",
        "baselines": [
            oof_macro_f1(rows, feature, args.folds, args.seed)
            for feature in ("origin", "task", "group_prefix", "metadata")
        ],
        "seed": args.seed,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
