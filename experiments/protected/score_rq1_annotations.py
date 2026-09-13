#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

VALID = {"clean", "visual", "integration", "reasoning"}
TRUE = {"1", "true", "yes", "y"}
FALSE = {"0", "false", "no", "n"}


def read_csv(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        bid = str(row.get("blind_id", "")).strip()
        if not bid or bid in out:
            raise SystemExit(f"STOP: missing/duplicate blind_id in {path}: {bid!r}")
        out[bid] = row
    return out


def parse_bool(value: str, field: str, bid: str) -> bool:
    x = str(value).strip().lower()
    if x in TRUE:
        return True
    if x in FALSE:
        return False
    raise SystemExit(f"STOP: invalid {field} for {bid}: {value!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score two blinded RQ1 source-stage annotation sheets")
    ap.add_argument("--annotator-a", required=True)
    ap.add_argument("--annotator-b", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    a = read_csv(args.annotator_a)
    b = read_csv(args.annotator_b)
    key_payload = json.loads(Path(args.key).read_text(encoding="utf-8"))
    key = {str(r["blind_id"]): r for r in key_payload["rows"]}

    shared = sorted(set(a) & set(b) & set(key))
    if not shared:
        raise SystemExit("STOP: no shared blinded IDs")
    if set(a) != set(b):
        raise SystemExit("STOP: annotator sheets do not cover identical blind IDs")

    labels_a, labels_b = [], []
    verified = []
    rejected = []
    for bid in shared:
        sa = str(a[bid].get("annotated_stage", "")).strip().lower()
        sb = str(b[bid].get("annotated_stage", "")).strip().lower()
        if sa not in VALID or sb not in VALID:
            raise SystemExit(f"STOP: invalid stage label for {bid}: {sa!r}, {sb!r}")
        labels_a.append(sa)
        labels_b.append(sb)
        preserve_a = parse_bool(a[bid].get("non_target_preserved", ""), "non_target_preserved", bid)
        preserve_b = parse_bool(b[bid].get("non_target_preserved", ""), "non_target_preserved", bid)
        ambiguous_a = parse_bool(a[bid].get("ambiguous", ""), "ambiguous", bid)
        ambiguous_b = parse_bool(b[bid].get("ambiguous", ""), "ambiguous", bid)

        accepted = sa == sb and preserve_a and preserve_b and not ambiguous_a and not ambiguous_b
        row = {
            **key[bid],
            "annotator_a_stage": sa,
            "annotator_b_stage": sb,
            "verified_stage": sa if accepted else None,
            "accepted": accepted,
            "non_target_preserved_a": preserve_a,
            "non_target_preserved_b": preserve_b,
            "ambiguous_a": ambiguous_a,
            "ambiguous_b": ambiguous_b,
        }
        (verified if accepted else rejected).append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in verified:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "n_double_annotated": len(shared),
        "cohen_kappa_stage": float(cohen_kappa_score(labels_a, labels_b)),
        "accepted_n": len(verified),
        "rejected_n": len(rejected),
        "accepted_rate": len(verified) / len(shared),
        "acceptance_rule": "stage agreement AND both preserve non-target factors AND neither ambiguous",
        "rejected": rejected,
    }
    sp = Path(args.summary)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
