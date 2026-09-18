#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from umirage.io import load_jsonl


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def norm(x: str) -> str:
    s = str(x or "").strip().lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"^\s*(final\s+answer|answer)\s*[:\-]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" \t\r\n.,;:!?\"'()[]{}")


def relaxed_parse(answer: str, gold: str) -> tuple[str | None, bool]:
    raw = str(answer or "").strip()
    if not raw:
        return None, False
    g = norm(gold)
    p = norm(raw)
    if p == g:
        return p, True
    if g in {"yes", "no"}:
        m = re.match(r"^(yes|no)\b", p)
        if m:
            value = m.group(1)
            return value, value == g
        return p, False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", g):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", p)
        if len(nums) == 1:
            try:
                return nums[0], float(nums[0]) == float(g)
            except ValueError:
                pass
    return p, False


def answer_shape(answer: str) -> str:
    s = str(answer or "").strip()
    if not s:
        return "empty"
    if "\n" in s:
        return "multiline"
    if len(s.split()) == 1:
        return "single_token"
    return "single_line_multi_token"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-hoc diagnostic-only audit for completed RQ3 repair outputs."
    )
    ap.add_argument("--raw", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.raw)
    episodes = {ep.id: ep for ep in load_jsonl(args.episodes)}
    if not rows:
        raise SystemExit("STOP: no RQ3 rows")
    if any(str(r["id"]) not in episodes for r in rows):
        raise SystemExit("STOP: raw rows contain IDs absent from episode file")

    by = defaultdict(list)
    for row in rows:
        by[(str(row["policy"]), int(row["budget"]))].append(row)

    cells = {}
    for (policy, budget), rr in sorted(by.items()):
        if budget == 0:
            continue
        exact = 0
        relaxed = 0
        parseable = 0
        shapes = defaultdict(int)
        examples = []
        for row in rr:
            gold = str(episodes[str(row["id"])].answer)
            answer = str(row.get("final_answer", ""))
            shapes[answer_shape(answer)] += 1
            exact += int(norm(answer) == norm(gold))
            parsed, ok = relaxed_parse(answer, gold)
            parseable += int(parsed is not None)
            relaxed += int(ok)
            if len(examples) < 5:
                examples.append({
                    "id": str(row["id"]),
                    "gold": gold,
                    "answer": answer[:300],
                    "shape": answer_shape(answer),
                    "frozen_exact_correct": bool(row.get("correct_after")),
                    "diagnostic_relaxed_correct": bool(ok),
                })
        n = len(rr)
        cells[f"{policy}|{budget}"] = {
            "n": n,
            "frozen_exact_accuracy_recomputed": exact / n,
            "diagnostic_relaxed_accuracy": relaxed / n,
            "diagnostic_parseable_rate": parseable / n,
            "answer_shapes": dict(sorted(shapes.items())),
            "examples": examples,
        }

    result = {
        "schema": "umirage.rq3.output_validity_audit.v1",
        "status": "posthoc_diagnostic_only",
        "warning": (
            "The relaxed parser is a post-hoc validity diagnostic. It must not replace "
            "the frozen RQ3 primary estimator or be reported as confirmatory efficacy."
        ),
        "raw": str(args.raw),
        "episodes": str(args.episodes),
        "cells": cells,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("RQ3_OUTPUT_VALIDITY_AUDIT=PASS")
    for key, x in cells.items():
        print(
            f"{key:22s} "
            f"frozen_exact={x['frozen_exact_accuracy_recomputed']:.4f} "
            f"diagnostic_relaxed={x['diagnostic_relaxed_accuracy']:.4f} "
            f"parseable={x['diagnostic_parseable_rate']:.4f} "
            f"shapes={x['answer_shapes']}"
        )
    print("POSTHOC_DIAGNOSTIC_ONLY=TRUE")


if __name__ == "__main__":
    main()
