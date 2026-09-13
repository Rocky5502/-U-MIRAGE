#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def blind_id(raw_id: str, salt: str) -> str:
    return "ann-" + hashlib.sha256(f"{salt}|{raw_id}".encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a blinded RQ1 source-stage annotation sheet")
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True, help="Private blind-ID key; do not give to annotators")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--salt", default="umirage-rq1-v1")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    sheet_path = Path(args.sheet)
    key_path = Path(args.key)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    key_rows = []
    with sheet_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "blind_id",
            "task",
            "question",
            "image_refs",
            "observed_answer",
            "annotated_stage",
            "non_target_preserved",
            "ambiguous",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            bid = blind_id(str(row["id"]), args.salt)
            images = row.get("images") or row.get("image_ids") or row.get("image_refs") or []
            writer.writerow({
                "blind_id": bid,
                "task": row.get("task", ""),
                "question": row.get("question", ""),
                "image_refs": json.dumps(images, ensure_ascii=False),
                "observed_answer": row.get("observed_answer", row.get("prediction", "")),
                "annotated_stage": "",
                "non_target_preserved": "",
                "ambiguous": "",
                "notes": "",
            })
            key_rows.append({
                "blind_id": bid,
                "id": str(row["id"]),
                "pair_id": row.get("pair_id"),
                "group_id": row.get("group_id"),
                "construction_source": row.get("source_label"),
            })

    key_path.write_text(json.dumps({"seed": args.seed, "rows": key_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(rows), "sheet": str(sheet_path), "key": str(key_path), "source_labels_in_sheet": False}, indent=2))


if __name__ == "__main__":
    main()
