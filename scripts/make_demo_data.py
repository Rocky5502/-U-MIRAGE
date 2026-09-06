from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    output_path = Path("data/examples/mirage_demo.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = ["clean", "visual", "integration", "reasoning"]
    source_risk = {
        "clean": 0.05,
        "visual": 0.72,
        "integration": 0.78,
        "reasoning": 0.70,
    }

    rows = []
    for index in range(16):
        source = sources[index % len(sources)]
        rows.append(
            {
                "id": f"demo-{index:03d}",
                "images": [
                    f"virtual://{index}/a",
                    f"virtual://{index}/b",
                    f"virtual://{index}/c",
                ],
                "question": "Which synthetic label is supported by the evidence?",
                "answer": "alpha",
                "source_label": source,
                "metadata": {
                    "synthetic": {
                        "confidence": 0.92,
                        "risk": source_risk[source],
                        "bad_image": 0,
                        "wrong_answer": "beta",
                    }
                },
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} synthetic episodes -> {output_path}")


if __name__ == "__main__":
    main()
