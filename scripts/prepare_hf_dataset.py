from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HF_DATASETS = {
    "vischainbench": "eyehole/VisChainBench",
    "mihbench": "MM-Hallu/MIHBench",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an allowed HF benchmark and write a provenance manifest."
    )
    parser.add_argument("dataset", choices=sorted(HF_DATASETS))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `pip install -e .[vlm]` first") from exc

    hf_id = HF_DATASETS[args.dataset]
    dataset = load_dataset(
        hf_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    summary = {
        "id": args.dataset,
        "hf_id": hf_id,
        "requested_revision": args.revision,
        "splits": {name: len(split) for name, split in dataset.items()},
    }
    canonical = json.dumps(summary, sort_keys=True).encode()
    summary["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()

    output_path = Path("data/manifests") / f"{args.dataset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
