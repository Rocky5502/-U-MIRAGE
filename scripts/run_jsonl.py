from __future__ import annotations

import argparse
import json

from umirage.adapters.mock import MockMultiImageAdapter
from umirage.adapters.qwen_vl import QwenVLAdapter
from umirage.adapters.transformers_vlm import TransformersVLMAdapter
from umirage.experiment import run_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Run U-MIRAGE on canonical JSONL episodes")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter", choices=["mock", "qwen", "transformers"], default="mock")
    p.add_argument("--model-id")
    p.add_argument("--revision")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    if args.adapter == "mock":
        adapter = MockMultiImageAdapter()
    else:
        if not args.model_id:
            p.error("--model-id is required for real adapters")
        cls = QwenVLAdapter if args.adapter == "qwen" else TransformersVLMAdapter
        adapter = cls(
            args.model_id,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
        )

    config = {
        "adapter_kind": args.adapter,
        "model_id": args.model_id,
        "revision": args.revision,
        "max_new_tokens": args.max_new_tokens,
        "trust_remote_code": args.trust_remote_code,
    }
    metrics = run_experiment(args.input, adapter, args.output, config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
