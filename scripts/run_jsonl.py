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
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device-map", default="auto")
    p.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    quant = p.add_mutually_exclusive_group()
    quant.add_argument("--load-in-4bit", action="store_true")
    quant.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be a positive integer")

    if args.adapter == "mock":
        if args.load_in_4bit or args.load_in_8bit:
            p.error("quantization flags are only valid for real adapters")
        adapter = MockMultiImageAdapter()
    else:
        if not args.model_id:
            p.error("--model-id is required for real adapters")
        cls = QwenVLAdapter if args.adapter == "qwen" else TransformersVLMAdapter
        adapter = cls(
            args.model_id,
            revision=args.revision,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

    quantization = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "none"
    config = {
        "adapter_kind": args.adapter,
        "model_id": args.model_id,
        "revision": args.revision,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "quantization": quantization,
        "trust_remote_code": args.trust_remote_code,
        "deterministic_decoding": True,
    }
    metrics = run_experiment(args.input, adapter, args.output, config, limit=args.limit)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
