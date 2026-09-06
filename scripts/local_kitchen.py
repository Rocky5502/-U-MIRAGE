from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

GATE_SIZES = (20, 50, 100, 500, 1000)
DEFAULT_COMPATIBILITY_INPUT = "data/materialized/mihbench_pilot/episodes.jsonl"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def run_doctor(output: Path, require_cuda: bool) -> None:
    cmd = [sys.executable, "scripts/check_local_lab.py", "--output", str(output)]
    if require_cuda:
        cmd.append("--require-cuda")
    _run(cmd)


def run_mock() -> None:
    _run([sys.executable, "-m", "pytest", "-q"])
    _run([sys.executable, "scripts/make_demo_data.py"])
    _run([sys.executable, "scripts/run_smoke.py"])
    _run([sys.executable, "scripts/init_paper_results.py"])


def run_gate(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Input manifest does not exist: {input_path}\n"
            "For the MIHBench compatibility ladder, first run "
            "`python scripts/materialize_mihbench_pilot.py --limit 1000 --seed 27`. "
            "For the 500+ MIRAGE-U scientific pilot, pass the frozen source-balanced JSONL with --input."
        )
    if args.size >= 500 and not args.revision:
        raise SystemExit("500+ episode confirmatory runs require a frozen --revision.")
    if args.size >= 500 and args.input == DEFAULT_COMPATIBILITY_INPUT:
        raise SystemExit(
            "The default MIHBench input is for 20/50/100 compatibility gates only. "
            "For the 500+ scientific pilot, pass the frozen source-balanced MIRAGE-U JSONL with --input."
        )

    model_slug = _slug(args.model_id)
    output = Path(args.output_root) / model_slug / f"gate_{args.size:04d}"
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise SystemExit(f"Refusing to overwrite existing run: {output}; pass --force explicitly.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    launch = {
        "stage": "gate",
        "size": args.size,
        "input": str(input_path),
        "adapter": args.adapter,
        "model_id": args.model_id,
        "revision": args.revision,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "output": str(output),
    }
    (output / "launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True), encoding="utf-8"
    )

    run_doctor(output / "hardware.json", require_cuda=True)

    cmd = [
        sys.executable,
        "scripts/run_jsonl.py",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--adapter",
        args.adapter,
        "--model-id",
        args.model_id,
        "--limit",
        str(args.size),
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if args.quantization == "4bit":
        cmd.append("--load-in-4bit")
    elif args.quantization == "8bit":
        cmd.append("--load-in-8bit")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    _run(cmd)
    (output / "COMPLETE").write_text("ok\n", encoding="utf-8")
    print(f"\nCompleted U-MIRAGE gate: {output}")
    print("Inspect predictions.jsonl, metrics.json, risk_coverage.json, provenance.json, and hardware.json.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U-MIRAGE local experiment kitchen")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="capture local hardware/CUDA state")
    doctor.add_argument("--output", default="artifacts/local_lab/hardware.json")
    doctor.add_argument("--require-cuda", action="store_true")

    sub.add_parser("mock", help="run tests, mock smoke, and initialize empty paper results")

    gate = sub.add_parser("gate", help="run a deterministic real-model experiment gate")
    gate.add_argument("--size", type=int, choices=GATE_SIZES, required=True)
    gate.add_argument("--input", default=DEFAULT_COMPATIBILITY_INPUT)
    gate.add_argument("--output-root", default="artifacts/local")
    gate.add_argument("--adapter", choices=["qwen", "transformers"], default="qwen")
    gate.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    gate.add_argument("--revision")
    gate.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    gate.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    gate.add_argument("--device-map", default="auto")
    gate.add_argument("--max-new-tokens", type=int, default=128)
    gate.add_argument("--trust-remote-code", action="store_true")
    gate.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "doctor":
        run_doctor(Path(args.output), args.require_cuda)
    elif args.command == "mock":
        run_mock()
    else:
        run_gate(args)


if __name__ == "__main__":
    main()
