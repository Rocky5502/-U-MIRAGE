# Local pilot run

Recommended for the Windows lab machine: use **WSL2 Ubuntu** for the final reproducible environment unless a model dependency forces native Windows.

## Lab machine

- Windows 10 Pro
- NVIDIA GeForce RTX 5070 Ti
- NVIDIA reference dedicated VRAM: 16 GB GDDR7
- Windows-reported total GPU memory: approximately 48 GB may include shared system RAM; **do not use this as the VRAM value in experiment planning**
- RAM: 64 GB
- Storage: Samsung SSD 990 PRO 2 TB
- CPU: Intel; exact model must be recorded before official runs

## 0. Clone the frozen pilot branch

```bash
git clone https://github.com/Rocky5502/-U-MIRAGE.git
cd -- -U-MIRAGE
git checkout agent/pilot-decision-core-v1
```

Do not run the scientific pilot from `main` until the pilot PR is validated and merged.

## 1. Environment bootstrap

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,vlm,data,analysis,quant]'
```

If quantized loading is unavailable on the current platform, reinstall without `quant` and use an unquantized configuration only if it fits without silently changing the protocol.

## 2. Hardware doctor — run this first

```bash
python scripts/local_kitchen.py doctor --require-cuda
```

This writes `artifacts/local_lab/hardware.json` and records Python, PyTorch/CUDA, GPU name, dedicated VRAM, compute capability, driver, RAM, git branch, and git HEAD when available.

The dedicated-memory value reported by PyTorch / `nvidia-smi` is the canonical VRAM value for experiment manifests.

## 3. Zero-cost code gate

```bash
python scripts/local_kitchen.py mock
```

This runs tests, demo-data generation, the mock smoke path, and initializes the empty paper-facing result files. Synthetic outputs are never manuscript evidence.

## 4. Materialize the deterministic MIHBench cohort

Materialize one cohort large enough for every local gate so all stages share the same input manifest and SHA256:

```bash
python scripts/materialize_mihbench_pilot.py --limit 1000 --seed 27
```

Inspect:

- `data/cache/mihbench_pilot.jsonl`
- local image paths
- manifest SHA256
- task/source balance

Do not create separate hand-edited 20/50/100 subsets. Gate size is controlled by `--size`, while provenance preserves the same full input-manifest hash.

## 5. First real GPU gate — Qwen3-VL-8B

Default local strategy for the 16 GB RTX 5070 Ti is deterministic batch-1 inference with BF16 compute and 4-bit model loading:

```bash
python scripts/local_kitchen.py gate \
  --size 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --quantization 4bit
```

The run directory is deterministic:

`artifacts/local/qwen-qwen3-vl-8b-instruct/gate_0020/`

It must contain:

- `launch.json`
- `hardware.json`
- `predictions.jsonl`
- `metrics.json`
- `risk_coverage.json`
- `provenance.json`
- `COMPLETE`

Inspect raw answers and diagnostics before progressing.

## 6. Scale only after inspection

```bash
python scripts/local_kitchen.py gate --size 50  --model-id Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --quantization 4bit
python scripts/local_kitchen.py gate --size 100 --model-id Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --quantization 4bit
```

Repeat the compatibility ladder for:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

For any 500+ confirmatory run, `scripts/local_kitchen.py` requires an explicit frozen `--revision`:

```bash
python scripts/local_kitchen.py gate \
  --size 500 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision <FROZEN_HF_REVISION> \
  --dtype bfloat16 \
  --quantization 4bit
```

## Precision / OOM policy

1. Start from the frozen local default: batch size 1, deterministic decoding, BF16 compute, 4-bit loading.
2. If 4-bit is unavailable but 8-bit fits, change to `--quantization 8bit` and record a new run.
3. If unquantized BF16/FP16 is used, record it explicitly with `--quantization none`.
4. Never silently reduce image count, image resolution, context, precision, or output length after OOM.
5. Any protocol-affecting memory change requires a new run directory/config and a deviation note.

## Scientific progression gate

Do **not** start the four-family sweep just because the code runs. Progress from 100 to 500 only after:

- generated-token confidence and entropy are present and plausible;
- leave-one-image-out, image-order, and reasoning diagnostics are complete;
- episode IDs/source labels are stable;
- no systematic parser/answer-normalization failure is visible;
- integration uncertainty is not trivially identical to visual uncertainty;
- output/provenance files agree on cohort size and model settings.

Do not add a learned propagation module until the 500-example pilot demonstrates source separability and the transparent propagator leaves a reproducible gap.

## Cloud escalation rule

Use local compute by default. Escalate only when the experiment has passed the scientific pilot gate:

- 24–32 GB rented GPU: compatibility fallback for a configuration that does not fit locally.
- 48 GB rented GPU: final BF16/high-image-count or white-box sweeps.
- 80 GB A100/H100-class or equivalent: larger-family validation or dense final sweeps where additional compute directly answers an RQ.

Every rental must have an experiment ID, exact accelerator SKU, VRAM, driver/runtime, rental duration, and internal cost record. See `docs/COMPUTE_PLAN.md`, `configs/experiments/compute_tiers.yaml`, and `configs/experiments/local_kitchen.yaml`.

## Optional closed-model evaluation

Closed multimodal APIs are external validation only and must not be required for the main method. Keep credentials in environment variables, never in the repository:

```bash
export UMIRAGE_API_BASE_URL="<OPENAI_COMPATIBLE_BASE_URL>"
export UMIRAGE_API_KEY="<SECRET>"
export UMIRAGE_API_MODEL="<FROZEN_MODEL_ID>"
```

For any reported API run, freeze the exact model ID, date/time, prompt hash, temperature, max output tokens, retry policy, and call count.
