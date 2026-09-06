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

First record the real environment:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory/1024**3)"
```

The dedicated-memory value above is the canonical value for manifests.

## Environment bootstrap

```bash
git clone https://github.com/Rocky5502/-U-MIRAGE.git
cd -- -U-MIRAGE
git checkout agent/pilot-decision-core-v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,vlm,data,analysis]'
pytest -q
python scripts/make_demo_data.py
python scripts/run_smoke.py
```

## Dataset smoke gate

Materialize a small deterministic MIHBench cohort first:

```bash
python scripts/materialize_mihbench_pilot.py --limit 50 --seed 27
```

Inspect the generated JSONL, image paths, manifest hash, and task balance before loading a real VLM.

## First real GPU gate

1. Verify **dedicated** VRAM with `nvidia-smi`.
2. Freeze one Qwen model revision in `configs/models/`.
3. Start with MIHBench **20 examples**, then **50**, then **100**.
4. Use batch size 1 and deterministic decoding.
5. Prefer 4-bit/8-bit loading on the 16 GB local GPU if BF16/FP16 does not fit cleanly.
6. Never silently reduce image resolution, image count, precision, or context after an OOM; change the frozen config and record it.
7. Inspect raw decoded outputs, generated-token confidence, entropy, source diagnostics, latency, and provenance after each gate.
8. Only after both Qwen3-VL-8B and Qwen2.5-VL-7B pass the smoke gates should the **500-example MIRAGE-U pilot** begin.
9. Do not launch the four-family sweep until integration uncertainty adds predictive value beyond visual + reasoning uncertainty and source-conditioned routing beats a matched-cost generic policy.

Example:

```bash
python scripts/run_jsonl.py \
  --input data/cache/mihbench_pilot.jsonl \
  --output artifacts/pilot/qwen3vl8b \
  --adapter qwen \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision <FROZEN_REVISION>
```

## Cloud escalation rule

Use local compute by default. Escalate only when the experiment has passed the pilot gate:

- 24-32 GB rented GPU: compatibility fallback if a specific 7B-13B configuration does not fit locally.
- 48 GB rented GPU: final BF16/high-image-count or white-box sweeps.
- 80 GB A100/H100-class or equivalent: larger-family validation or dense final sweeps where additional compute directly answers an RQ.

Every rental must have an experiment ID, exact accelerator SKU, VRAM, driver/runtime, rental duration, and internal cost record. See `docs/COMPUTE_PLAN.md` and `configs/experiments/compute_tiers.yaml`.

## Optional closed-model evaluation

Closed multimodal APIs are external validation only and must not be required for the main method. Keep credentials in environment variables, never in the repository:

```bash
export UMIRAGE_API_BASE_URL="<OPENAI_COMPATIBLE_BASE_URL>"
export UMIRAGE_API_KEY="<SECRET>"
export UMIRAGE_API_MODEL="<FROZEN_MODEL_ID>"
```

For any reported API run, freeze the exact model ID, date/time, prompt hash, temperature, max output tokens, retry policy, and call count.
