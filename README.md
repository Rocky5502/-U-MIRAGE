# U-MIRAGE

**Source-Aware Uncertainty Propagation for Reliable Multi-Image Vision-Language Reasoning**

U-MIRAGE is a research codebase for studying **where uncertainty originates** in multi-image VLM reasoning and how it propagates from visual evidence, through cross-image integration, to downstream reasoning and final answer risk.

> Status: **CVPR 2027 pre-experiment freeze / local execution ready**. The repository is executable on synthetic examples, has real-model adapters and deterministic local GPU gates, and claims no paper result until the frozen experiment suite is run.

## Scientific contract

U-MIRAGE separates three uncertainty sources:

1. **Visual uncertainty** — an individual image/region is unreliable or ambiguous.
2. **Integration uncertainty** — images are individually interpretable, but their joint evidence is unstable/conflicting.
3. **Reasoning uncertainty** — evidence is adequate, but composition/inference is unreliable.

The system emits a risk estimate, a source posterior, and a source-conditioned action recommendation. The first confirmatory implementation is intentionally transparent/non-learned; a learned propagator is considered only after the 500-example pilot establishes source separability and leaves a reproducible gap.

## Quick start — local kitchen

For the Windows lab machine, prefer WSL2 Ubuntu for final reproducible runs.

```bash
git checkout agent/pilot-decision-core-v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,vlm,data,analysis,quant]'

python scripts/local_kitchen.py doctor --require-cuda
python scripts/local_kitchen.py mock
```

The hardware doctor records the real Python/PyTorch/CUDA/GPU/VRAM/git state. Synthetic smoke artifacts are **not paper results**.

## Real-model compatibility ladder

Materialize one deterministic MIHBench cohort:

```bash
python scripts/materialize_mihbench_pilot.py --limit 1000 --seed 27
```

This writes the canonical compatibility input to:

`data/materialized/mihbench_pilot/episodes.jsonl`

Then run the first real gate:

```bash
python scripts/local_kitchen.py gate \
  --size 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --quantization 4bit
```

Progress only after inspection:

```bash
python scripts/local_kitchen.py gate --size 50  --model-id Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --quantization 4bit
python scripts/local_kitchen.py gate --size 100 --model-id Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --quantization 4bit
```

Repeat the compatibility ladder for `Qwen/Qwen2.5-VL-7B-Instruct`.

MIHBench is a compatibility/external benchmark path. The **500+ scientific pilot must use a frozen source-balanced MIRAGE-U JSONL** passed explicitly with `--input`, and `scripts/local_kitchen.py` requires a frozen model revision for that stage.

See `docs/LOCAL_RUN.md` for the exact run contract.

## Real runner guarantees

The real runner supports:

- deterministic gate limits on one immutable input manifest;
- Qwen and generic Transformers image-text adapters;
- explicit BF16/FP16/FP32/auto dtype;
- optional 4-bit or 8-bit loading;
- generated-token probability confidence rather than vocabulary-max confidence;
- normalized predictive entropy;
- raw leave-one-image-out, image-order, and reasoning counterfactual diagnostics;
- run-level hardware, launch, model, dataset, and configuration provenance.

A completed local run contains `launch.json`, `hardware.json`, `predictions.jsonl`, `metrics.json`, `risk_coverage.json`, `provenance.json`, and `COMPLETE`.

## Paper-result contract

The v6 manuscript is synchronized with the repository and contains no synthetic or placeholder empirical values. Initialize empty paper-facing artifacts with:

```bash
python scripts/init_paper_results.py --output-dir results/paper
```

Canonical outputs include source attribution/selectivity, calibration, intervention, ablation, transfer, risk-coverage, confusion, statistics, and compute artifacts. No manuscript number is official unless it is reproducible from immutable run artifacts.

## Compute policy

The development machine is Windows 10 Pro with an NVIDIA GeForce RTX 5070 Ti, 64 GB system RAM, and a Samsung 990 PRO 2 TB SSD. NVIDIA's reference RTX 5070 Ti has 16 GB dedicated GDDR7. Windows may report a larger total GPU-memory number because shared system memory is included, so official experiment manifests use the dedicated-memory value reported by `nvidia-smi` / PyTorch.

Local compute is the default for smoke tests, 7B–8B pilot inference, cached features, calibration, and propagation. Rented 24–32/48/80 GB accelerators are reserved for scientifically justified compatibility or final sweeps after the pilot demonstrates value. See `docs/COMPUTE_PLAN.md`, `configs/experiments/compute_tiers.yaml`, and `configs/experiments/local_kitchen.yaml`.

Closed multimodal APIs are external validation only. API credentials are supplied through environment variables and are never committed. Exact model IDs, dates, prompts, retry settings, and call counts are frozen for any reported API experiment.

## Repository map

```text
configs/                 frozen experiment/model/dataset configs
src/umirage/             package code
  adapters/              mock + real VLM adapters
  signals/               visual/integration/reasoning uncertainty
  propagation.py         source-aware uncertainty propagation
  intervention.py        source-conditioned action policy
  evaluation.py          reliability/calibration metrics
  statistics.py          paired bootstrap / McNemar / Holm utilities
scripts/                 materializers, local kitchen, analysis and smoke entry points
tests/                   deterministic unit/integration tests
data/                     manifests/examples/local materializations; no redistributed restricted media
artifacts/                generated outputs (git-ignored except schema docs)
docs/                     experiment protocol, paper freeze and reproducibility notes
```

## Frozen benchmark/model plan

Compatibility starts with MIHBench. The source-controlled scientific pilot uses MIRAGE-U. Primary external evaluation then prioritizes MIOH, VisChainBench and MIHBench, followed by selected transfer benchmarks where release terms and protocol fit.

Primary model families are Qwen3-VL-8B and Qwen2.5-VL-7B; additional InternVL/Molmo-class families are added only after the two-model scientific gate passes.

## Reproducibility rules

- Every official run records git SHA, model ID/revision, dataset input hash, prompt/generation settings, dtype/quantization, environment, and hardware.
- Raw benchmark media is never silently modified; perturbations are deterministic and manifest-driven.
- Synthetic/CI outputs are separated from paper-result directories.
- No hand-entered numbers are copied into manuscript tables or plots.
- OOM/config changes create a new recorded run; image count/resolution/precision are never silently altered.
- Cloud GPU rental and API use require scientific justification and experiment provenance.

## License

Code is released under the MIT License. Dataset/media licenses remain with their original owners and are documented separately.
