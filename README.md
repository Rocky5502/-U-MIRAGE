# U-MIRAGE

**Source-Aware Uncertainty Propagation for Reliable Multi-Image Vision-Language Reasoning**

U-MIRAGE is a research codebase for studying **where uncertainty originates** in multi-image VLM reasoning and how it propagates from visual evidence, through cross-image integration, to downstream reasoning and final answer risk.

> Status: **research foundation / pre-experiment**. The repository is fully executable on synthetic examples and includes production interfaces for real VLMs and benchmarks. No paper result is claimed until the frozen experiment suite is run.

## Scientific contract

U-MIRAGE separates three uncertainty sources:

1. **Visual uncertainty** — an individual image/region is unreliable or ambiguous.
2. **Integration uncertainty** — images are individually interpretable, but their joint evidence is unstable/conflicting.
3. **Reasoning uncertainty** — evidence is adequate, but composition/inference is unreliable.

The system emits a calibrated risk estimate, a source posterior, and a source-conditioned action recommendation.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
python scripts/make_demo_data.py
python scripts/run_smoke.py
pytest -q
```

Smoke artifacts are written to `artifacts/smoke/` and are **not paper results**.

## Real-model setup

```bash
pip install -e .[vlm]
python -m umirage.cli registry
python -m umirage.cli run \
  --dataset data/examples/mirage_demo.jsonl \
  --adapter mock \
  --output artifacts/dev/run.jsonl
```

The `transformers` adapter supports local Hugging Face VLM checkpoints when installed with the `vlm` extra. For the paper, model/checkpoint IDs will be frozen in `configs/models/` before the first official run.

## Real pilot runner

For real local inference use the canonical JSONL runner; start with 20 examples before scaling:

```bash
pip install -e '.[dev,vlm,data,analysis]'
python scripts/run_jsonl.py --input <pilot.jsonl> --output artifacts/pilot/qwen3vl8b --adapter qwen --model-id Qwen/Qwen3-VL-8B-Instruct --revision <FROZEN_REVISION>
```

The runner writes raw predictions, reliability metrics, risk--coverage data, and an environment/model/dataset provenance manifest. See `docs/LOCAL_RUN.md`, `docs/COMPUTE_PLAN.md`, `data/SCHEMA.md`, and `docs/MODEL_ADAPTERS.md`.

## Compute policy

The development machine is Windows 10 Pro with an NVIDIA GeForce RTX 5070 Ti, 64 GB system RAM, and a Samsung 990 PRO 2 TB SSD. NVIDIA's reference RTX 5070 Ti has 16 GB dedicated GDDR7. Windows may report a larger total GPU-memory number because shared system memory is included, so official experiment manifests must use the dedicated-memory value reported by `nvidia-smi`.

Local compute is the default for smoke tests, 7B–8B pilot inference, cached features, calibration, and propagation. Rented 48–80 GB accelerators are reserved for scientifically justified final sweeps, larger model-family validation, or white-box/high-resolution runs that do not fit cleanly on local hardware. See `configs/experiments/compute_tiers.yaml`.

Closed multimodal APIs are evaluation-only. API credentials must be supplied through environment variables and are never committed. Exact model IDs, dates, prompts, retry settings, and call counts are frozen for any reported API experiment.

## Repository map

```text
configs/                 frozen experiment/model/dataset configs
src/umirage/             package code
  adapters/              mock + optional Transformers VLM adapters
  signals/               visual/integration/reasoning uncertainty
  propagation.py         source-aware uncertainty propagation
  intervention.py        source-conditioned action policy
  datasets.py            benchmark registry + JSONL schema
  evaluation.py          reliability/calibration metrics
scripts/                 data preparation and smoke-run entry points
tests/                   deterministic unit/integration tests
data/                     manifests/examples only; no restricted media
artifacts/                generated outputs (git-ignored except schema docs)
docs/                     experiment protocol and reproducibility notes
```

## Planned benchmark suite

Primary: **MIOH**, **VisChainBench**, **MIHBench**, **REST/REST+**.  
Generalization: HallusionBench, MMMU/MMMU-Pro, VLM-UQBench, HalLoc where release terms permit.

See [`data/DATASETS.md`](data/DATASETS.md) before downloading or redistributing any data.

## Planned model families

- Qwen3-VL (8B-class primary)
- Qwen2.5-VL 7B
- InternVL3.5 (8B-class)
- Molmo 2 (8B/open variant)

Closed models, if used, are evaluation-only and must be frozen by exact API identifier/date in the experiment manifest.

## Reproducibility rules

- Every official run gets a manifest containing git SHA, model ID/revision, dataset manifest hash, seed, prompt hash, generation settings, environment, and hardware.
- Raw benchmark media is never silently modified; perturbations are specified by deterministic manifests.
- Synthetic/CI runs are clearly separated from paper-result directories.
- No hand-entered numbers are copied into paper tables; table builders consume exported CSV/JSON artifacts.
- Quantization, dtype, attention backend, image-resolution policy, dedicated VRAM, wall time, and peak memory are frozen for official runs.
- Cloud GPU rental and API use require an experiment ID and must not be used to compensate for an unvalidated hypothesis.

## License

Code is released under the MIT License. Dataset/media licenses remain with their original owners and are documented separately.
