# U-MIRAGE

**Source-Aware Uncertainty Propagation for Reliable Multi-Image Vision-Language Reasoning**

Research-engineering foundation for the CVPR 2027 project. U-MIRAGE decomposes reliability into **visual**, **cross-image integration**, and **reasoning** uncertainty, propagates those signals into final risk, and recommends source-conditioned actions.

> **Pre-experiment status.** The repository is executable end-to-end on deterministic synthetic episodes. Synthetic outputs validate plumbing only and are never treated as paper results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/WSL: source .venv/bin/activate
pip install -e .[dev]
python scripts/run_smoke.py
pytest -q
```

For the real local VLM path:

```bash
pip install -e '.[vlm,data]'
python -m umirage.cli registry
python scripts/run_jsonl.py --input <pilot.jsonl> --output artifacts/pilot/qwen3vl8b --model-id Qwen/Qwen3-VL-8B-Instruct --revision <FROZEN_REVISION>
```

See `docs/LOCAL_RUN.md` before the first GPU run.

## Scientific pipeline

```text
images + question
  ├─ visual uncertainty      (local evidence quality)
  ├─ integration uncertainty (leave-one-image-out/order counterfactuals)
  └─ reasoning uncertainty   (structured prompt/trace consistency)
                ↓
        source-aware propagation
                ↓
       failure risk + source posterior
                ↓
reinspect image / compare images / verify reasoning / abstain
```

## Planned benchmark suite

Primary: MIOH, VisChainBench, MIHBench, REST/REST+. Generalization: HallusionBench, MMMU/MMMU-Pro, VLM-UQBench, HalLoc where release terms permit. Raw third-party media is not vendored. See `data/DATASETS.md` and `data/SCHEMA.md`.

## Planned model families

Qwen3-VL 8B (primary), Qwen2.5-VL 7B, InternVL3.5 8B-class, Molmo 2 8B/open variant. Proprietary models, if used, are evaluation-only. Exact revisions are frozen before scientific runs.

## Reliability metrics implemented

The experiment runner writes raw predictions plus Accuracy, AUROC, ECE, Brier score, AURC, and source Macro-F1. Paper tables remain downstream of raw artifacts.

## Reproducibility contract

Every official run records git SHA, model ID/revision, dataset manifest hash, prompt/configuration, seed, environment, hardware, and output artifacts. Paper tables/figures are generated from result artifacts—never hand-entered.

## Repository map

- `src/umirage/`: research library, real/mock adapters, metrics, experiment runner
- `configs/`: pilot/model/dataset freeze points
- `scripts/`: smoke, dataset, and JSONL run utilities
- `tests/`: deterministic correctness tests
- `docs/`: experiment, local-run, and result contracts
- `data/`: schemas/manifests only; raw media stays upstream
- `artifacts/`: generated outputs (ignored)

MIT-licensed code; dataset/media licenses remain upstream.
