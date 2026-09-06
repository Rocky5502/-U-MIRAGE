# U-MIRAGE

**Source-Aware Uncertainty Propagation for Reliable Multi-Image Vision-Language Reasoning**

Research-engineering foundation for the CVPR 2027 project. U-MIRAGE decomposes reliability into **visual**, **cross-image integration**, and **reasoning** uncertainty, propagates those signals into final risk, and recommends source-conditioned actions.

> **Pre-experiment status.** The repository is executable end-to-end on deterministic synthetic episodes. Synthetic outputs validate plumbing only and are never treated as paper results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
python scripts/run_smoke.py
pytest -q
```

## Real VLM path

```bash
pip install -e .[vlm]
python -m umirage.cli registry
python -m umirage.cli smoke --output artifacts/smoke
```

The Hugging Face adapter is intentionally optional. Exact model revisions, prompts, dataset revisions, and hardware are frozen in run manifests before official experiments.

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

Primary: MIOH, VisChainBench, MIHBench, REST/REST+. Generalization: HallusionBench, MMMU/MMMU-Pro, VLM-UQBench, HalLoc where release terms permit. Raw third-party media is not vendored. See `data/DATASETS.md`.

## Planned model families

Qwen3-VL 8B (primary), Qwen2.5-VL 7B, InternVL3.5 8B-class, Molmo 2 8B/open variant. Proprietary models, if used, are evaluation-only.

## Reproducibility contract

Every official run records git SHA, model ID/revision, dataset manifest hash, prompt hash, generation parameters, seed, environment, hardware, and output hashes. Paper tables/figures are generated from result artifacts—never hand-entered.

## Repository map

- `src/umirage/`: research library and adapters
- `configs/`: pilot/model/dataset freeze points
- `scripts/`: smoke/data utilities
- `tests/`: deterministic correctness tests
- `docs/`: experiment and result contracts
- `data/`: manifests/examples only
- `artifacts/`: generated outputs (ignored)

MIT-licensed code; dataset/media licenses remain upstream.
