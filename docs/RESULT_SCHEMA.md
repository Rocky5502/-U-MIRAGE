# Paper-facing artifact contract

Official U-MIRAGE runs must be reproducible from machine-readable artifacts. No manuscript number may be hand-entered without a corresponding frozen result artifact.

## Per-episode predictions

`predictions.jsonl` is the canonical raw record. Every row must contain at least:

- `id`
- `prediction`
- `gold`
- `correct`
- `source_label`
- `visual`
- `integration`
- `reasoning`
- `risk`
- `predicted_source`
- `source_posterior`
- `action`
- `model_confidence`
- `token_entropy`
- `evidence_scores`
- `model_metadata`
- `n_images`
- `episode_metadata`
- `diagnostics`

`diagnostics` must preserve the raw evidence used to derive source scores:

- `full`: original answer/confidence/entropy/evidence scores
- `leave_one_out`: one record for each omitted image
- `order_variants`: image-order counterfactual records
- `reasoning_variants`: prompt/reasoning consistency records
- `n_images`

This record is the audit trail for source attribution. Aggregated tables must be reproducible from it.

## Run-level outputs

Each run directory must contain:

- `predictions.jsonl`
- `metrics.json`
- `risk_coverage.json`
- `provenance.json`

`provenance.json` must freeze the adapter/checkpoint identifier, input manifest path and SHA256, experiment configuration hash, example count, and run configuration.

## Paper-facing aggregate artifacts

Official aggregation should export at minimum:

- `source_attribution.csv`: `model,dataset,method,visual_f1,integration_f1,reasoning_f1,macro_f1`
- `failure_calibration.csv`: `model,dataset,method,auroc,auprc,ece,brier,aurc`
- `intervention.csv`: `model,dataset,policy,accuracy,hallucination_rate,coverage,extra_calls,latency_ms`
- `ablation.csv`: `model,dataset,variant,metric,value`
- `risk_coverage.csv`: `model,dataset,method,coverage,selective_risk`
- `source_confusion.csv`: `model,dataset,true_source,predicted_source,count,rate`
- `statistics.json`: paired tests, bootstrap confidence intervals, correction method, seeds
- `compute.json`: GPU, dedicated VRAM, wall time, vision tokens, generated tokens, forward passes

## Scientific integrity rules

1. Synthetic smoke artifacts must contain `synthetic_only: true` and are never manuscript inputs.
2. Calibration fitting data must be disjoint from the human-verified pilot test subset.
3. Raw `predictions.jsonl` files are immutable after a run is frozen; corrections require a new run ID.
4. Paper tables and figures are generated from aggregate artifacts, never manually edited numbers.
5. Model API/checkpoint IDs, revisions, prompts, decoding settings, perturbation seeds, and dataset manifests must be frozen in provenance.
6. Failed or partially completed runs must never be mixed into final aggregate artifacts without an explicit status field and documented exclusion rule.
