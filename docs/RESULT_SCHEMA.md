# Paper-facing artifact contract

Official U-MIRAGE runs must be reproducible from machine-readable artifacts. No manuscript number may be hand-entered without a corresponding frozen result artifact.

## Empty pre-result bootstrap

Before any confirmatory run, initialize the paper-facing directory with:

```bash
python scripts/init_paper_results.py --output-dir results/paper
```

This command creates header-only CSV files and empty JSON arrays. It creates **no empirical values**. The v6 manuscript mirrors this state: every empirical cell is visually empty until an official frozen artifact is available.

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

Each official run directory must contain:

- `predictions.jsonl`
- `metrics.json`
- `risk_coverage.json`
- `provenance.json`

When a run is compared against another method on the same episodes, the comparison directory must also contain a paired statistics JSON generated from `scripts/compare_runs.py` or an equivalent frozen analysis command.

`provenance.json` must freeze the adapter/checkpoint identifier, input manifest path and SHA256, experiment configuration hash, example count, run configuration, dtype/quantization, prompt/template identity, decoding parameters, and relevant software/hardware identifiers.

## Paired comparison JSON

Paired comparison artifacts must record:

- left/right run paths or immutable run IDs;
- pairing key (`id`);
- episode count;
- bootstrap sample count;
- bootstrap seed;
- confidence level / alpha;
- per-metric observed delta;
- lower and upper bootstrap confidence bounds;
- paired win/loss/tie counts for correctness;
- exact McNemar p-value for correctness;
- multiple-testing correction when a family of tests is reported.

The default frozen policy is 5,000 paired bootstrap samples, seed 27, 95% confidence intervals, exact McNemar for paired correctness, and Holm-Bonferroni for a reported family of significance tests.

## Paper-facing aggregate artifacts

Official aggregation must export:

- `source_attribution.csv`: `run_id,model,dataset,method,visual_f1,integration_f1,reasoning_f1,clean_f1,macro_f1,image_localization,seed`
- `source_selectivity.csv`: `run_id,model,source,perturbation_family,severity,delta_visual,delta_integration,delta_reasoning,selectivity,negative_control,seed`
- `failure_calibration.csv`: `run_id,model,dataset,method,accuracy,auroc,auprc,ece,brier,aurc,coverage_at_target_risk,seed`
- `intervention.csv`: `run_id,model,dataset,policy,budget_calls,accuracy,hallucination_rate,aurc,coverage,extra_calls,vision_tokens,generation_tokens,latency_ms,seed`
- `ablation.csv`: `run_id,model,dataset,variant,macro_f1,auroc,aurc,accuracy,seed`
- `transfer.csv`: `run_id,train_model,test_model,train_dataset,test_dataset,method,macro_f1,auroc,aurc,relative_degradation`
- `risk_coverage.csv`: `run_id,model,dataset,method,coverage,selective_risk,seed`
- `source_confusion.csv`: `run_id,model,dataset,true_source,predicted_source,count,rate`
- `statistics.json`: paired tests, bootstrap confidence intervals, correction method, seeds, effect sizes
- `compute.json`: GPU, dedicated VRAM, wall time, vision tokens, generated tokens, forward passes, extra calls

## Matched-compute intervention contract

For RQ3, intervention rows must preserve the explicit extra-call budget. U-MIRAGE and every comparator must be evaluated on identical base episodes under the same budget in `{0,1,2}` extra model calls unless a separately documented sensitivity study states otherwise.

A method may not receive additional images, prompts, model calls, or hidden judge calls that are unavailable to its matched comparator.

## Scientific integrity rules

1. Synthetic smoke artifacts must contain `synthetic_only: true` and are never manuscript inputs.
2. Calibration fitting data must be disjoint from the human-verified pilot test subset.
3. Raw `predictions.jsonl` files are immutable after a run is frozen; corrections require a new run ID.
4. Paper tables and figures are generated from aggregate artifacts, never manually edited numbers.
5. Model API/checkpoint IDs, revisions, prompts, decoding settings, perturbation seeds, and dataset manifests must be frozen in provenance.
6. Failed or partially completed runs must never be mixed into final aggregate artifacts without an explicit status field and documented exclusion rule.
7. Post-hoc metric, baseline, threshold, or cohort changes after inspecting results require an explicit protocol deviation record.
8. Statistical comparisons are paired by episode whenever methods were evaluated on the same cohort.
9. Closed/API-model results are external validation only and cannot be required for the method definition.
10. No result is labeled CVPR-paper-ready until its raw run, aggregate artifact, statistics artifact, and provenance all agree on cohort size and identifiers.
11. Empty pre-result templates may contain headers/schema metadata only; they must never contain placeholder numeric values.
