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

Official aggregation should export at minimum:

- `source_attribution.csv`: `model,dataset,method,visual_f1,integration_f1,reasoning_f1,macro_f1,implicated_image_accuracy`
- `failure_calibration.csv`: `model,dataset,method,accuracy,auroc,auprc,ece,brier,aurc`
- `intervention.csv`: `model,dataset,policy,budget,accuracy,hallucination_rate,coverage,extra_calls,recovered_accuracy_per_extra_call,latency_ms,abstention_rate`
- `ablation.csv`: `model,dataset,variant,metric,value`
- `risk_coverage.csv`: `model,dataset,method,coverage,selective_risk`
- `source_confusion.csv`: `model,dataset,true_source,predicted_source,count,rate`
- `generalization.csv`: `train_scope,test_scope,model,dataset,image_count_bin,metric,value`
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
9. Closed/API-model results are supplementary validation only and cannot be required for the method definition.
10. No result is labeled CVPR-paper-ready until its raw run, aggregate artifact, statistics artifact, and provenance all agree on cohort size and identifiers.
