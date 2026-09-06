# Frozen experiment protocol

This protocol is frozen before paper results. Any post-hoc change to datasets, metrics, baselines, thresholds, or statistical tests must be documented in provenance and justified in the manuscript.

## Gate 0 — environment and reproducibility

Capture before every official run:

- `git rev-parse HEAD`;
- Python, PyTorch, Transformers, CUDA, driver, and ffmpeg versions;
- GPU model and dedicated VRAM from `nvidia-smi`;
- OS, CPU, RAM, and storage;
- exact model repository + revision;
- exact dataset manifest + SHA256;
- prompt template hash;
- decoding parameters;
- quantization/dtype;
- image preprocessing and ordering policy.

Synthetic smoke data is never a scientific result.

## Gate 1 — local compatibility ladder

Run in order: **20 → 50 → 100 real episodes**.

At each level manually inspect:

- decoded images and image order;
- full prediction;
- generated-token confidence and entropy;
- leave-one-image-out diagnostic;
- image-order diagnostic;
- reasoning counterfactual;
- visual/integration/reasoning components;
- source posterior;
- final action.

Do not scale when diagnostics are malformed, source posterior collapses to one class, or deterministic reruns disagree unexpectedly.

## Gate 2 — MIRAGE-U scientific pilot

Run a balanced **500-example** source-controlled pilot on:

- Qwen3-VL-8B;
- Qwen2.5-VL-7B.

Scale to 1,000 only after the 500-example analysis is complete.

Required pilot questions:

1. Does integration uncertainty add information beyond visual + reasoning signals?
2. Does source Macro-F1 exceed majority-source and scalar-confidence baselines?
3. Does propagation outperform the same source signals without propagation?
4. Does source-conditioned intervention beat generic verification at matched extra-call budget?

If the answer is no after debugging and sensitivity checks, pivot the scientific claim instead of scaling compute.

## Gate 3 — RQ1 source localization

Primary datasets:

- MIRAGE-U;
- MIHBench controlled subset where source labels can be mapped reliably.

Primary metrics:

- visual F1;
- integration F1;
- reasoning F1;
- Macro-F1;
- implicated-image accuracy where labels exist.

Mandatory baselines:

- scalar generated-token confidence;
- token entropy;
- verbalized confidence;
- self-consistency;
- leave-one-image-out without propagation;
- majority-source predictor.

## Gate 4 — RQ2 failure prediction and propagation

Primary datasets:

- MIOH;
- VisChainBench;
- MIHBench.

Primary metrics:

- failure AUROC;
- failure AUPRC;
- ECE;
- Brier score;
- AURC;
- selective risk at frozen coverage levels.

Mandatory ablation/baseline conditions:

- visual only;
- integration only;
- reasoning only;
- unpropagated source vector;
- equal-weight aggregation;
- generic weighted risk;
- full source-aware propagation.

## Gate 5 — RQ3 matched-compute intervention

Evaluate extra-call budgets **0, 1, 2** on identical base episodes.

Methods:

- answer only;
- global abstention threshold;
- generic verify;
- generic re-query;
- source-conditioned U-MIRAGE.

Report:

- post-intervention accuracy;
- hallucination/error rate;
- recovered accuracy per extra call;
- mean extra calls;
- latency;
- abstention rate.

No method may receive extra evidence, calls, or larger prompts unavailable to its matched comparator.

## Gate 6 — RQ4 generalization

After the two-model pilot passes, expand to additional open model families and test:

- leave-one-model-family-out transfer;
- leave-one-dataset-out transfer;
- 2-image versus 3–4-image versus 5+-image episodes;
- corruption/severity shift where available.

Primary outputs:

- source Macro-F1;
- AUROC;
- AURC;
- calibration shift.

## Gate 7 — mandatory ablations

Complete before final paper freeze:

- remove visual source;
- remove integration source;
- remove reasoning source;
- no propagation;
- equal source weights;
- remove leave-one-image-out probe;
- remove image-order probe;
- remove reasoning counterfactual;
- generic action policy;
- fixed abstention-only policy.

## Gate 8 — statistics

All method comparisons are paired by episode `id`.

Frozen policy:

- 5,000 paired bootstrap samples;
- seed 27;
- 95% confidence intervals;
- exact McNemar test for paired correctness;
- Holm-Bonferroni correction for a family of significance tests;
- report effect sizes and confidence intervals with point estimates.

Use `scripts/compare_runs.py` for machine-readable paired comparisons.

Never report a significance claim from synthetic smoke data.

## Gate 9 — paper export

Only immutable experiment artifacts may populate manuscript tables or figures.

Every official run directory must include:

- `predictions.jsonl`;
- `metrics.json`;
- `risk_coverage.json`;
- paired `statistics.json` / comparison JSON when applicable;
- `provenance.json`.

Any manuscript number that cannot be regenerated from these artifacts is not an official result.

## Gate 10 — final freeze

Internal target: **2026-11-01**.

After freeze, permit only:

- correctness fixes with documented reruns;
- replication of already-frozen experiments;
- paper/supplementary writing;
- anonymous artifact packaging;
- policy/format checks.

Do not add a new method variant after freeze simply because a result table looks weak.
