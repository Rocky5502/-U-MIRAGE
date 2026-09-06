# U-MIRAGE — CVPR 2027 execution plan

## Deadline status

As of 2026-09-06, the official CVPR 2027 site is live but the 2027 paper submission dates have not yet been posted. The prior CVPR cycle used an early-November abstract deadline and a mid-November paper deadline. Treat that pattern only as a planning assumption, not as an official CVPR 2027 deadline.

Internal target: freeze all paper-critical experiments by **2026-11-01** so the final submission window is used for writing, checks, and contingency rather than new science.

## Scientific objective

The paper succeeds only if it establishes all three parts of the claim:

1. source-specific uncertainty is measurable in multi-image reasoning;
2. explicit source-aware propagation improves reliability estimation beyond scalar and unpropagated baselines; and
3. source-conditioned interventions recover accuracy more efficiently than generic verification at matched compute.

If any one of these fails, pivot the claim instead of scaling blindly.

## Execution order

### Stage A — local correctness gate

Run before any benchmark sweep:

- `pytest -q`
- deterministic mock smoke
- environment + GPU capture
- 20 real examples
- manual audit of every prediction, source posterior, perturbation diagnostic, and action

Do not scale if image order, leave-one-image-out, or reasoning diagnostics are malformed.

### Stage B — 50/100-example compatibility gate

Primary models:

- Qwen3-VL-8B
- Qwen2.5-VL-7B

Goals:

- verify processor/model compatibility;
- measure wall-clock and VRAM;
- verify generated-token confidence scoring;
- inspect visual/integration/reasoning signal distributions;
- confirm source posterior is not degenerate;
- validate that repeated deterministic runs reproduce the same outputs.

### Stage C — 500-example scientific pilot

Use frozen MIRAGE-U source labels with balanced source classes.

Required decision tests:

- source Macro-F1 versus majority-source and scalar-confidence baselines;
- integration signal added on top of visual + reasoning signals;
- propagation versus the same source signals without propagation;
- U-MIRAGE versus generic verify at identical extra-call budget;
- bootstrap confidence intervals on all paper-primary deltas.

Scale to 1,000 only after the 500-example gate is scientifically positive or diagnostically informative.

### Stage D — external benchmark sweep

Priority order:

1. MIOH
2. VisChainBench
3. MIHBench
4. HallusionBench / REST+ for reliability transfer
5. MMMU or MMMU-Pro for broader reasoning transfer

The paper does not need every available benchmark. Prefer complete, paired, reproducible evaluation on the frozen primary suite over many shallow datasets.

### Stage E — model-family generalization

After the two-model pilot passes, expand to:

- InternVL3.5 8B-class
- Molmo 2 8B-class

Closed/API models are evaluation-only and must not be required for the method to work.

### Stage F — matched-compute intervention study

Budgets: 0, 1, and 2 extra model calls.

Compare:

- answer only;
- global abstention;
- generic verify;
- generic re-query;
- source-conditioned U-MIRAGE.

Every method must use the exact same base episodes. Report accuracy, hallucination rate, recovered accuracy per extra call, mean extra calls, and latency.

### Stage G — mandatory ablations

Minimum ablations:

- remove visual source;
- remove integration source;
- remove reasoning source;
- no propagation;
- equal source weights;
- no leave-one-image-out probe;
- no image-order probe;
- no reasoning counterfactual;
- generic action policy;
- fixed abstention only.

Do not postpone these until the last week. They are part of the main scientific argument.

## Statistics policy

All method comparisons use paired episodes keyed by `id`.

- 5,000 paired bootstrap samples, seed 27;
- 95% confidence intervals;
- exact McNemar test for paired correctness;
- Holm-Bonferroni correction when reporting a family of significance tests;
- effect sizes and confidence intervals reported with point estimates;
- no significance claim from synthetic smoke data.

Use `scripts/compare_runs.py` to generate machine-readable comparison files.

## Paper artifact policy

The manuscript may only use numbers generated from immutable experiment directories containing:

- `predictions.jsonl`
- `metrics.json`
- `risk_coverage.json`
- `statistics.json` or paired comparison JSON
- `provenance.json`

No manual spreadsheet value becomes an official result unless it can be regenerated from these artifacts.

## Internal timeline

### Sep 6–13
- merge/validate pilot core locally;
- finish 20 → 50 → 100 real-example path;
- freeze model revisions and benchmark materializers.

### Sep 14–27
- run balanced 500-example MIRAGE-U pilot;
- decide GO / PIVOT;
- complete source-separability analysis.

### Sep 28–Oct 11
- external benchmark sweep on two primary models;
- matched-compute intervention study;
- first full paper tables populated.

### Oct 12–24
- expand model families;
- complete mandatory ablations;
- run generalization and robustness checks;
- freeze primary figures.

### Oct 25–Nov 1
- replicate critical experiments;
- run final paired statistics;
- lock results and provenance;
- no new method variants after freeze unless fixing a correctness bug.

### After Nov 1
- writing, supplementary, reproducibility package, anonymous repository snapshot, format/policy checks, and contingency only.

## Kill / pivot rules

Pivot rather than overspend compute if any of these persist after the 500-example pilot:

1. integration uncertainty is not distinguishable from visual uncertainty;
2. leave-one-image-out alone matches the full propagation pipeline;
3. source-conditioned intervention does not beat generic verification at matched compute;
4. source labels have poor human agreement;
5. gains exist only on one model or one dataset.

A strong negative/diagnostic paper is preferable to an over-claimed positive paper.
