# U-MIRAGE CVPR 2027 paper freeze v6

This repository state is synchronized with the **v6 single-manuscript Overleaf freeze**.

## Manuscript policy

- The paper has one main manuscript and no supplementary manuscript.
- `references.bib` is the paper's single bibliography database.
- All six manuscript figures are rendered as PDF assets; result figures contain only empty axes before official runs.
- Every empirical table cell is intentionally empty before aggregation.
- No illustrative or synthetic value is permitted in a manuscript table or plot.

## Paper artifact bootstrap

Run:

```bash
python scripts/init_paper_results.py --output-dir results/paper
```

This creates **header-only CSVs** and empty JSON arrays. It does not create any empirical value.

Canonical outputs:

- `source_attribution.csv`
- `source_selectivity.csv`
- `failure_calibration.csv`
- `intervention.csv`
- `ablation.csv`
- `transfer.csv`
- `risk_coverage.csv`
- `source_confusion.csv`
- `statistics.json`
- `compute.json`
- `paper_results_manifest.json`

## Frozen claim ladder

### RQ1 — identifiability
A source decomposition is useful only if both conditions hold:
1. source Macro-F1 materially exceeds trivial/scalar baselines; and
2. matched source interventions have positive source selectivity.

### RQ2 — propagation value
Compare U-MIRAGE against **the exact same source features without propagation** on identical episode IDs. If propagation does not improve failure ranking/selective risk, do not claim propagation as necessary.

### RQ3 — operational value
Source-conditioned actions must beat generic verification/re-query at identical base episodes and the same 0/1/2 extra-call budget.

### RQ4 — transfer
Report held-out model-family, dataset, image-count, and corruption-severity transfer; no same-distribution-only generalization claim is sufficient.

## Statistical policy

- pairing key: `id`
- paired bootstrap: 5,000 samples
- seed: 27
- confidence level: 95%
- paired correctness: exact McNemar
- multiple comparisons: Holm-Bonferroni within predeclared families

## Local-to-scale gate

20 -> 50 -> 100 compatibility checks -> 500 source-balanced confirmatory pilot -> optional 1,000+ scale-up.

Do **not** rent large accelerators or activate a learned propagator until the 500-example pilot demonstrates source selectivity, propagation value, and matched-budget action value.

## Integrity rule

A number is paper-ready only when the raw run, aggregate artifact, statistics artifact, and provenance agree on cohort IDs and size. Corrections create a new run ID; frozen `predictions.jsonl` files are never edited in place.
