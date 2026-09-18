# Protected RQ1-RQ3 execution

This protocol prevents post-hoc tuning after calibration or final-holdout data are opened.

## Claim ladder

1. **RQ1 / source semantics**: controlled matched interventions must produce positive source selectivity; a separate natural stage-grounded set is used for source attribution.
2. **RQ2 / propagation**: compare the transparent propagator against an exact-same-source-signal no-propagation control.
3. **RQ3 / action utility**: execute source-conditioned repairs and compare with generic VERIFY/REQUERY at identical additional-call budgets.
4. **RQ4 / transfer**: freeze RQ1-RQ3 estimators and policies before architecture or dataset transfer; no model-specific retuning.

## Freeze before protected data

```bash
python experiments/protected/freeze_protocol.py
```

Commit the resulting provenance manifest before opening Calibration-60. Holdout-360 stays sealed until calibration, source thresholds, propagation weights, action prompts, budgets, metrics, and stopping rules are immutable.

## RQ1-A: fresh matched source-selectivity cohort

Candidate rows must contain `id`, `group_id`, `pair_id`, `control_id`, `source_label`, `origin`, `task`, and upstream IDs where available. Build the deterministic origin-independent manifest while excluding every previous development/calibration/holdout ID:

```bash
python experiments/protected/build_rq1_matched_manifest.py \
  --candidates data/candidates/rq1_source_candidates.jsonl \
  --forbidden-id-file provenance/ids/development_ids.txt \
  --forbidden-id-file provenance/ids/calibration60_ids.txt \
  --forbidden-id-file provenance/ids/holdout360_ids.txt \
  --target-per-source 120 \
  --output data/protected/rq1_matched_manifest.jsonl \
  --manifest provenance/protected/RQ1_MATCHED_MANIFEST.json
```

Never construct the protected source label from intervention intent alone. Human/stage validation remains required before the confirmatory source claim.

Controlled selectivity rows use exactly one control and one intervention per `pair_id`:

```json
{"id":"x","pair_id":"p","group_id":"g","condition":"intervention","source_label":"integration","scores":{"visual":0.1,"integration":0.7,"reasoning":0.2}}
```

Evaluate:

```bash
python experiments/protected/evaluate_rq1.py \
  --input artifacts/protected/rq1_selectivity.jsonl \
  --mode controlled \
  --output artifacts/protected/rq1_selectivity_results.json
```

For natural source attribution, first run the metadata/origin leakage audit:

```bash
python experiments/protected/audit_leakage.py \
  --input artifacts/protected/rq1_natural.jsonl \
  --output artifacts/protected/rq1_leakage_audit.json
```

After the clean/source threshold is frozen on the designated calibration partition:

```bash
python experiments/protected/evaluate_rq1.py \
  --input artifacts/protected/rq1_natural.jsonl \
  --mode natural \
  --clean-threshold <FROZEN_VALUE> \
  --output artifacts/protected/rq1_natural_results.json
```

## RQ2: exact same-signal propagation

Each row contains `id`, `group_id`, `correct`, and the three frozen source scores. Both conditions are constructed from those exact same source scores.

```bash
python experiments/protected/evaluate_rq2.py \
  --input artifacts/protected/rq2_holdout.jsonl \
  --output artifacts/protected/rq2_results.json
```

Primary effects are `propagation - control` for AUROC and `control - propagation` for favorable AURC, with group bootstrap confidence intervals.

## RQ3: matched-budget action efficacy

Execute the predeclared policies at 0/1/2 additional-call budgets:

```bash
python experiments/protected/run_rq3_actions.py \
  --episodes data/protected/holdout360_episodes.jsonl \
  --scores artifacts/protected/holdout360_source_scores.jsonl \
  --adapter qwen \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision <FROZEN_REVISION> \
  --output artifacts/protected/rq3_action_results.jsonl
```

Then evaluate:

```bash
python experiments/protected/evaluate_rq3.py \
  --input artifacts/protected/rq3_action_results.jsonl \
  --output artifacts/protected/rq3_results.json
```

The key causal diagnostic is the one-call **Source x Repair** recovery matrix. The source-conditioned system must not receive more calls or a larger generation budget than generic baselines.

## Non-negotiable rule

If an experiment is negative after the protocol is frozen, report it. Do not lower thresholds, change source labels, add model-specific features, or rerun the final holdout to obtain a more favorable result.
