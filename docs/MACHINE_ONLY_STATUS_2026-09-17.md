# Machine-only development status — 2026-09-17

This note records the latest completed **development** evidence after the machine-only amendment. It does not promote any development result to Calibration-60 or Holdout-360 evidence. Both protected partitions remain sealed.

## RQ1 — construction-intent selectivity stress test

The human-validation arm was skipped under the machine-only amendment. The completed RQ1-A run is therefore a construction-intent stress test only, not human-validated source attribution.

| Intended source | n | Mean selectivity | 95% group-bootstrap CI | Positive rate |
|---|---:|---:|---:|---:|
| Visual | 120 | -0.06537 | [-0.08944, -0.04296] | 0.325 |
| Integration | 120 | +0.00215 | [-0.03513, +0.03809] | 0.592 |
| Reasoning | 120 | -0.09188 | [-0.11767, -0.06811] | 0.250 |

Interpretation: the construction-intent source-selectivity gate failed. Visual and reasoning are negative; integration is null. No redesign is performed from the observed signs.

## RQ2 — exact same-signal propagation

P81 natural-stage development bank, n=128, seed 27, 5,000 paired group-bootstrap resamples.

| Condition | AUROC | AURC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| No propagation | 0.68621 | 0.23836 | 0.62401 | 0.27543 | 0.25632 |
| Propagation | 0.68380 | 0.23972 | 0.61616 | 0.27318 | 0.25260 |

Primary paired effects:
- Delta AUROC (propagation - control): -0.00241, 95% CI [-0.01301, +0.00851].
- Favorable Delta AURC (control - propagation): -0.00137, 95% CI [-0.01502, +0.01050].

Interpretation: no primary propagation benefit is established. Small favorable Brier/ECE changes are secondary only.

## RQ3 — matched-budget action run

The P81 machine-only execution completed all 128 episodes and all 1,920 policy-budget rows. Raw SHA256:

`a37cd2215a1788183505e17195c0591a51fc27a6b0bd39535d3be3a51572c803`

Frozen exact-match scoring produced:
- base / no-extra-call accuracy: 0.64844;
- every nonzero-budget U-MIRAGE, VERIFY, REQUERY, and random-source cell: 0.00000;
- every paired U-MIRAGE-vs-VERIFY / U-MIRAGE-vs-REQUERY difference: 0.00000.

This all-policy collapse is treated as a **degenerate output-validity result**, not as evidence that all repair policies are truly equivalent or universally harmful. The raw rows retain final answers, so output-shape / parser diagnostics can be performed without rerunning the expensive model calls. Any relaxed parser is post-hoc diagnostic only and must not replace the frozen primary estimator.

No Source x Repair causal claim is made on this machine-only P81 run. The random-source arm is auxiliary.

## RQ4 — hosted access regime and executor transfer

The common black-box uncertainty interface remained invalid: both hosted endpoints supported text and multi-image calls, but no shared log-probability interface was available and all three predeclared common-observable fallbacks failed their common usability gate.

A separate exploratory executor-transfer study then used 32 fresh P81 development episodes, frozen Qwen3 source diagnoses, and matched one-call conditions.

| Model | Base | Source-conditioned | VERIFY | REQUERY |
|---|---:|---:|---:|---:|
| gpt-6-astra | 0.8438 | 0.8438 | 0.8438 | 0.8438 |
| gemini-3.8-flash | 0.6250 | 0.6875 | 0.5625 | 0.6250 |

Paired exploratory effects:
- gpt-6-astra: source - VERIFY = +0.0000, 95% CI [-0.0938, +0.0938]; source - REQUERY = +0.0000, 95% CI [0, 0].
- gemini-3.8-flash: source - VERIFY = +0.1250, 95% CI [-0.0625, +0.3125]; source - REQUERY = +0.0625, 95% CI [-0.0938, +0.2500].

All hosted executor intervals include zero. Gemini parse rates also differ by condition, so the point differences must not be attributed solely to source-conditioned repair quality.

## Protected boundary

- Calibration-60 opened: **false**
- Holdout-360 opened: **false**
- Human RQ1 validation completed: **false**
- Development evidence promoted to protected final evidence: **false**

Negative and null outcomes remain visible.