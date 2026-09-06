# U-MIRAGE Pilot Gate

This document freezes the decision gate that must pass before the project scales to the full CVPR 2027 experiment matrix.

## Pilot objective

Test whether **visual**, **cross-image integration**, and **reasoning** uncertainty are empirically separable and whether source-aware routing provides value beyond a scalar risk score.

## Pilot cohort

- 500–1,000 MIRAGE-U episodes.
- Balanced across `clean`, `visual`, `integration`, and `reasoning` source labels.
- 2 primary open VLMs: Qwen3-VL-8B-Instruct and Qwen2.5-VL-7B-Instruct.
- Fixed prompts, decoding settings, perturbation seeds, and image manifests.
- A human-verified subset must be kept separate from any calibrator fitting.

## Required outputs

Every run must emit the result-schema artifacts documented in `docs/RESULT_SCHEMA.md`, including per-episode:

- correctness,
- visual uncertainty,
- integration uncertainty,
- reasoning uncertainty,
- propagated risk,
- source posterior,
- predicted source,
- selected action,
- model/checkpoint identifier,
- dataset/source identifier,
- seed and perturbation metadata.

## Primary gate metrics

1. **Source Macro-F1** on the source-labelled subset.
2. **Failure AUROC / AUPRC** using propagated risk.
3. **AURC** and selective risk at fixed coverage.
4. **Brier score / ECE** for risk calibration.
5. **Intervention utility** at matched additional model calls.
6. **Per-source confusion matrix** to check whether integration uncertainty is genuinely distinct from perception/reasoning.

## Go criteria

Proceed to the full four-family sweep when all of the following hold:

- Source Macro-F1 is substantially above the 1/3 random-source baseline on both primary models.
- Integration-source F1 is non-trivial and does not collapse into the visual class.
- Propagated risk improves at least one ranking/selective metric over the strongest scalar component without materially degrading the others.
- Source-conditioned routing beats a matched-cost generic re-query policy on at least one reliability metric.
- Results are directionally consistent across both model families.

These are scientific decision criteria, not promised paper results. Exact statistical thresholds are frozen only after the pilot sample size and class balance are finalized.

## Pivot criteria

Do **not** scale blindly if any of the following occur:

- integration uncertainty is not separable from visual uncertainty;
- leave-one-image-out alone matches the full source model;
- source-conditioned intervention is no better than generic verification;
- source labels show poor human agreement;
- results depend strongly on one model family or one perturbation recipe.

If a pivot is triggered, preserve the pilot artifacts and update the research claim before adding more compute.

## Run order

1. Environment check and deterministic mock tests.
2. Dataset manifest validation.
3. 50-episode smoke run per model.
4. 500–1,000 episode pilot.
5. Aggregate metrics and bootstrap confidence intervals.
6. Inspect source confusion + failure examples.
7. Make the scale/pivot decision.

## Reproducibility rule

No manual editing of result numbers. Paper tables and figures must be generated from frozen machine-readable artifacts.
