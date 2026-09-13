# Reproducibility and provenance

Every paper-quality run must save enough information to reproduce both the computation and the **claim boundary**.

## Required run provenance

- repository git SHA;
- exact model identifier + immutable revision, or hosted API model ID + access date;
- dataset identifier + revision + manifest SHA256;
- episode IDs and split role (`development`, `calibration`, `final_holdout`);
- prompt/template and parser hashes/versions;
- generation parameters;
- perturbation seed + transform manifest where applicable;
- Python/PyTorch/Transformers/CUDA/GPU environment;
- model-call counts and wall time;
- output artifact SHA256 values.

Use `PYTHONPATH=src python scripts/check_environment.py` before a local model run when available. Canonical runners should write provenance into the run directory rather than relying on shell history.

## Statistical conventions used by current development results

- shared development cohort: 128 episodes per local model;
- paired Static-vs-Full uncertainty: 10,000 paired bootstrap resamples;
- seed: 2027;
- positive AUROC effect: `Full - Static`;
- favorable AURC effect: `Static - Full`.

## Split discipline

Development diagnostics may shape engineering decisions but must remain labeled as development evidence. Calibration data are used only after the corresponding estimator/policy is frozen. The final holdout is consumed once after upstream decisions are immutable.

## Secret safety

Run `python scripts/check_secrets.py` before pushing. Credentials are configuration of the execution environment, not research artifacts.
