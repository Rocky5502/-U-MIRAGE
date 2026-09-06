# Experiment protocol

1. **Pilot gate first:** 500 controlled MIRAGE-U episodes on Qwen3-VL-8B and Qwen2.5-VL-7B.
2. Freeze model/dataset revisions and prompt templates before official runs.
3. Report visual/integration/reasoning source attribution, failure prediction, calibration, selective risk, and action efficiency.
4. Use paired bootstrap confidence intervals and fixed seeds; keep raw predictions for audit.
5. Scale to additional model families only if integration uncertainty adds information beyond simple visual/reasoning baselines.
6. Synthetic smoke artifacts are excluded from every paper result directory.
