# Paper-facing artifact contract

Official runs should export at minimum:

- `source_attribution.csv`: model,dataset,method,visual_f1,integration_f1,reasoning_f1,macro_f1
- `failure_calibration.csv`: model,dataset,method,auroc,auprc,ece,brier,aurc
- `intervention.csv`: model,dataset,policy,accuracy,hallucination_rate,coverage,extra_calls,latency_ms
- `ablation.csv`: model,dataset,variant,metric,value
- `risk_coverage.csv`: model,dataset,method,coverage,selective_risk
- `statistics.json`: paired tests, bootstrap CIs, correction method, seeds
- `compute.json`: GPU, VRAM, wall time, tokens, forward passes

Synthetic smoke artifacts must contain `synthetic_only: true` and are never manuscript inputs.
