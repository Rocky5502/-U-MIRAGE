# Paper-facing result contract

Official aggregation should export:
- `source_attribution.csv`: model,benchmark,method,visual_f1,integration_f1,reasoning_f1,macro_f1
- `failure_calibration.csv`: model,benchmark,method,auroc,auprc,ece,brier,aurc
- `intervention.csv`: model,benchmark,policy,accuracy,hallucination_rate,coverage,extra_calls,latency_ms
- `ablation.csv`: model,benchmark,variant,auroc,macro_f1,aurc
- `statistics.json`: paired comparisons, confidence intervals, multiple-test correction
- `provenance.json`: git/model/data/prompt/environment hashes

No paper table may be populated from synthetic artifacts.
