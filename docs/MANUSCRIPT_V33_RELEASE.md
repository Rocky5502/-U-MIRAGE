# U-MIRAGE CVPR 2027 manuscript release v33

**Release date:** 2026-10-29. This record documents the manuscript packages prepared from v32, synchronized with the completed RQ3 machine-only development execution and the exploratory hosted RQ4c executor-transfer study.

## Manuscript packages

- Full scientific manuscript: `UMIRAGE_CVPR2027_v33_FULL_RESEARCH_OVERLEAF.zip` (SHA256 `e197648d1e2cff19c8b957f6e7eab17a94b401a1d97413a6ccd0ccbd3d6da2ef`), 15-page compiled main.
- Compact CVPR-format candidate: `UMIRAGE_CVPR2027_v33_COMPACT_CVPR_OVERLEAF.zip` (SHA256 `fdf7851a6118bfa9a15868b54a995781b0c5a9035f387671f64456faf180035e`), 7-page compiled main including the LLM Usage Statement and bibliography.
- Supplement: 6 compiled pages.

These are delivery artifacts attached to the research conversation, **not claimed to have been uploaded to GitHub or OpenReview**. The experiment runners and output-contract audit are already present on `main`: `experiments/protected/run_rq3_primary_cached.py`, `experiments/protected/evaluate_rq3.py`, `experiments/rq4_blackbox/run_executor_transfer.py`, and `scripts/audit_rq3_output_contract.py`.

## Frozen scientific boundaries

- RQ1: 360 construction-intent matched pairs fail source selectivity; the human annotation stage was not completed.
- RQ2: exact-same-source-signal propagation is null/slightly unfavorable on 128 natural development episodes.
- RQ3: 128 × 5 × 3 = 1,920 policy-budget rows completed; base accuracy 83/128. All nonzero-budget actions score zero under the frozen exact-match evaluator. Four principal paired differences are zero with degenerate [0,0] CIs. **Comparative repair efficacy is blocked pending a read-only raw-answer/output-contract audit**. Original raw input SHA256: `a37cd2215a1788183505e17195c0591a51fc27a6b0bd39535d3be3a51572c803`.
- RQ4: four-model feature-enrichment effects heterogeneous; hosted uncertainty interface gate fails. RQ4c executor-only `gpt-6-astra` / `gemini-3.8-flash` 32-case development pilot has no paired CI excluding zero.
- Calibration-60 and Holdout-360 remain sealed; no final protected-result claims are made.

## Submission gates — do not claim CVPR readiness yet

The full manuscript **exceeds** CVPR 2027's eight-page technical-content cap and is research-archive only. The compact paper fits the current length estimate but still requires official 2027 author-kit verification, coauthor OpenReview enrollment, peer review of exact source claims, and resolution of the RQ3 output-contract anomaly. Both include an anonymous LLM Usage Statement.

Official 2027 AoE deadlines: **November 10** paper enrollment and complete author profiles, **November 16** main paper, **November 23** supplement. See [CVPR 2027 guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines) and [dates](https://cvpr.thecvf.com/Conferences/2027/Dates).

No experimental results, thresholds, prompts, model identities, or protected splits were altered in preparing this document.
