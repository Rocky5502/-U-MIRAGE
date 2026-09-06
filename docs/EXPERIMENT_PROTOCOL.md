# Frozen experiment protocol

## Gate 0 — environment
Capture `git rev-parse HEAD`, Python/PyTorch/CUDA versions, GPU model + dedicated VRAM, OS, CPU/RAM, and model cache revisions.

## Gate 1 — MIRAGE-U pilot
Run 500–1,000 source-controlled episodes on two model families. **Do not scale** unless integration uncertainty adds predictive information beyond visual + reasoning uncertainty.

## Gate 2 — benchmark sweep
Primary: MIOH, VisChainBench, MIHBench. Generalization: REST+, HallusionBench, MMMU/Pro, VLM-UQBench where permitted.

## Gate 3 — model families
Freeze exact model revisions. Primary intended families: Qwen3-VL, Qwen2.5-VL, InternVL3.5, Molmo 2.

## Gate 4 — statistics
Paired evaluation, bootstrap confidence intervals, correction for multiple paired tests, effect sizes, and explicit seed count.

## Gate 5 — paper export
Only immutable experiment artifacts can populate manuscript tables/figures.
