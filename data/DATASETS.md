# Dataset registry and license policy

U-MIRAGE does **not** redistribute third-party benchmark media by default.

| ID | Intended role | Upstream | Notes |
|---|---|---|---|
| `mioh` | Primary multi-image hallucination | CVPR 2026 MIOH release | Verify exact release license before caching/redistribution. |
| `vischainbench` | Progressive multi-image reasoning | eyehole/VisChainBench | Dataset card reports CC BY 4.0; verify upstream media terms. |
| `mihbench` | Controlled multi-image diagnostic | MM-Hallu/MIHBench | Dataset card reports CC BY 4.0; COCO image terms still apply. |
| `rest_plus` | Cross-modal consistency | Same-Content-Different-Answers | Store only manifests unless upstream license permits more. |
| `hallusionbench` | Generalization | HallusionBench | Public benchmark; verify each media source. |
| `mmmu` | Utility/generalization | MMMU-Benchmark/MMMU | Repo licensing does not automatically override source-item rights. |
| `vlm_uqbench` | UQ stress test | 2026 preprint/release | Treat as non-peer-reviewed until venue status changes. |

## Required before an official run

1. Record upstream URL, commit/revision, access date, and license in `configs/datasets/*.yaml`.
2. Generate a manifest containing example IDs and SHA256 hashes.
3. Never commit restricted/raw media to this repository.
4. Derived perturbations must be reproducible from a transform name + parameters + seed.
