# Dataset registry and license policy

U-MIRAGE does **not** redistribute third-party benchmark media by default. Store manifests, IDs, hashes, and derived metadata unless upstream terms explicitly permit redistribution.

## Current development evidence

| Resource | Role | Public link | Notes |
|---|---|---|---|
| MIHBench | Primary multi-image hallucination development benchmark | https://huggingface.co/datasets/MM-Hallu/MIHBench | HF card reports CC BY 4.0; source images include COCO content. |
| MIHBench paper | Benchmark description | https://arxiv.org/abs/2508.00726 | ACM Multimedia 2025 paper. |
| MIHBench/DAB code | Upstream implementation/questions | https://github.com/pgtrece/DAB | Respect upstream code/data terms. |
| MS COCO | Instance annotations / source images | https://cocodataset.org/ | COCO terms apply independently of derived benchmark metadata. |
| POPE | Object-presence audit reference | https://github.com/RUCAIBox/POPE | Official code/data repository. |
| POPE paper | Object hallucination evaluation | https://aclanthology.org/2023.emnlp-main.20/ | EMNLP 2023. |

## Additional generalization candidates

These are useful external benchmarks but are not automatically part of the current headline development results.

| ID | Intended role | Public link | Notes |
|---|---|---|---|
| `mioh` | Fine-grained multi-image object hallucination | https://openaccess.thecvf.com/content/CVPR2026/html/Min_Fine-Grained_Multi_Image_Object_Hallucination_Benchmark_CVPR_2026_paper.html | CVPR 2026 paper; verify data release/license before caching. |
| `vischainbench` | Progressive multi-image reasoning | https://huggingface.co/datasets/eyehole/VisChainBench | Dataset card lists CC BY 4.0; verify upstream media terms. |
| `rest_plus` | Cross-modal consistency stress test | https://github.com/angelavansprang/Same-Content-Different-Answers | REST/REST+ paper: CVPR 2026; release status may evolve. |
| `hallusionbench` | Hallucination/illusion generalization | https://huggingface.co/datasets/rayguan/HallusionBench | Review benchmark/media licensing before redistribution. |
| `mmmu` | Broad multimodal reasoning generalization | https://huggingface.co/datasets/MMMU/MMMU | Dataset card lists Apache-2.0; source-item rights still matter. |

## Required before a paper-quality run

1. Record upstream URL, immutable revision/commit where available, access date, and license.
2. Generate a manifest containing example IDs and SHA256 hashes.
3. Never commit restricted/raw media merely for convenience.
4. Derived perturbations must be reproducible from transform name + parameters + seed.
5. Keep development, calibration, and final-holdout manifests distinct.
