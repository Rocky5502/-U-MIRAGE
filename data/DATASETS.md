# Dataset policy

U-MIRAGE never silently vendors third-party benchmark media. Official runs pin URL/revision/access date/license and store only IDs, hashes, perturbation manifests, and permitted derivatives.

| ID | Role | Expected source |
|---|---|---|
| `mioh` | primary multi-image hallucination | CVPR 2026 release |
| `vischainbench` | progressive multi-image reasoning | `eyehole/VisChainBench` |
| `mihbench` | controlled multi-image diagnostic | `MM-Hallu/MIHBench` |
| `rest_plus` | cross-modal inconsistency | official REST/REST+ release |
| `hallusionbench` | generalization | official release |
| `mmmu` | utility/generalization | official MMMU release |
| `vlm_uqbench` | UQ stress test | 2026 release; preprint status tracked |

Before any paper run: verify current licenses, freeze revisions, hash manifests, and document exclusions.
