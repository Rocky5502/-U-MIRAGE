# U-MIRAGE compute plan

This document freezes the compute policy before the first real pilot.

## Local lab machine

Development and pilot machine:

- OS: Windows 10 Pro; final reproducible runs should use WSL2 Ubuntu unless a Windows-only dependency is required.
- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell).
- Dedicated VRAM: **16 GB GDDR7 by NVIDIA reference specification**. Windows may display roughly 48 GB total GPU memory because shared system memory is included; official manifests must record the `nvidia-smi` dedicated-memory value rather than the Windows aggregate.
- System RAM: 64 GB.
- Storage: Samsung SSD 990 PRO 2 TB.
- CPU: Intel system; exact model is TBD and must be frozen from the local machine before official runs.

Every official run records GPU name, driver, CUDA runtime, PyTorch version, dedicated VRAM, CPU model, RAM, storage path, git SHA, wall time, peak VRAM, and model/dataset revisions.

## Compute tiers

### Tier L0 - CPU / mock validation
Use for schema validation, unit tests, deterministic smoke data, table/figure builders, and provenance checks. No paper result can come from synthetic data.

### Tier L1 - local 5070 Ti, 16 GB dedicated VRAM
Use for:

- Qwen2.5-VL-7B and Qwen3-VL-8B pilot inference with 4-bit or 8-bit loading where necessary;
- batch size 1 and gradient-free inference;
- MIHBench 20/50/100-example smoke gates;
- 500-example pilot if per-example latency is practical;
- cached visual/text features;
- propagation/calibration training;
- black-box perturbation runs;
- result aggregation and plotting.

Default safeguards:

- deterministic decoding for official comparisons;
- explicit image-count / resolution caps;
- no full VLM fine-tuning;
- stop on CUDA OOM rather than silently changing precision or input resolution;
- record quantization, dtype, attention backend, and processor settings.

### Tier C1 - rented 24-32 GB GPU
Use only if a specific 7B-13B model or higher image resolution does not fit cleanly on L1. This is a compatibility tier, not a default.

### Tier C2 - rented 48 GB GPU
Use for final 7B-13B multi-image sweeps at BF16/FP16, larger image counts, white-box feature extraction, and throughput-sensitive replications.

### Tier C3 - rented 80 GB GPU (A100/H100-class or equivalent)
Use when scientifically justified for:

- larger model-family validation;
- dense high-resolution multi-image evaluation;
- large white-box activation/logit extraction;
- final parallel sweeps where wall-clock time becomes the bottleneck.

Renting is gated by a written experiment ID and expected scientific value. We do not rent accelerators to compensate for an unvalidated hypothesis.

## Cloud provider policy

AI Galaxy is an available operational provider. The manuscript should report accelerator SKU/VRAM/driver/runtime rather than rely on a provider name for scientific meaning. Provider, instance ID, rental period, and cost are retained in internal provenance.

## Closed-model / API policy

Closed multimodal APIs are **evaluation-only**. They do not define U-MIRAGE's method or training target.

An OpenAI-compatible provider can be configured with environment variables such as:

- `UMIRAGE_API_BASE_URL`
- `UMIRAGE_API_KEY`
- `UMIRAGE_API_MODEL`

Never commit API keys. Official API runs must freeze exact model ID, provider endpoint family, date/time, prompt hash, temperature, max output tokens, retry policy, and call count. API models should be used only after the open-model pilot passes the scientific gate.

## Scale gate

Do not move from Tier L1 to rented C2/C3 compute until the pilot shows all of the following:

1. integration uncertainty is measurably separable from visual and reasoning uncertainty;
2. propagated risk improves at least one ranking/selective metric over the strongest scalar component without material regressions;
3. source-conditioned routing beats a matched-cost generic verification baseline on at least one reliability metric;
4. findings are directionally consistent across Qwen3-VL-8B and Qwen2.5-VL-7B;
5. human/source labels are reliable enough to support the claim.

If the gate fails, pivot the claim or signal design before buying more compute.
