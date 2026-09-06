# Model adapter contract

The experiment layer depends only on `MultiImageAdapter.answer(Episode, ...) -> ModelOutput`.

## Adapters

- `MockMultiImageAdapter`: deterministic CI/plumbing only; never a scientific baseline.
- `QwenVLAdapter`: preferred local path for Qwen2.5-VL / Qwen3-VL using chat templates and `qwen-vl-utils` when available.
- `TransformersVLMAdapter`: generic best-effort Hugging Face boundary for compatible image-text-to-text models.

Model-family preprocessing must remain isolated in adapters. Do not add model-specific branches to the uncertainty pipeline.

## Official-run rules

1. Freeze exact Hub model ID and immutable revision/commit.
2. Record `trust_remote_code`, dtype, quantization, device map, max tokens, and decoding settings.
3. Run 20 examples first and manually inspect decoded answers before scaling.
4. Do not compare API and local models without recording exact API model ID and access date.
