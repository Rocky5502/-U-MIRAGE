# Model adapter and provider contract

The reusable experiment layer depends on `MultiImageAdapter.answer(Episode, ...) -> ModelOutput`.

## Core local adapters

- `MockMultiImageAdapter`: deterministic CI/plumbing only; never a scientific baseline.
- `QwenVLAdapter`: local Qwen2.5-VL / Qwen3-VL path using Hugging Face chat templates and `qwen-vl-utils` when available.
- `TransformersVLMAdapter`: generic best-effort Hugging Face boundary for compatible image-text-to-text models.

Model-family preprocessing must remain inside adapters. Do not add model-specific branches to the uncertainty estimator merely to improve one model's result.

## Development model families

| Family | Official Hub repository |
|---|---|
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct |
| Qwen2.5-VL-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |
| InternVL3.5-8B-Instruct | https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct |
| Phi-4-multimodal-instruct | https://huggingface.co/microsoft/Phi-4-multimodal-instruct |

The exact immutable revision used for a paper-quality run must come from frozen provenance. Do not infer a historical run revision from today's Hub `main`.

## Hosted providers

First-party SDK configuration for OpenAI, Anthropic, Google Gemini, and Hugging Face is documented in [`API_PROVIDERS.md`](API_PROVIDERS.md). Hosted provider support must preserve the same scientific interface contract:

1. Freeze exact provider model ID and access date.
2. Record the observable uncertainty interface.
3. Do not give one provider a privileged signal unavailable to the others in the same comparison.
4. If the common-interface usability gate fails, stop rather than post-hoc lowering thresholds or inventing provider-specific estimators.

## Official-run metadata

Record at minimum:

- model ID and immutable revision (or API model ID + access date),
- trust/remote-code setting,
- dtype and quantization,
- device map,
- generation parameters,
- prompt/parser version,
- run git SHA and dataset manifest SHA.
