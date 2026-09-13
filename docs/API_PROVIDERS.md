# First-party API provider configuration

U-MIRAGE keeps hosted-model credentials outside version control. This document records **first-party** provider entry points and environment-variable conventions; it does not store keys and does not imply that every provider exposes the same uncertainty signals.

## Install provider SDKs

```bash
pip install -e '.[api]'
```

## OpenAI

- Official API documentation: https://developers.openai.com/api/
- Responses API: https://developers.openai.com/api/reference/resources/responses/
- Credential variable: `OPENAI_API_KEY`
- Model variable used by U-MIRAGE experiment wrappers: `OPENAI_MODEL`

Minimal client initialization:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = os.environ["OPENAI_MODEL"]
```

OpenAI's Responses API supports text and image inputs. Freeze the exact model identifier and access date in the run manifest rather than hard-coding a moving alias into result tables.

## Anthropic Claude

- Official documentation: https://docs.anthropic.com/
- Credential variable: `ANTHROPIC_API_KEY`
- Model variable: `ANTHROPIC_MODEL`

```python
import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
model = os.environ["ANTHROPIC_MODEL"]
```

Use the official Messages/vision documentation for image payloads. Record exact model IDs because provider model lifecycles can change.

## Google Gemini

- Official Gemini API documentation: https://ai.google.dev/gemini-api/docs
- GenerateContent reference: https://ai.google.dev/api/generate-content
- Credential variable: `GEMINI_API_KEY`
- Model variable: `GEMINI_MODEL`

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = os.environ["GEMINI_MODEL"]
```

Gemini supports multimodal content, including image inputs. Freeze the exact model string and access date per run.

## Hugging Face Hub

- Token documentation: https://huggingface.co/docs/hub/security-tokens
- Credential variable: `HF_TOKEN`

Public models normally do not require a token. Use a fine-grained/read token only when gated or private assets require authentication.

## Scientific access-regime rule

Hosted APIs and local open-weight models do not necessarily expose the same uncertainty observables. U-MIRAGE therefore does **not** substitute model-specific privileged signals after seeing results. A hosted-model efficacy extension is run only when a common predeclared observable interface passes the same usability gate across compared models.
