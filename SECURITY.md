# Security and credential policy

U-MIRAGE research runs may use local Hugging Face models and first-party hosted APIs. **No API key, access token, private key, service-account credential, or signed URL belongs in Git.**

## Supported credential variables

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `HF_TOKEN`

Model identifiers may be stored in configuration; credentials may not.

## Local setup

Use `.env.example` only as a template. A populated `.env` is ignored by Git. On Linux/WSL you may load a local file explicitly:

```bash
set -a
source .env
set +a
```

Do not paste secrets into YAML, JSON, notebooks, logs, shell transcripts, result artifacts, or paper sources.

## Before every push

```bash
python scripts/check_secrets.py
git status --short
git diff --cached
```

The CI workflow runs the same tracked-file token-pattern scan.

## If a credential is exposed

1. Revoke/rotate it at the provider immediately.
2. Remove it from the current tree.
3. Audit Git history; deleting the current file does not erase old commits.
4. Rewrite history only after coordinating with collaborators, then force-push deliberately.

Treat all previously exposed credentials as compromised even if the repository was private.
