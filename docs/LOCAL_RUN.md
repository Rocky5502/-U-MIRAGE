# Local reproducible run

For the Windows workstation, WSL2 Ubuntu is the recommended reproducible environment.

```bash
git clone https://github.com/Rocky5502/-U-MIRAGE.git U-MIRAGE
cd U-MIRAGE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,vlm,data,analysis]'
pytest -q
python scripts/check_secrets.py
python scripts/make_demo_data.py
python scripts/run_smoke.py
nvidia-smi
```

## Real local inference

Always freeze the model identifier/revision and dataset manifest before a paper-quality run.

```bash
python scripts/run_jsonl.py \
  --input <normalized.jsonl> \
  --output artifacts/dev/qwen3vl8b \
  --adapter qwen \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision <FROZEN_REVISION>
```

Recommended checks:

1. Verify dedicated GPU memory with `nvidia-smi`.
2. Confirm the normalized input IDs and manifest SHA256.
3. Run a small smoke subset before scaling.
4. Inspect decoded answers and parser behavior.
5. Record exact model revision, environment, prompts, seeds, and git SHA.
6. Keep development, calibration, and final-holdout outputs in distinct directories.
7. Never modify a frozen estimator or threshold after inspecting the sealed holdout.

## Hosted APIs

Install the optional provider SDKs:

```bash
pip install -e '.[api]'
```

Then follow [`API_PROVIDERS.md`](API_PROVIDERS.md). Provider keys must remain in the local environment, never in repository files or experiment logs.
