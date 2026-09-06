# Local pilot run

Recommended for the Windows lab machine: use **WSL2 Ubuntu** for the final reproducible environment.

```bash
git clone https://github.com/Rocky5502/-U-MIRAGE.git
cd -- -U-MIRAGE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,vlm,data,analysis]'
pytest -q
python scripts/make_demo_data.py
python scripts/run_smoke.py
nvidia-smi
```

## First real GPU gate

1. Verify **dedicated** VRAM with `nvidia-smi`.
2. Freeze one Qwen model revision in `configs/models/`.
3. Normalize only a small pilot subset to the canonical JSONL schema.
4. Start with **20 examples**, then **100**, then the **500-example MIRAGE-U pilot**.
5. Inspect decoded outputs, source scores, latency, and provenance after each gate.
6. Do not launch the four-family sweep until integration uncertainty adds predictive value beyond visual + reasoning uncertainty.

Example:

```bash
python scripts/run_jsonl.py \
  --input data/cache/mirage_u_pilot.jsonl \
  --output artifacts/pilot/qwen3vl8b \
  --adapter qwen \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision <FROZEN_REVISION>
```
