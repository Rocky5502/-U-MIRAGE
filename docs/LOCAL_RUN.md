# Local pilot run

Recommended on the Windows lab machine: use WSL2 Ubuntu for the final reproducible environment.

```bash
git clone <this-repository>
cd -- -U-MIRAGE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,vlm,data]'
pytest -q
python scripts/run_smoke.py
nvidia-smi
```

Then prepare/freeze a canonical pilot JSONL and run:

```bash
python scripts/run_jsonl.py \
  --input data/cache/mirage_u_pilot.jsonl \
  --output artifacts/pilot/qwen3vl8b \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision <FROZEN_REVISION>
```

Before the first real call verify dedicated VRAM with `nvidia-smi`. Start with 20 examples, then 100, then the 500-example pilot gate. Do not launch the four-family sweep before the pilot demonstrates separable integration uncertainty.
