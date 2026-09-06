# Reproducibility and provenance

Every official run must save:

- repository git SHA,
- exact model identifier + revision,
- dataset identifier + revision + manifest hash,
- prompt/template hash,
- generation parameters,
- perturbation seed + transform manifest,
- Python/PyTorch/CUDA/GPU environment,
- wall time and model-call counts.

Use `PYTHONPATH=src python scripts/check_environment.py` before the first local run. Official runners should call `umirage.provenance.write_manifest` into the run directory.
