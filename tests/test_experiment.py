import json
from pathlib import Path

from umirage.adapters.mock import MockMultiImageAdapter
from umirage.experiment import run_experiment


def test_experiment_writes_provenance_and_metrics(tmp_path: Path):
    inp = tmp_path / "x.jsonl"
    inp.write_text(json.dumps({
        "id": "x", "images": ["virtual://a", "virtual://b"], "question": "q",
        "answer": "alpha", "source_label": "clean",
        "metadata": {"synthetic": {"risk": 0.05, "wrong_answer": "beta"}},
    }) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    m = run_experiment(inp, MockMultiImageAdapter(), out, {"seed": 7})
    assert m["synthetic_only"] is True
    assert (out / "predictions.jsonl").exists()
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["seed"] == 7
    assert prov["input_sha256"]
