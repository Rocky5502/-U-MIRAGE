import json
from pathlib import Path

from umirage.adapters.mock import MockMultiImageAdapter
from umirage.experiment import run_experiment


def _episode(idx: int) -> dict:
    return {
        "id": f"x{idx}",
        "images": ["virtual://a", "virtual://b"],
        "question": "q",
        "answer": "alpha",
        "source_label": "clean",
        "metadata": {"synthetic": {"risk": 0.05, "wrong_answer": "beta"}},
    }


def test_experiment_writes_provenance_and_metrics(tmp_path: Path):
    inp = tmp_path / "x.jsonl"
    inp.write_text(json.dumps(_episode(0)) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    m = run_experiment(inp, MockMultiImageAdapter(), out, {"seed": 7})
    assert m["synthetic_only"] is True
    assert (out / "predictions.jsonl").exists()
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["seed"] == 7
    assert prov["input_sha256"]
    assert prov["requested_limit"] is None


def test_experiment_limit_is_recorded_and_deterministic(tmp_path: Path):
    inp = tmp_path / "many.jsonl"
    inp.write_text("".join(json.dumps(_episode(i)) + "\n" for i in range(5)), encoding="utf-8")
    out = tmp_path / "limited"
    run_experiment(inp, MockMultiImageAdapter(), out, {"stage": "gate"}, limit=2)

    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == ["x0", "x1"]
    assert prov["requested_limit"] == 2
    assert prov["n_examples"] == 2
