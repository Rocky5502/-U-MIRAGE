from __future__ import annotations

import csv
import json

import pytest

from scripts.init_paper_results import CSV_SCHEMAS, initialize


def test_initialize_creates_header_only_paper_artifacts(tmp_path):
    initialize(tmp_path)

    for name, fields in CSV_SCHEMAS.items():
        with (tmp_path / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows == [fields]

    statistics = json.loads((tmp_path / "statistics.json").read_text(encoding="utf-8"))
    compute = json.loads((tmp_path / "compute.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "paper_results_manifest.json").read_text(encoding="utf-8"))

    assert statistics["comparisons"] == []
    assert compute["runs"] == []
    assert manifest["status"] == "empty_pre_result_template"


def test_initialize_refuses_to_overwrite_without_force(tmp_path):
    initialize(tmp_path)
    with pytest.raises(FileExistsError):
        initialize(tmp_path)
