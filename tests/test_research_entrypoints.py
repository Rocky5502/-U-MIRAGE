"""Regression coverage for research entrypoints and the RQ3 claim boundary.

No GPU, hosted account, or private dataset is required.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_research_entrypoints_have_valid_python_syntax():
    paths = sorted(
        p
        for folder in ("experiments", "scripts", "src")
        for p in (ROOT / folder).rglob("*.py")
    )
    assert paths, "No research Python sources discovered"
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_rq3_output_contract_flags_uniform_repair_collapse():
    path = ROOT / "scripts" / "audit_rq3_output_contract.py"
    spec = importlib.util.spec_from_file_location("rq3_output_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    policies = ("umirage", "verify", "requery", "random_source", "no_extra_call")
    rows = [
        {
            "id": "synthetic-development-only",
            "policy": policy,
            "budget": budget,
            "correct_after": bool(budget == 0),
            "final_answer": "yes" if budget == 0 else "",
            "extra_tokens": 0 if budget == 0 else 16,
        }
        for policy in policies
        for budget in (0, 1, 2)
    ]
    result = module.audit(rows)
    assert result["total_rows"] == 15
    assert result["full_15_cell_grid"] is True
    assert result["all_nonzero_budget_arms_zero"] is True
    assert result["efficacy_interpretation"] == "BLOCKED"
    assert result["frozen_correctness_labels_modified"] is False
