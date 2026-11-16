# GitHub repository audit — 2026-11-16

## Verified state before this documentation change

- Repo: `Rocky5502/-U-MIRAGE` (private); default branch `main`.
- Baseline main SHA: `04f90f8a933de7bf72755bf33ceef9d53f544c75`; 86 tracked files, including source, research scripts, a CI workflow, and six test modules.
- Branch `protected-rq123-v1`: 0 commits ahead of main, 11 behind, so its committed history is already included in main.
- Branch `agent/final-results-sync-2026-09-18`: diverged, 10 commits ahead and 10 behind. Open draft PR #5 is stale relative to main; review and cherry-pick unique changes rather than merging the branch blindly.
- The last inspected main GitHub Actions run (40887170015) reports failure in both Python 3.10 and 3.11 jobs with runner_id=0 and zero recorded job steps. This is **not** evidence that the test suite passed or failed after execution.
- The current main already includes `scripts/audit_rq3_output_contract.py` and `tests/test_research_entrypoints.py`, along with CI compilation of research entrypoints. Draft PR #5 has a second output audit script and a September status note not present in main; these have not been merged automatically.

## Findings and blockers

1. RQ3: all 1,920 frozen development rows reportedly completed, but every nonzero-budget policy scores zero. Inspect raw generated answers and parser/output contract before any action-efficacy claim. No raw RQ3 archive was available in this connector inspection; the run cannot be certified.
2. Human source validation was not completed. Never describe construction-intent labels as causal source ground truth.
3. The calibration and holdout partitions remain sealed. No outcome from development may be represented as protected holdout evidence.
4. Initial remote CI failed Ruff and skipped pytest, but this was repaired. The final documented run passed 12 pytest tests in each of Python 3.10 and 3.11, including compilation, Ruff and smoke checks. No GPU/real-VLM inference was executed in this session.
5. Keep historic experiment branches for provenance. Do not force-push or automatically merge old draft PRs.
6. `src/umirage/adapters/transformers_vlm.py` reports confidence as a mean of per-step maximum vocabulary probabilities; this relies on the greedy decoding behavior and warrants direct generated-token alignment regression tests if generation mode changes.

## Final verified engineering CI (after the above baseline audit)

On 2026-11-16, the repository was updated to correct the concrete lint failures revealed by GitHub Actions, using only formatting/import and routine lint fixes plus an older reviewed baseline formatting change where the target files were unchanged in main.

- **Verified passing run:** [GitHub Actions run 43878180192](https://github.com/Rocky5502/-U-MIRAGE/actions/runs/43878180192).
- **Python 3.10:** 12 pytest tests passed; Ruff checks passed; compileall, secret scan, and synthetic smoke test passed.
- **Python 3.11:** 12 pytest tests passed; Ruff checks passed; compileall, secret scan, and synthetic smoke test passed.
- These tests validate repository plumbing and deterministic unit behavior, **not** GPU evaluation results, raw RQ3 validity, scientific source semantics, or protected-held-out results.
- The 16 November figure/README documentation was committed and the four SVG paths were independently fetched from main. The original PNGs are provided separately for manual upload if exact pixels are wanted.
- The final-results draft PR #5 remains open and diverged. Some branch-only audit code is intentionally not merged because an overlapping read-only diagnostic already exists in main; any unique script must be reconciled and tested rather than copied indiscriminately.

## Changes made in this commit

- README gallery and four lightweight SVG redrawings of author-supplied figure concepts.
- Correct source-validation wording in the Figure 2 redraw.
- Figure provenance and current code-review notes.
- No experimental code, frozen thresholds, model results, original PNGs, CI workflow, or protected inputs modified.

## Local verification still required

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src experiments scripts tests
ruff check src tests scripts
pytest -q
python scripts/check_secrets.py
python scripts/make_demo_data.py && python scripts/run_smoke.py
```

Avoid describing this commit as a full scientific validation or as a complete main-branch synchronization of all historical experimental branches.
