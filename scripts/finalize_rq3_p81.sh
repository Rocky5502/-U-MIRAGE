#!/usr/bin/env bash
set -euo pipefail

cd "${1:-$HOME/U-MIRAGE/repo}"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

OUTDIR="artifacts/protected/machine_only_v1"
RAW="$OUTDIR/rq3_qwen3_p81_primary.jsonl"
RES="$OUTDIR/rq3_qwen3_p81_results.json"
AUDIT="$OUTDIR/rq3_qwen3_p81_final_audit.json"
HASHES="$OUTDIR/rq3_qwen3_p81_final_SHA256SUMS.txt"
BUNDLE="U_MIRAGE_RQ3_P81_FINAL_RESULTS_20260917.tar.gz"

echo "============================================================"
echo " U-MIRAGE RQ3 — FINAL VALIDATION + EVALUATION"
echo " P81 DEVELOPMENT | MACHINE-ONLY"
echo "============================================================"

test -f "$RAW" || { echo "STOP: missing $RAW"; exit 2; }

ROWS="$(wc -l < "$RAW")"
echo "RAW_ROWS=$ROWS"

if [ "$ROWS" -ne 1920 ]; then
  echo "STOP: expected exactly 1920 rows"
  exit 3
fi

python - "$RAW" "$AUDIT" <<'PY'
import hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

raw = Path(sys.argv[1])
audit_path = Path(sys.argv[2])

rows = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
expected_policies = {"umirage","verify","requery","random_source","no_extra_call"}
expected_budgets = {0,1,2}
expected_grid = {(p,b) for p in expected_policies for b in expected_budgets}

by_id = defaultdict(list)
key_counts = Counter()
for r in rows:
    eid = str(r["id"])
    policy = str(r["policy"])
    budget = int(r["budget"])
    by_id[eid].append(r)
    key_counts[(eid,policy,budget)] += 1

ids = sorted(by_id)
if len(ids) != 128:
    raise SystemExit(f"STOP: expected 128 unique IDs, got {len(ids)}")
if any(len(rr) != 15 for rr in by_id.values()):
    bad = {eid:len(rr) for eid,rr in by_id.items() if len(rr) != 15}
    raise SystemExit(f"STOP: IDs without exactly 15 rows: {bad}")
dups = [k for k,n in key_counts.items() if n != 1]
if dups:
    raise SystemExit(f"STOP: duplicate policy-budget cells: {dups[:10]}")
bad_grid = []
for eid, rr in by_id.items():
    grid = {(str(r["policy"]),int(r["budget"])) for r in rr}
    if grid != expected_grid:
        bad_grid.append(eid)
if bad_grid:
    raise SystemExit(f"STOP: incomplete grids: {bad_grid[:10]}")

policies = sorted({str(r["policy"]) for r in rows})
budgets = sorted({int(r["budget"]) for r in rows})
if set(policies) != expected_policies:
    raise SystemExit(f"STOP: policy mismatch: {policies}")
if set(budgets) != expected_budgets:
    raise SystemExit(f"STOP: budget mismatch: {budgets}")

h = hashlib.sha256(raw.read_bytes()).hexdigest()
audit = {
    "raw_rows": len(rows),
    "unique_ids": len(ids),
    "rows_per_id": 15,
    "policies": policies,
    "budgets": budgets,
    "full_grid": True,
    "duplicate_policy_budget_cells": 0,
    "raw_sha256": h,
    "evidence_level": "P81 development",
    "machine_only": True,
    "cal60_opened": False,
    "holdout360_opened": False,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print("IDS=128")
print("ROWS_PER_ID=15")
print("POLICIES=", policies)
print("BUDGETS=", budgets)
print("RQ3_RAW_GRID=PASS")
print("RAW_SHA256=", h)
PY

echo
echo "=== CANONICAL EVALUATOR ==="

python experiments/protected/evaluate_rq3.py   --input "$RAW"   --output "$RES"   --bootstrap 5000   --seed 27   > "$OUTDIR/rq3_qwen3_p81_evaluator_stdout.json"

test -s "$RES" || { echo "STOP: evaluator result missing/empty"; exit 4; }

python - "$RES" "$AUDIT" <<'PY'
import json, math, sys
from pathlib import Path

res_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
x = json.loads(res_path.read_text(encoding="utf-8"))
summary = x.get("summary") or {}
paired = x.get("paired_primary") or {}

required_summary = [
    "umirage|0","umirage|1","umirage|2",
    "verify|1","verify|2","requery|1","requery|2",
    "random_source|1","random_source|2","no_extra_call|0"
]
required_paired = [
    "umirage_vs_verify_b1","umirage_vs_requery_b1",
    "umirage_vs_verify_b2","umirage_vs_requery_b2"
]

ms = [k for k in required_summary if k not in summary]
mp = [k for k in required_paired if k not in paired]
if ms: raise SystemExit(f"STOP: missing summary keys: {ms}")
if mp: raise SystemExit(f"STOP: missing paired keys: {mp}")

for k in required_summary:
    if int(summary[k].get("n",-1)) != 128:
        raise SystemExit(f"STOP: {k} n={summary[k].get('n')} expected 128")
for k in required_paired:
    if int(paired[k].get("n_pairs",-1)) != 128:
        raise SystemExit(f"STOP: {k} n_pairs={paired[k].get('n_pairs')} expected 128")

if int(x.get("seed",-1)) != 27:
    raise SystemExit(f"STOP: seed={x.get('seed')}")
if int(x.get("bootstrap_resamples",-1)) != 5000:
    raise SystemExit(f"STOP: bootstrap_resamples={x.get('bootstrap_resamples')}")

print()
print("=== RQ3 PRIMARY RESULTS ===")
for k in required_summary:
    y = summary[k]
    rr = y.get("recovery_rate")
    rr_text = "NA" if rr is None or (isinstance(rr,float) and math.isnan(rr)) else f"{rr:.6f}"
    print(
        f"{k:22s} "
        f"base={y['base_accuracy']:.6f} "
        f"post={y['post_accuracy']:.6f} "
        f"gain={y['accuracy_gain']:+.6f} "
        f"recovered={y['recovered_errors']}/{y['eligible_errors']} "
        f"recovery={rr_text} "
        f"tokens={y['extra_tokens']:.0f}"
    )

print()
print("=== PAIRED PRIMARY ===")
for k in required_paired:
    y = paired[k]
    lo, hi = y["ci95"]
    print(
        f"{k:28s} "
        f"n_pairs={y['n_pairs']} "
        f"delta={y['accuracy_difference']:+.6f} "
        f"CI95=[{lo:+.6f},{hi:+.6f}]"
    )

degenerate_keys = [
    "umirage|1", "umirage|2",
    "verify|1", "verify|2",
    "requery|1", "requery|2",
]
degenerate = all(float(summary[k]["post_accuracy"]) == 0.0 for k in degenerate_keys)

audit = json.loads(audit_path.read_text(encoding="utf-8"))
audit.update({
    "evaluation_complete": True,
    "result_mode": x.get("mode"),
    "seed": x.get("seed"),
    "bootstrap_resamples": x.get("bootstrap_resamples"),
    "required_summary_keys_present": True,
    "paired_primary_n128": True,
    "source_repair_matrix_claim_allowed": False,
    "random_source_role": "auxiliary only",
    "degenerate_repair_output_collapse": degenerate,
    "efficacy_claim_valid": not degenerate,
    "note": (
        "P81 development evidence only; source×repair causal claims are not made. "
        "If every nonzero-budget primary repair cell collapses to zero accuracy, "
        "treat the run as an output-validity failure rather than an efficacy result."
    ),
})
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True)+"\n", encoding="utf-8")

print()
print("RQ3_RESULTS_NONEMPTY=PASS")
print("RQ3_PAIRED_N128=PASS")
print("RQ3_EVALUATION_FILE=PASS")
print(f"RQ3_DEGENERATE_REPAIR_OUTPUT={'TRUE' if degenerate else 'FALSE'}")
print(f"RQ3_EFFICACY_CLAIM_VALID={'FALSE' if degenerate else 'TRUE'}")
PY

echo
echo "=== FINAL HASHES ==="
: > "$HASHES"
for f in   "$RAW"   "$RES"   "$AUDIT"   "$OUTDIR/rq3_qwen3_p81_evaluator_stdout.json"   experiments/protected/run_rq3_primary_cached.py   experiments/protected/evaluate_rq3.py   src/umirage/adapters/qwen_vl.py
do
  if [ -f "$f" ]; then
    sha256sum "$f" | tee -a "$HASHES"
  fi
done

echo
echo "=== PACKAGE FINAL RQ3 EVIDENCE ==="
rm -f "$BUNDLE"
tar -czf "$BUNDLE"   "$RAW"   "$RES"   "$AUDIT"   "$HASHES"   "$OUTDIR/rq3_qwen3_p81_evaluator_stdout.json"   experiments/protected/run_rq3_primary_cached.py   experiments/protected/evaluate_rq3.py   src/umirage/adapters/qwen_vl.py

sha256sum "$BUNDLE"

echo
echo "RQ3_FINALIZATION_COMPLETE=TRUE"
echo "RQ3_EVIDENCE_LEVEL=P81_DEVELOPMENT"
echo "SOURCE_REPAIR_MATRIX_CLAIM=FALSE"
echo "RANDOM_SOURCE_ROLE=AUXILIARY"
echo "CAL60_OPENED=FALSE"
echo "HOLDOUT360_OPENED=FALSE"