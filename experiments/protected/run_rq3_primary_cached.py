#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from umirage.adapters.qwen_vl import QwenVLAdapter
from umirage.io import load_jsonl
from umirage.protected import (
    SOURCES,
    generic_requery_prompt,
    generic_verify_prompt,
    scores_from_row,
    source_action,
    source_action_prompt,
)

POLICIES = ("umirage", "verify", "requery", "random_source", "no_extra_call")
BUDGETS = (0, 1, 2)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def norm(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def stable_random_source(seed: int, episode_id: str, call_index: int) -> str:
    payload = f"{seed}|{episode_id}|{call_index}".encode("utf-8")
    h = hashlib.sha256(payload).digest()
    return SOURCES[int.from_bytes(h[:8], "big") % len(SOURCES)]


def with_call_suffix(prompt: str, call_index: int) -> str:
    if call_index == 1:
        return prompt
    return (
        prompt
        + "\n\nThis is the second and final allowed repair call. "
          "Independently check the evidence again and return the final answer."
    )


def prompt_for(policy: str, predicted_source: str, question: str,
               call_index: int, seed: int, episode_id: str) -> tuple[str, str]:
    if policy == "umirage":
        src = predicted_source
        return source_action(src), with_call_suffix(
            source_action_prompt(src, question), call_index
        )
    if policy == "verify":
        return "generic_verify", with_call_suffix(
            generic_verify_prompt(question), call_index
        )
    if policy == "requery":
        return "generic_requery", with_call_suffix(
            generic_requery_prompt(question), call_index
        )
    if policy == "random_source":
        src = stable_random_source(seed, episode_id, call_index)
        return source_action(src), with_call_suffix(
            source_action_prompt(src, question), call_index
        )
    raise ValueError(policy)


def group_id_for(ep, score_row: dict) -> str:
    md = ep.metadata or {}
    pmd = score_row.get("episode_metadata") or {}
    candidates = (
        score_row.get("group_id"),
        pmd.get("mirageu_group_id"),
        pmd.get("group_id"),
        md.get("mirageu_group_id"),
        md.get("group_id"),
        ep.id,
    )
    for x in candidates:
        if x is not None and str(x).strip():
            return str(x)
    raise RuntimeError(f"no group id for {ep.id}")


def complete_ids(rows: list[dict]) -> set[str]:
    expected = {(p, b) for p in POLICIES for b in BUDGETS}
    by_id: dict[str, set[tuple[str, int]]] = {}
    for r in rows:
        eid = str(r.get("id"))
        key = (str(r.get("policy")), int(r.get("budget", -1)))
        by_id.setdefault(eid, set()).add(key)
    return {eid for eid, combos in by_id.items() if combos == expected}


def rewrite_only_complete(path: Path, rows: list[dict], keep_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if str(r.get("id")) in keep_ids:
                f.write(json.dumps(r, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="U-MIRAGE RQ3 primary matched-budget runner with frozen-base reuse and per-episode prompt caching."
    )
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    ep_path = Path(args.episodes)
    sc_path = Path(args.scores)
    out_path = Path(args.output)

    forbidden = ("cal60", "calibration-60", "holdout360", "holdout-360")
    for p in (ep_path, sc_path, out_path):
        low = str(p).lower()
        if any(x in low for x in forbidden):
            raise SystemExit(f"STOP: protected partition path detected: {p}")

    episodes = {ep.id: ep for ep in load_jsonl(ep_path)}
    scores = {str(r["id"]): r for r in read_jsonl(sc_path)}

    shared = sorted(set(episodes) & set(scores))
    if not shared:
        raise SystemExit("STOP: no shared episode IDs")
    if set(episodes) != set(scores):
        raise SystemExit(
            f"STOP: exact ID set required; episodes={len(episodes)} scores={len(scores)} shared={len(shared)}"
        )

    splits = {}
    for eid in shared:
        ep = episodes[eid]
        split = str((ep.metadata or {}).get("mirageu_split") or (ep.metadata or {}).get("split") or "UNKNOWN")
        splits[split] = splits.get(split, 0) + 1
        sl = split.lower()
        if "calibration" in sl or "holdout" in sl or sl == "test":
            raise SystemExit(f"STOP: non-development split detected: {split}")

    old_rows = read_jsonl(out_path) if out_path.exists() else []
    done = complete_ids(old_rows)
    rewrite_only_complete(out_path, old_rows, done)

    amendment = out_path.with_suffix(".protocol.json")
    amendment.write_text(
        json.dumps(
            {
                "schema": "umirage.rq3.machine_only_primary.v1",
                "recorded_before_new_rq3_outcomes": True,
                "episodes": str(ep_path),
                "scores": str(sc_path),
                "splits": splits,
                "seed": args.seed,
                "budgets": list(BUDGETS),
                "primary_policies": list(POLICIES),
                "base_answer_source": "frozen_prediction_file",
                "base_model_requery": False,
                "repair_model": str(args.model_id),
                "max_new_tokens_per_extra_call": args.max_new_tokens,
                "random_source_rule": "sha256(seed|episode_id|call_index) mod 3",
                "source_repair_matrix": "not_claimed_on_P81_machine_only_run_because_no_human_validated_source_ground_truth",
                "primary_claim": "matched-budget policy efficacy only",
                "cal60_opened": False,
                "holdout360_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    remaining = [eid for eid in shared if eid not in done]
    print("=" * 72, flush=True)
    print("U-MIRAGE RQ3 PRIMARY — CACHED MACHINE-ONLY RUN", flush=True)
    print(f"N_TOTAL={len(shared)} N_COMPLETE={len(done)} N_REMAINING={len(remaining)}", flush=True)
    print(f"SPLITS={splits}", flush=True)
    print("POLICIES=" + ",".join(POLICIES), flush=True)
    print("BUDGETS=0,1,2", flush=True)
    print("BASE_PREDICTIONS=FROZEN_REUSE", flush=True)
    print("CAL60_OPENED=FALSE", flush=True)
    print("HOLDOUT360_OPENED=FALSE", flush=True)
    print("=" * 72, flush=True)

    if not remaining:
        print("RQ3_PRIMARY_ALREADY_COMPLETE=TRUE", flush=True)
        return

    adapter = QwenVLAdapter(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
    )

    with out_path.open("a", encoding="utf-8") as f:
        for idx, eid in enumerate(remaining, 1):
            ep = episodes[eid]
            sr = scores[eid]
            predicted_source = scores_from_row(sr).argmax()
            gid = group_id_for(ep, sr)

            if "correct" not in sr or "prediction" not in sr:
                raise SystemExit(f"STOP: frozen score row lacks correct/prediction for {eid}")

            base_answer = str(sr["prediction"])
            base_correct = bool(sr["correct"])

            cache: dict[str, dict] = {}

            def run_prompt(prompt: str) -> dict:
                if prompt in cache:
                    return cache[prompt]
                t0 = time.perf_counter()
                out = adapter.answer(ep, prompt_override=prompt)
                latency = time.perf_counter() - t0
                item = {
                    "answer": out.answer,
                    "correct": norm(out.answer) == norm(ep.answer),
                    "latency_s": latency,
                    "input_tokens": int(out.metadata.get("input_tokens", 0)),
                    "output_tokens": int(out.metadata.get("output_tokens", 0)),
                }
                cache[prompt] = item
                return item

            episode_rows = []

            for policy in POLICIES:
                if policy == "no_extra_call":
                    for budget in BUDGETS:
                        episode_rows.append(
                            {
                                "id": eid,
                                "group_id": gid,
                                "source_label": ep.source_label,
                                "predicted_source": predicted_source,
                                "policy": policy,
                                "matrix_arm": False,
                                "budget": budget,
                                "executed_action": "none",
                                "correct_before": base_correct,
                                "correct_after": base_correct,
                                "base_answer": base_answer,
                                "final_answer": base_answer,
                                "extra_calls": 0,
                                "extra_tokens": 0,
                                "extra_latency_s": 0.0,
                                "calls": [],
                            }
                        )
                    continue

                call_results = []
                for call_index in (1, 2):
                    action, prompt = prompt_for(
                        policy,
                        predicted_source,
                        ep.question,
                        call_index,
                        args.seed,
                        eid,
                    )
                    rr = run_prompt(prompt)
                    call_results.append(
                        {
                            "call_index": call_index,
                            "action": action,
                            "answer": rr["answer"],
                            "correct": rr["correct"],
                            "latency_s": rr["latency_s"],
                            "input_tokens": rr["input_tokens"],
                            "output_tokens": rr["output_tokens"],
                        }
                    )

                episode_rows.append(
                    {
                        "id": eid,
                        "group_id": gid,
                        "source_label": ep.source_label,
                        "predicted_source": predicted_source,
                        "policy": policy,
                        "matrix_arm": False,
                        "budget": 0,
                        "executed_action": "none",
                        "correct_before": base_correct,
                        "correct_after": base_correct,
                        "base_answer": base_answer,
                        "final_answer": base_answer,
                        "extra_calls": 0,
                        "extra_tokens": 0,
                        "extra_latency_s": 0.0,
                        "calls": [],
                    }
                )

                for budget in (1, 2):
                    used = call_results[:budget]
                    final = used[-1]
                    episode_rows.append(
                        {
                            "id": eid,
                            "group_id": gid,
                            "source_label": ep.source_label,
                            "predicted_source": predicted_source,
                            "policy": policy,
                            "matrix_arm": False,
                            "budget": budget,
                            "executed_action": final["action"],
                            "correct_before": base_correct,
                            "correct_after": bool(final["correct"]),
                            "base_answer": base_answer,
                            "final_answer": final["answer"],
                            "extra_calls": budget,
                            "extra_tokens": sum(x["input_tokens"] + x["output_tokens"] for x in used),
                            "extra_latency_s": sum(x["latency_s"] for x in used),
                            "calls": used,
                        }
                    )

            # all 15 rows are committed together: resume-safe at episode granularity
            for row in episode_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

            print(
                f"[{idx}/{len(remaining)}] id={eid} "
                f"pred_source={predicted_source} unique_model_calls={len(cache)}",
                flush=True,
            )

    print("RQ3_PRIMARY_RAW_COMPLETE=TRUE", flush=True)
    print("CAL60_OPENED=FALSE", flush=True)
    print("HOLDOUT360_OPENED=FALSE", flush=True)


if __name__ == "__main__":
    main()
