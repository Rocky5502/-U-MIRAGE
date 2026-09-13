#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from umirage.adapters.mock import MockMultiImageAdapter
from umirage.adapters.qwen_vl import QwenVLAdapter
from umirage.adapters.transformers_vlm import TransformersVLMAdapter
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


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def norm(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def make_adapter(args):
    if args.adapter == "mock":
        return MockMultiImageAdapter()
    if not args.model_id:
        raise SystemExit("STOP: --model-id required for real adapter")
    cls = QwenVLAdapter if args.adapter == "qwen" else TransformersVLMAdapter
    return cls(
        args.model_id,
        revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
    )


def prompt_for(policy: str, predicted_source: str, question: str, call_index: int, rng):
    if policy == "umirage":
        action = source_action(predicted_source)
        prompt = source_action_prompt(predicted_source, question)
    elif policy == "verify":
        action = "verify_reasoning"
        prompt = generic_verify_prompt(question)
    elif policy == "requery":
        action = "generic_requery"
        prompt = generic_requery_prompt(question)
    elif policy == "random_source":
        source = rng.choice(list(SOURCES))
        action = source_action(source)
        prompt = source_action_prompt(source, question)
    elif policy == "no_extra_call":
        return "none", None
    else:
        raise ValueError(policy)
    if call_index > 0:
        prompt += "\n\nSecond and final allowed repair call: independently check again and return the final answer."
    return action, prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute protected RQ3 matched-budget repair policies")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter", choices=["mock", "qwen", "transformers"], default="mock")
    ap.add_argument("--model-id")
    ap.add_argument("--revision")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--budgets", default="0,1,2")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--seed", type=int, default=27)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x]
    policies = [x for x in args.policies.split(",") if x]
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise SystemExit(f"STOP: unknown policies {unknown}")

    episodes = {ep.id: ep for ep in load_jsonl(args.episodes)}
    score_rows = {str(r["id"]): r for r in read_jsonl(args.scores)}
    shared = sorted(set(episodes) & set(score_rows))
    if not shared:
        raise SystemExit("STOP: no shared episode IDs between episodes and frozen scores")
    if set(episodes) - set(score_rows):
        raise SystemExit("STOP: at least one episode lacks frozen source scores")

    adapter = make_adapter(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with out.open("w", encoding="utf-8") as f:
        for eid in shared:
            ep = episodes[eid]
            sr = score_rows[eid]
            predicted_source = scores_from_row(sr).argmax()
            t0 = time.perf_counter()
            base = adapter.answer(ep)
            base_latency = time.perf_counter() - t0
            base_correct = norm(base.answer) == norm(ep.answer)

            for policy in policies:
                for budget in budgets:
                    current_answer = base.answer
                    calls = []
                    executed_action = "none"
                    extra_tokens = 0
                    extra_latency = 0.0
                    if policy != "no_extra_call":
                        for call_index in range(budget):
                            action, prompt = prompt_for(policy, predicted_source, ep.question, call_index, rng)
                            executed_action = action
                            t1 = time.perf_counter()
                            repaired = adapter.answer(ep, prompt_override=prompt)
                            latency = time.perf_counter() - t1
                            extra_latency += latency
                            extra_tokens += int(repaired.metadata.get("input_tokens", 0))
                            extra_tokens += int(repaired.metadata.get("output_tokens", 0))
                            current_answer = repaired.answer
                            calls.append({
                                "call_index": call_index + 1,
                                "action": action,
                                "latency_s": latency,
                                "input_tokens": int(repaired.metadata.get("input_tokens", 0)),
                                "output_tokens": int(repaired.metadata.get("output_tokens", 0)),
                            })
                    row = {
                        "id": eid,
                        "group_id": str(sr["group_id"]),
                        "source_label": str(sr.get("source_label", ep.source_label)),
                        "predicted_source": predicted_source,
                        "policy": policy,
                        "budget": budget,
                        "executed_action": executed_action,
                        "correct_before": base_correct,
                        "correct_after": norm(current_answer) == norm(ep.answer),
                        "extra_calls": len(calls),
                        "extra_tokens": extra_tokens,
                        "extra_latency_s": extra_latency,
                        "base_latency_s": base_latency,
                        "calls": calls,
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    f.flush()
                    print(f"{eid} policy={policy} budget={budget} before={int(base_correct)} after={int(row['correct_after'])}", flush=True)


if __name__ == "__main__":
    main()
