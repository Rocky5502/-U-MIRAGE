#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from umirage.protected import (
    generic_requery_prompt,
    generic_verify_prompt,
    scores_from_row,
    source_action,
    source_action_prompt,
)

ROOT = Path(".")
EPISODES = Path("data/materialized/mirageu_p81_natural_stage_bank/episodes.jsonl")
SCORES = Path(
    os.environ.get(
        "UMIRAGE_RQ4C_SCORES",
        "artifacts/local/qwen3-vl-8b-instruct/"
        "p81_natural_stage128/predictions.jsonl",
    )
)
OUTDIR = Path("artifacts/rq4_blackbox/executor_transfer_v1")

EXPECTED_EPISODE_SHA256 = (
    "dba13a003e43521b6ed74e0411e90410929516b3f9d1494b81194b6f9a454443"
)

MODELS = ("gpt-6-astra", "gemini-3.8-flash")
CONDITIONS = ("base", "source_conditioned", "verify", "requery")
REPAIR_CONDITIONS = ("source_conditioned", "verify", "requery")

# These eight P81 episodes were already consumed by the earlier black-box
# uncertainty-interface pilots and are excluded from the fresh executor cohort.
PRIOR_BLACKBOX_IDS = tuple(f"mirageu-p81-n{i:03d}" for i in range(8))

N_FRESH = 32
SEED = 27
BOOTSTRAPS = 5000
MAX_TOKENS = 256
TIMEOUT_S = 180
SELECTION_SALT = "umirage-rq4c-executor-transfer-v1"

BASE_URL = os.environ.get(
    "ZZZ_BASE_URL",
    "https://api.zhizengzeng.com/v1",
).rstrip("/")
API_KEY = os.environ.get("ZZZ_API_KEY", "")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    b = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"^\s*(final\s+answer|answer)\s*[:\-]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" \t\r\n.,;:!?\"'`()[]{}")
    return s


def parse_answer(text: str, gold: str) -> tuple[str | None, bool]:
    """
    Deterministic machine-only parser frozen before new RQ4c outcomes.

    It prefers strict normalized equality. For yes/no and numeric gold answers,
    it permits an unambiguous leading yes/no token or a single numeric token.
    No LLM-as-judge is used.
    """
    raw = str(text or "").strip()
    if not raw:
        return None, False

    # Use the first non-empty line because prompts explicitly request only a
    # final answer; this avoids post-hoc free-form interpretation.
    first = next((x.strip() for x in raw.splitlines() if x.strip()), raw)
    p = norm(first)
    g = norm(gold)

    if p == g:
        return p, True

    if g in {"yes", "no"}:
        m = re.match(r"^(yes|no)\b", p)
        if m:
            val = m.group(1)
            return val, val == g
        return p, False

    # Integer / simple decimal answers.
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", g):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", p)
        if len(nums) == 1:
            val = norm(nums[0])
            try:
                return val, float(val) == float(g)
            except ValueError:
                pass
        return p, False

    return p, False


def image_data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image missing: {p}")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extract_text(body: dict) -> str:
    try:
        content = body["choices"][0]["message"].get("content", "")
    except Exception:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(x for x in parts if x).strip()

    return str(content or "").strip()


def usage(body: dict) -> dict:
    u = body.get("usage") or {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens") or 0),
        "completion_tokens": int(u.get("completion_tokens") or 0),
        "total_tokens": int(u.get("total_tokens") or 0),
        "reasoning_tokens": int(
            ((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            or 0
        ),
    }


def request_key(model: str, prompt: str, images: list[str]) -> str:
    obj = {
        "model": model,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "image_sha256": [sha256_file(Path(x)) for x in images],
    }
    return canonical_sha256(obj)


def api_call(model: str, prompt: str, images: list[str]) -> dict:
    cache_dir = OUTDIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = request_key(model, prompt, images)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        obj["cache_hit"] = True
        return obj

    content = [
        *[
            {"type": "image_url", "image_url": {"url": image_data_uri(p)}}
            for p in images
        ],
        {"type": "text", "text": prompt},
    ]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt, delay in enumerate((0, 2, 5), 1):
        if delay:
            time.sleep(delay)
        t0 = time.perf_counter()
        try:
            r = requests.post(
                BASE_URL + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT_S,
            )
            latency = time.perf_counter() - t0
            try:
                body = r.json()
            except Exception:
                body = {"raw_text": r.text}

            obj = {
                "request_key": key,
                "model_requested": model,
                "reported_model": body.get("model") if isinstance(body, dict) else None,
                "http_status": int(r.status_code),
                "latency_s": latency,
                "text": extract_text(body) if isinstance(body, dict) else "",
                "usage": usage(body) if isinstance(body, dict) else {},
                "response": body,
                "cache_hit": False,
                "attempt": attempt,
            }

            # Do not retry a model-level blank/parse failure after HTTP 200.
            # Retries are transport/rate-limit/server-error only.
            if r.status_code == 200:
                atomic_json(cache_path, obj)
                return obj

            if r.status_code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                atomic_json(cache_path, obj)
                return obj

            last_error = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            latency = time.perf_counter() - t0
            last_error = repr(exc)
            obj = {
                "request_key": key,
                "model_requested": model,
                "reported_model": None,
                "http_status": None,
                "latency_s": latency,
                "text": "",
                "usage": {},
                "response": None,
                "cache_hit": False,
                "attempt": attempt,
                "transport_error": repr(exc),
            }

    obj["final_transport_error"] = last_error
    atomic_json(cache_path, obj)
    return obj


def base_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Answer using only the provided images. "
        "Return only the final answer, with no explanation."
    )


def condition_prompt(condition: str, predicted_source: str, question: str) -> str:
    if condition == "base":
        return base_prompt(question)
    if condition == "source_conditioned":
        p = source_action_prompt(predicted_source, question)
    elif condition == "verify":
        p = generic_verify_prompt(question)
    elif condition == "requery":
        p = generic_requery_prompt(question)
    else:
        raise ValueError(condition)
    return p + "\nReturn only the final answer, with no explanation."


def deterministic_order(items: tuple[str, ...], key: str) -> list[str]:
    return sorted(
        items,
        key=lambda x: hashlib.sha256(
            f"{SEED}|{key}|{x}".encode("utf-8")
        ).hexdigest(),
    )


def build_manifest() -> tuple[list[dict], dict]:
    if sha256_file(EPISODES) != EXPECTED_EPISODE_SHA256:
        raise SystemExit("STOP: canonical P81 episode SHA256 mismatch")

    episodes = read_jsonl(EPISODES)
    scores = read_jsonl(SCORES)
    if len(episodes) != 128 or len(scores) != 128:
        raise SystemExit(
            f"STOP: expected 128 episode/score rows, got {len(episodes)}/{len(scores)}"
        )

    ep = {str(x["id"]): x for x in episodes}
    sc = {str(x["id"]): x for x in scores}
    if set(ep) != set(sc):
        raise SystemExit("STOP: episode/score ID sets differ")

    candidates = sorted(set(ep) - set(PRIOR_BLACKBOX_IDS))
    if len(candidates) != 120:
        raise SystemExit(f"STOP: expected 120 fresh candidates, got {len(candidates)}")

    selected = sorted(
        candidates,
        key=lambda eid: hashlib.sha256(
            f"{SELECTION_SALT}|{SEED}|{eid}".encode("utf-8")
        ).hexdigest(),
    )[:N_FRESH]

    rows = []
    for eid in selected:
        row = ep[eid]
        sr = sc[eid]
        md = row.get("metadata") or {}
        split = str(md.get("mirageu_split") or md.get("split") or "UNKNOWN")
        sl = split.lower()
        if "calibration" in sl or "holdout" in sl or sl == "test":
            raise SystemExit(f"STOP: non-development split in manifest: {eid} {split}")

        predicted_source = scores_from_row(sr).argmax()
        rows.append(
            {
                "id": eid,
                "split": split,
                "predicted_source": predicted_source,
                "selection_rank_hash": hashlib.sha256(
                    f"{SELECTION_SALT}|{SEED}|{eid}".encode("utf-8")
                ).hexdigest(),
            }
        )

    protocol = {
        "schema": "umirage.rq4c.hosted_executor_transfer.v1",
        "status": "exploratory_development_protocol_frozen_before_new_outcomes",
        "scientific_question": (
            "Given a frozen upstream Qwen3 source diagnosis, do source-conditioned "
            "repair prompts transfer as executors to two hosted black-box multimodal models "
            "better than matched-budget generic verify or generic requery?"
        ),
        "important_boundary": (
            "This is NOT hosted-model uncertainty-estimator transfer. Earlier common-observable "
            "uncertainty gates remain failed and closed."
        ),
        "models": list(MODELS),
        "cohort": {
            "source": str(EPISODES),
            "episode_sha256": EXPECTED_EPISODE_SHA256,
            "score_source": str(SCORES),
            "score_sha256": sha256_file(SCORES),
            "n": N_FRESH,
            "selection": (
                "SHA256 ordering of the 120 P81 development IDs not used in the earlier "
                "black-box pilot; salt and seed fixed below."
            ),
            "selection_salt": SELECTION_SALT,
            "seed": SEED,
            "excluded_prior_blackbox_ids": list(PRIOR_BLACKBOX_IDS),
        },
        "routing_signal": (
            "argmax(visual,integration,reasoning) from the already-frozen Qwen3 P81 score file"
        ),
        "conditions": {
            "base": "one direct hosted-model answer",
            "source_conditioned": "one hosted-model call using frozen source-specific repair prompt",
            "verify": "one hosted-model call using generic verification prompt",
            "requery": "one hosted-model call using generic from-scratch requery prompt",
        },
        "matched_budget": "exactly one hosted-model call per condition",
        "call_order": (
            "base first; the three repair conditions and model order are deterministically "
            "SHA256-ordered per episode to reduce fixed temporal ordering effects"
        ),
        "max_tokens": MAX_TOKENS,
        "temperature": "omitted",
        "answer_scoring": (
            "deterministic machine-only parser; strict normalized equality, with only "
            "unambiguous yes/no and single-number fallbacks; no LLM judge"
        ),
        "primary_estimands": [
            "post_accuracy(source_conditioned) - post_accuracy(verify)",
            "post_accuracy(source_conditioned) - post_accuracy(requery)",
        ],
        "secondary": [
            "base accuracy",
            "error recovery relative to base",
            "regression relative to base",
            "parse rate",
            "tokens",
            "latency",
        ],
        "uncertainty_intervals": {
            "method": "paired episode bootstrap",
            "resamples": BOOTSTRAPS,
            "seed": SEED,
        },
        "evidence_level": "exploratory development only",
        "cal60_used": False,
        "holdout360_used": False,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return rows, protocol


def freeze() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTDIR / "manifest.jsonl"
    protocol_path = OUTDIR / "PROTOCOL_FROZEN.json"

    rows, protocol = build_manifest()

    # Never silently replace an existing frozen cohort/protocol.
    if manifest_path.exists():
        old = read_jsonl(manifest_path)
        if old != rows:
            raise SystemExit("STOP: existing manifest differs from deterministic freeze")
    else:
        write_jsonl(manifest_path, rows)

    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        # created_at is allowed to differ only if this is the first write. Once written,
        # exact content must remain frozen.
        compare_new = dict(protocol)
        compare_new["created_at_local"] = old.get("created_at_local")
        if old != compare_new:
            raise SystemExit("STOP: existing frozen protocol differs")
    else:
        atomic_json(protocol_path, protocol)

    freeze_txt = OUTDIR / "FREEZE_SHA256.txt"
    freeze_txt.write_text(
        "\n".join(
            [
                f"{sha256_file(EPISODES)}  {EPISODES}",
                f"{sha256_file(SCORES)}  {SCORES}",
                f"{sha256_file(manifest_path)}  {manifest_path}",
                f"{sha256_file(protocol_path)}  {protocol_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("U-MIRAGE RQ4c — HOSTED EXECUTOR TRANSFER FREEZE")
    print("=" * 72)
    print("MANIFEST_N =", len(rows))
    print("MODELS =", ",".join(MODELS))
    print("EXCLUDED_PRIOR_BLACKBOX =", len(PRIOR_BLACKBOX_IDS))
    print("CAL60_USED=FALSE")
    print("HOLDOUT360_USED=FALSE")
    print("MANIFEST_SHA256 =", sha256_file(manifest_path))
    print("PROTOCOL_SHA256 =", sha256_file(protocol_path))
    print("RQ4C_PROTOCOL_FREEZE=PASS")


def rows_index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    out = {}
    for r in rows:
        key = (str(r["id"]), str(r["model"]), str(r["condition"]))
        out[key] = r
    return out


def execute() -> None:
    if not API_KEY:
        raise SystemExit("STOP: ZZZ_API_KEY is not set")

    manifest_path = OUTDIR / "manifest.jsonl"
    protocol_path = OUTDIR / "PROTOCOL_FROZEN.json"
    if not manifest_path.exists() or not protocol_path.exists():
        raise SystemExit("STOP: run --mode freeze first")

    manifest = read_jsonl(manifest_path)
    if len(manifest) != N_FRESH:
        raise SystemExit("STOP: frozen manifest does not contain 32 rows")

    episodes = {str(x["id"]): x for x in read_jsonl(EPISODES)}
    scores = {str(x["id"]): x for x in read_jsonl(SCORES)}

    raw_path = OUTDIR / "raw_results.jsonl"
    existing = read_jsonl(raw_path)
    done = rows_index(existing)

    print("=" * 72, flush=True)
    print("U-MIRAGE RQ4c — HOSTED BLACK-BOX EXECUTOR TRANSFER", flush=True)
    print(f"N={len(manifest)} MODELS={len(MODELS)} CONDITIONS={len(CONDITIONS)}", flush=True)
    print("EXPECTED_CALL_CELLS=256", flush=True)
    print(f"ALREADY_COMPLETE={len(done)}", flush=True)
    print("CAL60_USED=FALSE", flush=True)
    print("HOLDOUT360_USED=FALSE", flush=True)
    print("=" * 72, flush=True)

    for epi, m in enumerate(manifest, 1):
        eid = str(m["id"])
        ep = episodes[eid]
        sr = scores[eid]
        predicted_source = scores_from_row(sr).argmax()
        if predicted_source != m["predicted_source"]:
            raise SystemExit(f"STOP: source diagnosis drift for {eid}")

        question = str(ep["question"])
        gold = str(ep["answer"])
        images = [str(x) for x in ep["images"]]

        model_order = deterministic_order(MODELS, f"model-order|{eid}")

        for model in model_order:
            condition_order = ["base"] + deterministic_order(
                REPAIR_CONDITIONS, f"condition-order|{model}|{eid}"
            )
            for condition in condition_order:
                key = (eid, model, condition)
                if key in done:
                    continue

                prompt = condition_prompt(condition, predicted_source, question)
                t0 = time.perf_counter()
                api = api_call(model, prompt, images)
                wall = time.perf_counter() - t0
                parsed, correct = parse_answer(api.get("text", ""), gold)

                row = {
                    "id": eid,
                    "model": model,
                    "condition": condition,
                    "predicted_source": predicted_source,
                    "executed_action": (
                        source_action(predicted_source)
                        if condition == "source_conditioned"
                        else condition
                    ),
                    "gold": gold,
                    "raw_answer": api.get("text", ""),
                    "parsed_answer": parsed,
                    "parse_ok": parsed is not None,
                    "correct": bool(correct),
                    "http_status": api.get("http_status"),
                    "reported_model": api.get("reported_model"),
                    "latency_s": float(api.get("latency_s") or 0.0),
                    "wall_s": wall,
                    "usage": api.get("usage") or {},
                    "cache_hit": bool(api.get("cache_hit")),
                    "request_key": api.get("request_key"),
                    "evidence_level": "exploratory_development",
                    "cal60_used": False,
                    "holdout360_used": False,
                }
                existing.append(row)
                write_jsonl(raw_path, existing)
                done[key] = row

        n_done = sum(1 for k in done if k[0] == eid)
        print(
            f"[{epi}/{len(manifest)}] id={eid} cells={n_done}/8 "
            f"source={predicted_source}",
            flush=True,
        )

    expected = len(manifest) * len(MODELS) * len(CONDITIONS)
    if len(done) != expected:
        raise SystemExit(f"STOP: expected {expected} unique cells, got {len(done)}")

    print("RQ4C_RAW_COMPLETE=TRUE", flush=True)
    print(f"RAW_ROWS={len(done)}", flush=True)


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def paired_bootstrap(a: list[int], b: list[int], seed: int) -> dict:
    if len(a) != len(b) or not a:
        raise ValueError("paired vectors must have equal nonzero length")
    n = len(a)
    point = statistics.fmean(a) - statistics.fmean(b)
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOTSTRAPS):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(
            statistics.fmean(a[i] for i in idx)
            - statistics.fmean(b[i] for i in idx)
        )
    return {
        "n_pairs": n,
        "point": point,
        "ci95": [percentile(vals, 0.025), percentile(vals, 0.975)],
        "resamples": BOOTSTRAPS,
        "seed": seed,
    }


def evaluate() -> None:
    manifest = read_jsonl(OUTDIR / "manifest.jsonl")
    raw = read_jsonl(OUTDIR / "raw_results.jsonl")
    if not manifest or not raw:
        raise SystemExit("STOP: manifest/raw results missing")

    expected = len(manifest) * len(MODELS) * len(CONDITIONS)
    idx = rows_index(raw)
    if len(idx) != expected:
        raise SystemExit(f"STOP: incomplete run: expected {expected}, got {len(idx)}")

    ids = [str(x["id"]) for x in manifest]
    result = {
        "schema": "umirage.rq4c.hosted_executor_transfer.results.v1",
        "evidence_level": "exploratory development only",
        "n": len(ids),
        "models": {},
        "cal60_used": False,
        "holdout360_used": False,
    }

    for model_i, model in enumerate(MODELS):
        mr = {"conditions": {}, "paired_primary": {}}
        base_vec = [int(idx[(eid, model, "base")]["correct"]) for eid in ids]

        for condition in CONDITIONS:
            rows = [idx[(eid, model, condition)] for eid in ids]
            corr = [int(r["correct"]) for r in rows]
            parse = [int(bool(r["parse_ok"])) for r in rows]
            prompt_tokens = [
                int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows
            ]
            completion_tokens = [
                int((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows
            ]
            lat = [float(r.get("latency_s") or 0.0) for r in rows]

            recovered = sum(
                1 for b, c in zip(base_vec, corr) if b == 0 and c == 1
            ) if condition != "base" else 0
            regressions = sum(
                1 for b, c in zip(base_vec, corr) if b == 1 and c == 0
            ) if condition != "base" else 0
            base_errors = sum(1 for x in base_vec if x == 0)

            mr["conditions"][condition] = {
                "accuracy": statistics.fmean(corr),
                "parse_rate": statistics.fmean(parse),
                "correct": sum(corr),
                "n": len(corr),
                "recovered_base_errors": recovered,
                "base_errors": base_errors,
                "recovery_rate": (
                    recovered / base_errors if base_errors else None
                ),
                "regressions_from_base": regressions,
                "mean_prompt_tokens": statistics.fmean(prompt_tokens),
                "mean_completion_tokens": statistics.fmean(completion_tokens),
                "mean_latency_s": statistics.fmean(lat),
            }

        src = [int(idx[(eid, model, "source_conditioned")]["correct"]) for eid in ids]
        ver = [int(idx[(eid, model, "verify")]["correct"]) for eid in ids]
        req = [int(idx[(eid, model, "requery")]["correct"]) for eid in ids]

        mr["paired_primary"]["source_minus_verify"] = paired_bootstrap(
            src, ver, SEED + model_i * 10 + 1
        )
        mr["paired_primary"]["source_minus_requery"] = paired_bootstrap(
            src, req, SEED + model_i * 10 + 2
        )
        mr["paired_secondary"] = {
            "source_minus_base": paired_bootstrap(
                src, base_vec, SEED + model_i * 10 + 3
            )
        }
        result["models"][model] = mr

    out = OUTDIR / "results.json"
    atomic_json(out, result)

    hashes = [
        OUTDIR / "manifest.jsonl",
        OUTDIR / "PROTOCOL_FROZEN.json",
        OUTDIR / "raw_results.jsonl",
        OUTDIR / "results.json",
    ]
    (OUTDIR / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(p)}  {p}\n" for p in hashes),
        encoding="utf-8",
    )

    print("=" * 72)
    print("U-MIRAGE RQ4c — EXPLORATORY HOSTED EXECUTOR RESULTS")
    print("=" * 72)
    for model in MODELS:
        m = result["models"][model]
        print(f"\nMODEL {model}")
        for c in CONDITIONS:
            z = m["conditions"][c]
            print(
                f"{c:20s} acc={z['accuracy']:.4f} "
                f"parse={z['parse_rate']:.4f} "
                f"recover={z['recovered_base_errors']}/{z['base_errors']} "
                f"regress={z['regressions_from_base']}"
            )
        for name, z in m["paired_primary"].items():
            print(
                f"{name}: delta={z['point']:+.4f} "
                f"CI95=[{z['ci95'][0]:+.4f},{z['ci95'][1]:+.4f}]"
            )
    print("\nRQ4C_EVALUATION_COMPLETE=TRUE")
    print("EVIDENCE_LEVEL=EXPLORATORY_DEVELOPMENT")
    print("CAL60_USED=FALSE")
    print("HOLDOUT360_USED=FALSE")


def status() -> None:
    manifest = read_jsonl(OUTDIR / "manifest.jsonl")
    raw = read_jsonl(OUTDIR / "raw_results.jsonl")
    expected = len(manifest) * len(MODELS) * len(CONDITIONS) if manifest else 0
    print("MANIFEST_N =", len(manifest))
    print("RAW_UNIQUE_CELLS =", len(rows_index(raw)))
    print("EXPECTED_CELLS =", expected)
    if expected:
        print("PERCENT =", round(100 * len(rows_index(raw)) / expected, 2))
    print("RESULTS_EXISTS =", (OUTDIR / "results.json").exists())
    print("CAL60_USED=FALSE")
    print("HOLDOUT360_USED=FALSE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=("freeze", "run", "evaluate", "status"),
    )
    args = ap.parse_args()

    if args.mode == "freeze":
        freeze()
    elif args.mode == "run":
        execute()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "status":
        status()


if __name__ == "__main__":
    main()