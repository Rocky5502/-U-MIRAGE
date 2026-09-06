from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a small deterministic MIHBench cohort into U-MIRAGE Episode JSONL. "
            "This is intended for smoke/pilot runs and never redistributes upstream media."
        )
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="data/materialized/mihbench_pilot")
    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.buffer_size <= 0:
        raise SystemExit("--buffer-size must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `pip install -e .[data]` before materializing datasets") from exc

    output_dir = Path(args.output_dir)
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "MM-Hallu/MIHBench",
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    episodes_path = output_dir / "episodes.jsonl"
    task_counts: Counter[str] = Counter()
    rows_written = 0

    with episodes_path.open("w", encoding="utf-8") as handle:
        for sampled_index, row in enumerate(dataset):
            if rows_written >= args.limit:
                break

            images = row.get("images") or []
            if not images:
                continue

            episode_id = f"mihbench-{args.seed}-{rows_written:05d}"
            episode_image_dir = image_root / episode_id
            episode_image_dir.mkdir(parents=True, exist_ok=True)

            image_paths: list[str] = []
            for image_index, image in enumerate(images):
                if image is None:
                    image_paths = []
                    break
                image_path = episode_image_dir / f"image_{image_index:02d}.png"
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                if not hasattr(image, "save"):
                    image_paths = []
                    break
                image.save(image_path, format="PNG")
                image_paths.append(image_path.as_posix())

            if not image_paths:
                continue

            question = str(row.get("question", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            if not question or label not in {"yes", "no"}:
                continue

            task = str(row.get("task", "unknown"))
            task_counts[task] += 1
            metadata = {
                "dataset": "MIHBench",
                "hf_id": "MM-Hallu/MIHBench",
                "upstream_sample_index": sampled_index,
                "task": task,
                "num_images": _json_safe(row.get("num_images", len(image_paths))),
                "image_names": _json_safe(row.get("image_names")),
                "injected": _json_safe(row.get("injected")),
                "object_counts": _json_safe(row.get("object_counts")),
                "materialization_seed": args.seed,
                "source_type": "existing_benchmark_clean",
            }
            episode = {
                "id": episode_id,
                "images": image_paths,
                "question": question,
                "answer": label,
                "source_label": "clean",
                "metadata": metadata,
            }
            handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise SystemExit("No valid MIHBench rows were materialized")

    manifest = {
        "dataset": "MIHBench",
        "hf_id": "MM-Hallu/MIHBench",
        "requested_revision": args.revision,
        "split": args.split,
        "streaming": True,
        "shuffle_seed": args.seed,
        "shuffle_buffer_size": args.buffer_size,
        "requested_limit": args.limit,
        "materialized_examples": rows_written,
        "task_counts": dict(sorted(task_counts.items())),
        "episodes_jsonl": episodes_path.as_posix(),
        "episodes_sha256": _sha256(episodes_path),
        "redistribution": (
            "Local materialization only. Do not redistribute upstream images without separately "
            "verifying MIHBench and underlying COCO media terms."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"materialized={rows_written}")
    print(f"episodes={episodes_path}")
    print(f"manifest={manifest_path}")
    print(f"sha256={manifest['episodes_sha256']}")


if __name__ == "__main__":
    main()
