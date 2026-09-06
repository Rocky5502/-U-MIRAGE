# Canonical episode schema

Each prepared benchmark is normalized to JSONL before inference. One line:

```json
{"id":"sample-001","images":["data/cache/.../a.jpg","data/cache/.../b.jpg"],"question":"Which image contains the red object?","answer":"image 2","source_label":"clean","metadata":{"benchmark":"mioh","split":"test"}}
```

Required: `id`, non-empty `images`, `question`, `answer`. `source_label` is one of `clean`, `visual`, `integration`, `reasoning`. Benchmark adapters may preserve original fields inside `metadata`, but official scoring uses the canonical fields and frozen sample IDs.
