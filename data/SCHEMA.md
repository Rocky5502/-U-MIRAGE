# Canonical episode schema

Every prepared benchmark is normalized to JSONL before inference. A canonical record is:

```json
{"id":"sample-001","images":["data/cache/a.jpg","data/cache/b.jpg"],"question":"Which image contains the red object?","answer":"image 2","source_label":"clean","metadata":{"benchmark":"mioh","split":"test"}}
```

Required fields are `id`, non-empty `images`, `question`, and `answer`. `source_label` is one of `clean`, `visual`, `integration`, or `reasoning`. Benchmark-specific fields may be preserved inside `metadata`; official scoring uses frozen canonical IDs.

Image paths are resolved by the preparation stage. Raw benchmark media is never required to live in git.
