# Benchmark data

`full/` is a frozen, fully synthetic corpus with 500 memories and 50 queries.
All organizations, systems, incidents, operating rules, and values are
fictional. File order is the narrative ingest order. `linked_memory_ids`,
expected IDs, and intermediate IDs are evaluator annotations and are never
passed to a retriever.

`python -m bench.generate_dataset` deterministically rebuilds both profiles and
their checksum manifests. The full corpus contains 25 four-memory associative
chains, 25 direct-answer memories, and 375 unrelated background memories. Its
four memory types and four metadata sources each appear exactly 125 times.

`schedule.jsonl` assigns each `benchmark_id` a controlled ingestion offset.
Adjacent related memories are 120 seconds apart; unrelated boundaries are 900
seconds apart. This lets the future adapter exercise the real 600-second
temporal-learning window without pretending the archived source timestamps are
the times at which `store()` was called.

`smoke/` is a deterministic 50-memory subset with five direct and five
associative queries. It validates the harness in CI without FAISS or model
downloads. Neither profile uses evaluation queries as feedback events.
Warm-up feedback uses the public query-level `positive` boolean, including one
negative event, rather than supplying corrected memory IDs.

The full profile pins the historical cross-encoder runtime in the `bench`
extra. This is part of the benchmark contract: dependency drift can change
rankings even when the model name and retrieval code are unchanged. Synthetic
dataset version 1 reproduces 25/25 direct and 10/25 associative baseline hits.
