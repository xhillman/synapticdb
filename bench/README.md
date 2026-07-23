# SynapticDB benchmark

The harness is an in-repository merge gate and is intentionally excluded from
the wheel. It ingests memories in frozen narrative order, replays disjoint
warm-up events, and evaluates direct and associative holdout queries.

Dataset IDs such as `mem-0001` are evaluator-owned `benchmark_id` values, not
product IDs. A Synaptic adapter maps them to the UUIDs returned by `remember()`.
The ingestion schedule supplies controlled time offsets for temporal learning;
archived source timestamps are not passed as memory metadata.

Run the dependency-free CI profile:

```bash
python -m bench --profile smoke --retriever fixture --check
python -m bench --profile smoke --retriever synaptic --check
```

Install the full stack and reproduce the canonical baseline:

```bash
pip install -e '.[bench]'
python -m bench --profile full --retriever baseline --check
```

The full corpus is deterministic and fully synthetic: its organizations,
systems, incidents, operating rules, and values are fictional. Regenerate the
committed fixtures with `python -m bench.generate_dataset`.

The first full run downloads the revision-pinned
`cross-encoder/ms-marco-MiniLM-L-6-v2`. A valid reproduction on synthetic
dataset version 1 returns 25/25 direct and 10/25 associative hits at top 10.
No fallback is permitted. Reports are written beneath `bench/artifacts/`
unless `--no-write` is supplied.

The candidate gate, used once the Synaptic adapter exists, requires direct
recall within five percentage points of the baseline and at least 10/25
path-backed associative unique wins on every requested seed.

Canonical results live in `bench/records/`, one directory per promoted run
(`report.md` + `report.json`, stamped with the git commit of the code under
test). `bench/artifacts/` is transient and gitignored; promote a run by
copying its artifact directory into `bench/records/`. Every PR that changes
retrieval, learning, or parameter defaults includes a before/after table
diffed against the latest committed record.

The Phase 3 pre-learning baseline is `bench/records/phase3-prelearning/`,
produced by:

```bash
python -m bench --profile full --retriever synaptic --run-id phase3-prelearning
```

On synthetic dataset version 1 with seed 1337, Phase 3 returned 25/25 direct
hits and 17/25 associative hits. Path-backed unique wins remained 0/25 because
activation does not exist yet. The full learning gate therefore fails as
expected until later phases add association paths.

Gold relevance IDs, required path IDs, and corpus `linked_memory_ids` are
owned by the evaluator. Retriever implementations receive none of them. Phase
3 reports no `path_benchmark_ids`. Activation will translate stored path
evidence through the UUID-to-benchmark-ID map in a later phase.
