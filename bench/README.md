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

Override any PRD section 9 parameter with a repeatable `--param KEY=JSON`, and
sweep one across several values with `bench/sweep.py`:

```bash
python -m bench --profile full --retriever synaptic --param activation_blend_weight=0.9
python -m bench.sweep temporal_link '[600,3,0.2]' '[600,3,0.6]'
```

Every report records the **full effective parameter dict**, so a promoted record
always names the configuration that produced it.

**Sweeping is bounded evidence.** The holdout is 25 associative queries, and
tuning hard against it is how a benchmark gets overfit. Prefer one parameter at
a time, prefer values with a mechanical reason to help, and treat a config that
scores well without an explanation as a finding to investigate rather than a
default to adopt.

## Simulated time and the directional gates

The harness can replay a run against a simulated clock, which is what makes
decay, pruning, and usage effects observable at all. Every flag defaults to
zero, and with all of them absent a run is identical to one taken before the
clock existed.

```bash
python -m bench --profile full --retriever synaptic \
  --warmup-span-days 30 --query-offset-days 7 --decay-probe-days 90 \
  --measure trajectory --measure decay --measure diversity
```

Spans are measured **from the 2030 ingest epoch, not from today**. The epoch is
in the future relative to wall-clock time, so a run without a clock leaves every
edge future-dated and decay clamped to 1.0 — which is why v1 could never age a
graph.

Three gates, each asserting a relationship the PRD claims rather than a number
we chose. Directional by design: a threshold picked today would be a guess, and
a guessed threshold invites being renegotiated the moment it is missed.

| measurement | claim under test |
|---|---|
| `trajectory` | using the system improves it: warm associative hits ≥ cold |
| `decay` | an aged graph still serves search: aged direct hits ≥ warm |
| `diversity` | learning does not collapse results: repeat distinct ≥ first |

`trajectory` scores a **separate cold instance** rather than an earlier pass on
the warm one, because recall itself writes co-retrieval edges — a first pass
cannot be repeated as a clean baseline.

**Phase A cannot measure co-retrieval or explicit feedback.** Both need a
warm-up that traverses the same chains as the holdout, which needs the
multi-query chains of Phase B. What Phase A adds is the ability to age a graph
and to detect a collapse.

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

Phase 5 learning milestones are recorded separately:

- `bench/records/phase5-1-semantic-seeding/`
- `bench/records/phase5-2-temporal-proximity/`
- `bench/records/phase5-3-decay/`
- `bench/records/phase5-4-co-retrieval/`
- `bench/records/phase5-5-feedback/`

The Phase 5.2 record preserves 25/25 direct hits and 17/25 associative hits.
Temporal proximity raises path-backed associative unique wins from 0/25 to
7/25. The complete Phase 5 gate remains open until later learning mechanisms
reach 10/25 unique wins.

The Phase 5.3 decay record is byte-for-byte equivalent to Phase 5.2 in every
per-query outcome, which is the expected result. **The benchmark cannot
exercise edge decay:** ingestion stamps memories at `_INGEST_EPOCH`
(2030-01-01), so every edge carries a timestamp ahead of the read time and
decays by a factor of exactly 1.0. Decay is covered by unit tests instead. A
future phase that wants decay in the gate must move the ingest epoch into the
past and offset the query times with it.

The Phase 5.4 co-retrieval record matches Phase 5.3 on every per-query outcome.
**This harness structurally cannot reward co-retrieval, which is a property of
the benchmark rather than of the mechanism.** A pair must reach the top 5 of two
separate queries before its edge clears the 0.0625 weight needed to carry a
full-energy seed, but the 6 warm-up events are deliberately disjoint from the 25
associative chains and each holdout query targets a different chain. Making them
overlap would be teaching to the test.

The mechanism does work: across ingest, warm-up, and the 50 holdout queries the
graph grew from 150 temporal edges to 608, and 31 co-retrieval edges were
reinforced past the propagation threshold. None changed an outcome, because they
linked memories the fusion ranking had already returned together. Note this is a
different failure from semantic seeding below: that mechanism duplicated what
vector search already found, whereas co-retrieval's value case (repeated
querying of the same topics) is simply absent from this corpus.

The Phase 5.5 feedback record matches Phase 5.4 on every per-query outcome, for
the same reason plus one more. The warm-up's 5 positive and 1 negative events
grew co-retrieval edges from 60 to 235 and lifted average reinforcement from
0.000 to 0.164, so the mechanism plainly runs — but on warm-up topics the
holdout never traverses. Beyond that, a co-retrieval edge reinforced by one
positive feedback reaches only `0.05 + 0.15·(1-0.05) = 0.1925`, still under the
0.2 weight a temporal edge starts at, so even strengthened edges rank behind
ordinary proximity links.

**Any future harness change that lets feedback reach the holdout chains must
not simply overlap warm-up with holdout.** That is train/test leakage, and the
gate would stop measuring generalization. Exercising feedback honestly needs
warm-up queries that are topically distinct yet traverse the same chains — a
dataset design problem, not a parameter to tune.

**Semantic seeding is disabled by default** (`_params["semantic_seed"] = None`).
A full-corpus A/B and a threshold sweep showed it contributes +0 unique wins at
every threshold from 0.60 to 0.85: temporal-only and semantic+temporal both
score 7/25, because embedding similarity is orthogonal to the benchmark's
associative chains (a 0.60 threshold produced 1069 semantic edges, only 4 of
them on real chains). Per the merge-gate rule (a mechanism ships only if it
improves the benchmark), it does not ship enabled. The code and
`SEMANTIC_SEED_CALIBRATION` values remain; pass `--semantic-threshold` to the
bench (or set `_params["semantic_seed"]`) to re-run the A/B in a later phase —
e.g. once co-retrieval and feedback might reinforce a useful edge subset.

Gold relevance IDs, required path IDs, and corpus `linked_memory_ids` are
owned by the evaluator. Retriever implementations receive none of them. Phase
3 reports no `path_benchmark_ids`. Activation will translate stored path
evidence through the UUID-to-benchmark-ID map in a later phase.
