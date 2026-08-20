# SynapticDB

SynapticDB is a single-file memory store for AI agents. It combines keyword
search, vector search, and a graph of associations learned from how memories are
stored and retrieved. Every result carries a calibrated confidence value, so an
agent can tell "there is no good answer" from "here are ten weak ones".

> **Status: v0.1. One half is measured and working. The other half is not.**
>
> Hybrid search and confidence scoring work. Against the benchmark's locked
> baseline — BM25, FAISS, and a cross-encoder reranker — SynapticDB answers 17 of
> 25 associative queries to the baseline's 10, at identical direct recall of 25
> of 25.
>
> The association graph is implemented and enabled: temporal proximity,
> co-retrieval, decay, explicit feedback, manual links, and pruning. **The
> benchmark does not show the graph improving recall.** Before the graph existed,
> the same corpus already scored 17 of 25. The graph now supplies a reasoning
> path for the 7 answers the baseline misses, but hybrid search was already
> returning those 7 on its own. The graph explains results it did not produce.
>
> Use SynapticDB today for the search and confidence path. Treat the graph as
> experimental. [What does not work, and why](#what-does-not-work-and-why)
> explains each mechanism that fell short and how I know.

## Install

```console
pip install "synapticdb[embeddings]"
```

| install | pulls | embeddings come from |
|---|---|---|
| `synapticdb` | pydantic, numpy | your `embedding_fn`, which is then required |
| `synapticdb[embeddings]` | the above plus `sentence-transformers` | the built-in 384-dimensional `all-MiniLM-L6-v2`, downloaded on first use |

Take the plain install when you already have an embedding provider, or when you
do not want a PyTorch-sized dependency. Passing `embedding_fn` overrides the
default either way.

```python
from synapticdb import SynapticDB

with SynapticDB("synaptic.db") as memories:
    memories.store("Client X requires SOC2 for vendor deployments")
    result = memories.recall("deployment requirements for Client X")
    print(result.memories[0].content)
```

Every public result model keeps its typed Python fields and can also export
JSON-ready values. `to_dict()` returns a dictionary that `json.dumps()` accepts,
while `to_json()` returns a JSON string. UUIDs and timestamps become strings.

```python
payload = result.to_dict()
json_text = result.to_json()
```

Pass `embedding_fn` to `SynapticDB` to use a different local or hosted provider.
The function takes a string and returns a numeric sequence with a stable
dimension.

## How it works

The whole store is one SQLite file. There is no server, no index directory, and
no separate vector database. A path opens or creates a database; `":memory:"`
keeps it in process.

The schema is small: `memories`, an FTS5 virtual table mirrored from `memories`
by trigger, an `edges` table for the association graph, a `queries` table that
records each recall so feedback can refer back to it, and a `meta` key-value
table. Embeddings live in a `BLOB` on the memory row and are scored as one
cached NumPy matrix multiply. At v0 scale, that dense scan is cheaper than
maintaining an approximate-nearest-neighbor index.

One `recall()` runs five steps:

1. **Search twice.** FTS5 BM25 ranks by keyword. Cosine similarity over the
   stored embeddings ranks by meaning.
2. **Fuse.** Reciprocal rank fusion merges the two ranked lists, then min-max
   normalizes the result.
3. **Spread.** The top fused hits seed a bounded spreading activation pass
   across the edge graph. Edge weights decay to the current read time, so one
   query never sees two neighbors aged to different instants.
4. **Blend.** Activation claims a share of the final score set by graph
   maturity. A cold graph degrades to pure hybrid search, which is why an empty
   database behaves sensibly on its first query.
5. **Score.** Each returned memory gets a `score` and a `confidence`, described
   below, and is labeled `via` as `search`, `association`, or `both`.

Edges come from four origins: `temporal` (memories written close together),
`co_retrieval` (memories returned together), `explicit` (`connect()`), and
`semantic` (embedding similarity, shipped disabled — see below). Writing a
memory periodically triggers a maintenance pass that prunes weak edges.

## Knowing when there is no answer

Each result carries two numbers, and they answer different questions. `score`
ranks results within one recall and is normalized per query, so it cannot be
compared across queries. `confidence` is the cosine similarity between the query
and the memory. It is an absolute scale, so one threshold behaves the same on
every query.

```python
result = memories.recall("deployment requirements for Client X", min_confidence=0.6)
relevant = result.memories        # may be shorter than top_k, or empty
```

`min_confidence` drops results below the floor, so a recall can return fewer
than `top_k` memories, or none. The empty list is the useful part: it is how an
agent reports that it does not know. You can also filter afterward on
`item.confidence` if you would rather see everything.

On the benchmark, a 0.6 floor kept 41 of the 42 correct answers and rejected all
12 questions the corpus cannot answer. The highest-scoring unanswerable question
reached 0.540. Confidence AUC is 0.994, meaning a correct answer outscores an
unanswerable question 99.4% of the time.

**The threshold is only as good as your embeddings.** Confidence inherits
whatever separation your `embedding_fn` provides. A low-dimensional or poorly
fitted embedding leaves unrelated queries looking similar to everything, and no
threshold will repair that. Measure on your own data before relying on a fixed
floor.

Associations score low on `confidence` by construction. Weak textual similarity
is exactly why keyword and vector search missed them. Treat the floor as a dial:
raise it for direct matches only, lower it to admit associations.

## Reproducing the numbers

Every figure above comes from `bench/records/calibration-chained/`, which is
committed. Reproduce it with:

```console
uv sync --extra bench
uv run python -m bench --profile chained --retriever synaptic
```

The run takes about 35 seconds after a first-time model download, and prints a
per-seed table of direct hits, associative hits, path-backed wins, MRR, chain
coverage, and confidence AUC.

The benchmark is a merge gate, not a demonstration. It ingests a deterministic
synthetic corpus in frozen narrative order, replays warm-up events, and scores
holdout queries the retriever has never seen. Gold answers and reasoning paths
belong to the evaluator; no retriever receives them. Warm-up events may not
target a holdout chain's answer, and both the generator and the loader refuse a
dataset that violates that rule — the constraint is enforced in code that fails,
not prose that persuades.

`bench/README.md` documents the profiles, the parameter sweeps, and every
measurement.

## What an associative answer looks like

One of the 25 associative holdout queries, taken from that run. The baseline
misses it; SynapticDB returns it at rank 6 with confidence 0.640.

> **Query:** Why does the observatory dome trigger wind protection during calm
> weather?
>
> **Answer returned:** Converting knots before publishing reduced false closure
> alerts while preserving every genuinely windy shutdown.

The answer shares almost no words with the question. It is reachable only
through two intermediate memories, stored at different times and never
mentioned in the query:

1. Wind data reaches the dome controller through the Boreal weather adapter
   added for the new roof station.
2. Boreal labels its readings as meters per second but forwards the station's
   original knot values unchanged.

This is the shape of query hybrid search is built to lose, and the case the
association graph exists to serve. Be precise about the credit, though: the
graph records the path above, but hybrid search was already returning this
answer before the graph was built. See below.

## What does not work, and why

Four learning mechanisms are implemented and do not earn their place yet. I am
listing them because a benchmark that only reports wins is not a benchmark.

- **Semantic seeding ships disabled.** An A/B and a threshold sweep from 0.60 to
  0.85 showed it adds zero unique wins at every threshold. At 0.60 it created
  1069 semantic edges, and only 4 landed on a real associative chain. Embedding
  similarity is orthogonal to the reasoning the corpus asks for, and it
  duplicated what vector search already found. The code stays; the default does
  not.
- **Co-retrieval and feedback run but cannot be scored here.** Both grow edges as
  designed — feedback lifted average reinforcement from 0.000 to 0.164 — on
  warm-up topics the holdout never traverses. Warm-up is disjoint from the
  holdout on purpose. Making them overlap would be train/test leakage, so this
  needs a new dataset design rather than a parameter change.
- **Decay cannot be exercised by the benchmark.** Ingestion stamps memories at a
  2030 epoch, so every edge is future-dated relative to the read and decays by
  exactly 1.0. Unit tests cover decay instead. Fixing this means moving the
  epoch into the past.
- **Maintenance and pruning are a regression check, not a measurement.**
  Maintenance triggers on write, and the harness performs every write during
  ingest, before anything has aged. It deletes nothing. Measuring it needs
  writes interleaved with simulated time.

## Alternatives

If you want a vector store, use Chroma, LanceDB, or `sqlite-vec`. They are
mature, and vector search is their whole job. SynapticDB is worth a look when
you want keyword and vector search fused in a single file with no service to
run, and a confidence value you can threshold to decide whether to answer at
all. The association graph is the research question the project exists to test,
and it has not paid off yet.

## Development

SynapticDB requires Python 3.10 or newer. These checks are required in CI:

```console
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -W error -m pytest
uv run python -W error -m bench --profile smoke --retriever fixture --check --no-write
uv run python -W error -m bench --profile smoke --retriever synaptic --check --no-write
```

The smoke profile uses a fixture embedding and needs no model download. Build
the distributions with `uv run python -m build`. The package is distributed and
imported as `synapticdb`.

## License

MIT. See [LICENSE](https://github.com/xhillman/synapticdb/blob/main/LICENSE).
