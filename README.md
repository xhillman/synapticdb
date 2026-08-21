<div align="center">

# SynapticDB

A single-file memory store that gives AI agents hybrid search, learned associations, and confidence-aware recall.

[PyPI](https://pypi.org/project/synapticdb/) · [Benchmark](./bench/README.md) · [Report a bug](https://github.com/xhillman/synapticdb/issues)

</div>

---

## Why

AI agents need to retrieve useful context and recognize when no useful context exists. SynapticDB combines keyword search, vector search, and a graph of associations in one SQLite file. Each result includes an absolute confidence value, so an agent can reject weak matches instead of treating every ranked result as an answer.

SynapticDB v0.1 is alpha software. Hybrid search and confidence scoring are measured and working. The association graph is implemented, but current benchmark results do not show that the graph improves recall on its own.

## Install

Install the built-in local embedding model:

```console
pip install "synapticdb[embeddings]"
```

Install the smaller core package when you provide an embedding function:

```console
pip install synapticdb
```

**Requires Python 3.10 or newer.** The `embeddings` extra downloads the 384-dimensional `all-MiniLM-L6-v2` model on first use.

## Quick start

```python
from synapticdb import SynapticDB

with SynapticDB("synaptic.db") as memories:
    memories.store("Client X requires SOC2 for vendor deployments")
    result = memories.recall("deployment requirements for Client X")
    print(result.memories[0].content)
```

```text
Client X requires SOC2 for vendor deployments
```

The database, full-text index, embeddings, associations, and recall history all live in `synaptic.db`. Use `":memory:"` instead for a temporary in-process database.

## Features

- **Run without a service** — SQLite stores the complete memory system in one portable file.
- **Match words and meaning** — BM25 keyword search and cosine vector search are fused into one ranking.
- **Reject weak answers** — absolute confidence values support one threshold across different queries.
- **Learn relationships** — temporal, co-retrieval, feedback, and manual links build an association graph.
- **Bring your own embeddings** — a custom local or hosted embedding function can replace the built-in model.

## Usage

### Return nothing when evidence is weak

`min_confidence` filters results before SynapticDB records retrieval learning. A recall can return fewer than `top_k` memories, including none.

```python
result = memories.recall(
    "deployment requirements for Client X",
    top_k=5,
    min_confidence=0.6,
)
relevant = result.memories

if not relevant:
    print("No reliable answer found")
```

`score` ranks memories within one recall. `confidence` measures query-to-memory similarity and can be compared across recalls.

### Store metadata and filter results

```python
memories.store(
    "Client X requires SOC 2 for vendor deployments",
    metadata={"client": "x", "topic": "compliance"},
)

result = memories.recall(
    "deployment requirements",
    where={"client": "x"},
)
```

### Teach explicit relationships

```python
requirement = memories.store("Client X requires SOC 2")
deployment = memories.store("Project Atlas deploys to Client X")

memories.connect(requirement.id, deployment.id)

result = memories.recall("What affects the Atlas deployment?")
memories.feedback(result.query_id, positive=True)
```

Positive feedback strengthens the relationships used by a recall. Negative feedback weakens them without creating new links.

### Export results as JSON

```python
payload = result.to_dict()
json_text = result.to_json()
```

Every public result model retains typed Python fields. JSON exports convert UUIDs and timestamps to strings. `get()`, `connect()`, `forget()`, and `feedback()` accept UUID objects or their string forms.

## How it works

One `recall()` follows five bounded steps:

1. FTS5 BM25 ranks keyword matches, and cosine similarity ranks embedding matches.
2. Reciprocal rank fusion combines both result lists.
3. The highest-ranked memories seed spreading activation across the association graph.
4. Graph maturity controls how much activation contributes to the final ranking.
5. SynapticDB returns each memory with its ranking score, confidence, and retrieval source.

A cold or empty graph falls back to hybrid search. Edge weights decay over time, and periodic maintenance prunes weak edges.

## Benchmark

The committed chained benchmark compares SynapticDB with a locked BM25, FAISS, and cross-encoder baseline. SynapticDB answers 17 of 25 associative queries, compared with 10 of 25 for the baseline. Both answer all 25 direct queries.

A confidence floor of `0.6` keeps 41 of 42 correct answers and rejects all 12 unanswerable questions in the benchmark. Confidence AUC is `0.994`.

Reproduce the results after the first model download:

```console
uv sync --extra bench
uv run --extra bench python -m bench --profile chained --retriever synaptic
```

The run takes about 35 seconds on the development machine. See the [benchmark documentation](./bench/README.md) for profiles, measurements, and stored records.

## What an associative answer looks like

For one associative holdout query, the locked baseline misses the answer. SynapticDB returns it at rank 6 with confidence `0.640`.

> **Query:** Why does the observatory dome trigger wind protection during calm weather?
>
> **Answer:** Converting knots before publishing reduced false closure alerts while preserving every genuinely windy shutdown.

The query and answer share almost no words. The association graph connects them through two intermediate memories:

1. Wind data reaches the controller through the Boreal weather adapter.
2. Boreal labels readings as meters per second but forwards knot values unchanged.

The graph records a useful reasoning path, but it did not create this benchmark win. SynapticDB's hybrid search already returned the answer before the graph existed.

## What does not work yet

- **Semantic seeding is disabled.** A threshold sweep produced no unique wins. At `0.60`, only 4 of 1,069 semantic edges landed on a real associative chain.
- **Co-retrieval and feedback are not measured end to end.** Both update edges, but benchmark warm-up topics do not overlap the holdout paths. Adding overlap would leak training data into evaluation.
- **Decay and pruning are not measured end to end.** Future-dated ingestion and write-only maintenance prevent the benchmark from aging edges. Unit tests cover both behaviors.
- **Confidence depends on embedding quality.** Measure a threshold on your own data before using confidence as a production decision boundary.

## Alternatives

If you only need vector search, start with a mature tool such as Chroma, LanceDB, or `sqlite-vec`. Choose SynapticDB when you want fused keyword and vector search in one SQLite file, plus a confidence value that can suppress weak answers. Treat the association graph as an experimental research feature.

## Development

```console
git clone https://github.com/xhillman/synapticdb.git
cd synapticdb
uv sync --extra dev
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy
uv run --extra dev python -W error -m pytest
uv run --extra dev python -W error -m bench --profile smoke --retriever fixture --check --no-write
uv run --extra dev python -W error -m bench --profile smoke --retriever synaptic --check --no-write
```

Build the wheel and source distribution with `uv run --extra dev python -m build`.

## Contributing

Contributions are welcome. Open an issue before starting a large change so the proposed scope can be reviewed first.

## License

MIT © [Xavier Hillman](https://github.com/xhillman). See [LICENSE](./LICENSE).
