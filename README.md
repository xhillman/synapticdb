# SynapticDB

SynapticDB is a single-file memory store for AI agents where recall gets
smarter with use. It combines keyword and semantic search with a graph of
associations learned from how memories are stored and retrieved.

> **Status:** v0.1. Hybrid search and confidence scoring are measured and
> working: against the in-repository benchmark's locked baseline, SynapticDB
> returns 17/25 associative queries to the baseline's 10/25 at equal direct
> recall, and a 0.6 confidence floor rejects every unanswerable query.
>
> The association graph is implemented and enabled — temporal proximity,
> co-retrieval, decay, feedback, explicit links, and pruning — but **the
> benchmark does not yet show it improving recall.** Rank quality is
> unchanged from before the graph existed. Treat the graph as experimental and
> the search plus confidence path as the reason to use this today.

Install the embedding-enabled package:

```console
pip install "synapticdb[embeddings]"
```

Then remember and recall:

```python
from synapticdb import Synaptic
with Synaptic(":memory:") as memories:
    memories.remember("Client X requires SOC2 for vendor deployments")
    result = memories.recall("deployment requirements for Client X")
    print(result.memories[0].memory.content)
```

Pass `embedding_fn` to `Synaptic` to use a local or hosted embedding provider.
The function must accept a string and return a numeric sequence with a stable
dimension.

## Knowing when there is no answer

Each result carries two numbers, and they answer different questions. `score`
ranks results within one recall. `confidence` measures how well a memory
matches the query, on an absolute scale you can threshold across queries:

```python
result = memories.recall("deployment requirements for Client X", min_confidence=0.6)
relevant = result.memories        # may be shorter than top_k, or empty
```

`min_confidence` drops results below the floor, so a recall can return fewer
than `top_k` memories — or none. That empty list is the useful part: it is how
an agent tells "no good answer" from "here are ten weak ones". You can also
filter after the fact on `item.confidence` if you would rather see everything.

On the in-repository benchmark, a 0.6 threshold kept 41 of 42 correct answers
and rejected all 12 questions the corpus could not answer.

**The threshold is only as good as your embeddings.** Confidence is cosine
similarity, so it inherits whatever separation your `embedding_fn` provides. The
benchmark figures use the default 384-dimensional model; a low-dimensional or
poorly fitted embedding leaves unrelated queries looking similar to everything,
and no threshold will help. Check the numbers on your own data before relying on
a fixed floor.

Association results score low on `confidence` — weak textual similarity is
exactly why keyword and vector search missed them. Treat the threshold as a
dial: raise it for direct matches only, lower it to admit associations.

## Development

SynapticDB requires Python 3.10 or newer. Create or update the development
environment and run the test suite with:

```console
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -W error -m pytest
uv run python -W error -m bench --profile smoke --retriever fixture --check --no-write
uv run python -W error -m bench --profile smoke --retriever synaptic --check --no-write
```

These checks are required in CI.

Build the wheel and source distribution with:

```console
uv run python -m build
```

The package is distributed and imported as `synapticdb`.
