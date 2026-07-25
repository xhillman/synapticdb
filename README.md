# SynapticDB

SynapticDB is a single-file memory store for AI agents where recall gets
smarter with use. It combines keyword and semantic search with a graph of
associations learned from how memories are stored and retrieved.

> **Status:** v0 in progress. Durable hybrid search, spreading activation, and
> the learning mechanisms (temporal proximity, co-retrieval, decay, explicit
> feedback, and explicit links) are implemented. Associative-recall gains are
> still being validated against the in-repository benchmark.

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
