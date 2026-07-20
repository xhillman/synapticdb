# SynapticDB

SynapticDB is a single-file memory store for AI agents where recall gets
smarter with use. It combines keyword and semantic search with a graph of
associations learned from how memories are stored and retrieved.

> **Status:** SynapticDB is an early v0 rebuild and does not yet expose its
> public memory API.

## Development

SynapticDB requires Python 3.10 or newer. Create or update the development
environment and run the test suite with:

```console
uv sync --extra dev
uv run pytest
```

Build the wheel and source distribution with:

```console
uv run python -m build
```

The package is distributed and imported as `synapticdb`.
