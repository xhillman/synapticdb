import sys
from collections.abc import Sequence
from types import ModuleType
from typing import cast

import pytest

from synapticdb import EmbeddingError
from synapticdb.embeddings import Embedder


def test_custom_embedding_function_wins_and_dimension_is_captured() -> None:
    calls: list[str] = []

    def embed(text: str) -> Sequence[float]:
        calls.append(text)
        return (1.0, 0.0)

    wrapper = Embedder(embed)
    assert wrapper.embed("first") == (1.0, 0.0)
    assert calls == ["first"]


def test_custom_embedding_failures_are_wrapped() -> None:
    def fail(text: str) -> Sequence[float]:
        del text
        raise RuntimeError("service stopped")

    with pytest.raises(EmbeddingError, match="service stopped"):
        Embedder(fail).embed("query")


@pytest.mark.parametrize(
    "vector",
    [(), (0.0, 0.0), (float("nan"), 1.0), ("bad", 1.0)],
)
def test_invalid_custom_vectors_raise_embedding_error(vector: tuple[object, ...]) -> None:
    with pytest.raises(EmbeddingError):
        Embedder(lambda text: cast(Sequence[float], vector)).embed("query")


def test_embedder_rejects_dimension_changes() -> None:
    vectors = iter(((1.0, 0.0), (1.0, 0.0, 0.0)))
    embedder = Embedder(lambda text: next(vectors))
    embedder.embed("first")
    with pytest.raises(EmbeddingError, match="does not match"):
        embedder.embed("second")


def test_missing_default_dependency_raises_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_dependency = ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", missing_dependency)
    embedder = Embedder(None)
    with pytest.raises(EmbeddingError, match=r"pip install synapticdb\[embeddings\]"):
        embedder.embed("query")
