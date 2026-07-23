"""Embedding function boundary and lazy MiniLM default."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from synapticdb.models import EmbeddingError

EmbeddingFunction = Callable[[str], Sequence[float]]
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_MAX_EMBEDDING_DIMENSION = 4096


class Encoder(Protocol):
    def encode(self, text: str, *, convert_to_numpy: bool) -> object: ...


class Embedder:
    """Own a custom embedding function or one lazily loaded default model."""

    def __init__(
        self,
        embedding_fn: EmbeddingFunction | None,
        *,
        expected_dimension: int | None = None,
    ) -> None:
        if expected_dimension is not None and not 1 <= expected_dimension <= _MAX_EMBEDDING_DIMENSION:
            raise EmbeddingError("stored embedding dimension is outside the supported range")
        self._embedding_fn = embedding_fn
        self._expected_dimension = expected_dimension
        self._model: Encoder | None = None

    def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("embedding text must be a non-empty string")
        raw = self._call_custom(text) if self._embedding_fn is not None else self._call_default(text)
        vector = _validated_vector(raw)
        if self._expected_dimension is None:
            self._expected_dimension = len(vector)
        if len(vector) != self._expected_dimension:
            raise EmbeddingError(
                f"embedding dimension {len(vector)} does not match expected dimension {self._expected_dimension}"
            )
        return vector

    def _call_custom(self, text: str) -> Sequence[float]:
        function = self._embedding_fn
        if function is None:
            raise RuntimeError("custom embedding function is unavailable")
        try:
            return function(text)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"embedding_fn failed: {exc}") from exc

    def _call_default(self, text: str) -> Sequence[float]:
        model = self._load_default()
        try:
            return cast(Sequence[float], model.encode(text, convert_to_numpy=True))
        except Exception as exc:
            raise EmbeddingError(f"default embedding model failed: {exc}") from exc

    def _load_default(self) -> Encoder:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "default embeddings are unavailable; install with `pip install synapticdb[embeddings]`"
            ) from exc
        try:
            model_type = cast(Any, SentenceTransformer)
            self._model = cast(Encoder, model_type(_DEFAULT_MODEL))
        except Exception as exc:
            raise EmbeddingError(f"could not load default embedding model {_DEFAULT_MODEL!r}: {exc}") from exc
        return self._model


def _validated_vector(raw: Sequence[float]) -> tuple[float, ...]:
    try:
        dimension = len(raw)
    except (TypeError, OverflowError) as exc:
        raise EmbeddingError("embedding_fn must return a sized sequence of numbers") from exc
    if not 1 <= dimension <= _MAX_EMBEDDING_DIMENSION:
        raise EmbeddingError(f"embedding dimension must be between 1 and {_MAX_EMBEDDING_DIMENSION}")
    try:
        vector = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError("embedding_fn must return only numeric values") from exc
    if any(not math.isfinite(value) for value in vector):
        raise EmbeddingError("embedding values must be finite")
    norm_squared = sum(value * value for value in vector)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise EmbeddingError("embedding must be a finite non-zero vector")
    return vector
