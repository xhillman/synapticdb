"""Locked historical BM25 + FAISS + cross-encoder baseline."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import (
    MAX_EMBEDDING_FEATURES,
    MAX_MEMORY_COUNT,
    MAX_RECORD_CHARS,
    MAX_RERANK_CANDIDATES,
    MAX_TOP_K,
)
from .dataset import MemoryRecord
from .retrievers import Retrieval

_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class BaselineConfig:
    keyword_top_k: int = 40
    semantic_top_k: int = 40
    fusion_k: int = 60
    final_top_k: int = 10
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    embedding_max_features: int = 4096
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_revision: str = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    cross_encoder_weight: float = 0.60
    keyword_weight: float = 0.20
    semantic_weight: float = 0.15
    fusion_weight: float = 0.05

    def __post_init__(self) -> None:
        integer_values = (
            self.keyword_top_k,
            self.semantic_top_k,
            self.fusion_k,
            self.final_top_k,
            self.embedding_max_features,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("baseline integer settings must be positive")
        if self.keyword_top_k + self.semantic_top_k > MAX_RERANK_CANDIDATES:
            raise ValueError(f"baseline reranking is capped at {MAX_RERANK_CANDIDATES} candidates")
        if self.final_top_k > MAX_TOP_K or self.embedding_max_features > MAX_EMBEDDING_FEATURES:
            raise ValueError("baseline output and embedding dimensions exceed hard limits")
        if self.bm25_k1 <= 0.0 or not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("BM25 settings require positive k1 and b between 0 and 1")
        weights = (self.cross_encoder_weight, self.keyword_weight, self.semantic_weight, self.fusion_weight)
        if any(weight < 0.0 for weight in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("baseline weights must be non-negative and sum to 1")
        if not self.cross_encoder_model_name or not self.cross_encoder_revision:
            raise ValueError("baseline model name and revision are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LockedBaseline:
    """The exact baseline configuration used by the canonical archived run."""

    name = "baseline"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self._ready = False
        self._np: Any = None
        self._doc_ids: list[str] = []
        self._doc_text: dict[str, str] = {}
        self._tokens: dict[str, tuple[str, ...]] = {}
        self._bm25: Any = None
        self._vectorizer: Any = None
        self._index: Any = None
        self._reranker: Any = None

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
        _validate_memories(memories)
        try:
            import faiss
            import numpy as np
            from rank_bm25 import BM25Okapi
            from sentence_transformers import CrossEncoder
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError(
                "full baseline dependencies are unavailable; install with `pip install -e .[bench]`"
            ) from exc

        self._np = np
        self._doc_ids = [memory.benchmark_id for memory in memories]
        self._doc_text = {memory.benchmark_id: memory.content for memory in memories}
        self._tokens = {memory.benchmark_id: _tokenize(memory.content) for memory in memories}
        self._bm25 = BM25Okapi(
            list(self._tokens.values()),
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=self.config.embedding_max_features,
        )
        matrix = self._vectorizer.fit_transform([memory.content for memory in memories])
        embeddings = np.asarray(matrix.toarray(), dtype="float32")
        embeddings = _normalize(embeddings, np)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._reranker = CrossEncoder(
            self.config.cross_encoder_model_name,
            device="cpu",
            revision=self.config.cross_encoder_revision,
            config_args={"resume_download": None},
        )
        self._ready = True

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        if not self._ready:
            raise RuntimeError("baseline must be ingested before recall")
        _validate_query(text, top_k)
        keyword = self._keyword(text)
        semantic = self._semantic(text)
        fusion = _rrf(keyword, semantic, self.config.fusion_k)
        ranked = self._rerank(text, keyword, semantic, fusion)
        limit = min(top_k, self.config.final_top_k)
        return Retrieval(tuple(node_id for node_id, _ in ranked[:limit]))

    def feedback(
        self,
        retrieval: Retrieval,
        *,
        positive: bool,
    ) -> None:
        if not isinstance(retrieval, Retrieval) or not isinstance(positive, bool):
            raise TypeError("feedback requires a Retrieval and boolean positive value")

    def _keyword(self, text: str) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(list(_tokenize(text)))
        if len(scores) != len(self._doc_ids):
            raise RuntimeError("BM25 returned an unexpected score count")
        ranked = [(self._doc_ids[index], float(score)) for index, score in enumerate(scores) if score > 0]
        return _sort(ranked)[: self.config.keyword_top_k]

    def _semantic(self, text: str) -> list[tuple[str, float]]:
        vector = self._vectorizer.transform([text])
        query = self._np.asarray(vector.toarray(), dtype="float32")
        query = _normalize(query, self._np)
        count = min(self.config.semantic_top_k, len(self._doc_ids))
        scores, indices = self._index.search(query, count)
        if len(indices[0]) != count or len(scores[0]) != count:
            raise RuntimeError("FAISS returned an unexpected result count")
        ranked = [
            (self._doc_ids[int(index)], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]
        return _sort(ranked)

    def _rerank(
        self,
        text: str,
        keyword: list[tuple[str, float]],
        semantic: list[tuple[str, float]],
        fusion: dict[str, float],
    ) -> list[tuple[str, float]]:
        keyword_scores = dict(keyword)
        semantic_scores = dict(semantic)
        node_ids = sorted(fusion)
        raw = self._reranker.predict([[text, self._doc_text[node_id]] for node_id in node_ids])
        if len(raw) != len(node_ids):
            raise RuntimeError("cross-encoder returned an unexpected score count")
        cross_scores = {node_id: _sigmoid(float(score)) for node_id, score in zip(node_ids, raw, strict=True)}
        max_keyword = max(keyword_scores.values(), default=1.0)
        max_semantic = max(semantic_scores.values(), default=1.0)
        max_fusion = max(fusion.values(), default=1.0)
        rows = []
        for node_id in node_ids:
            score = (
                self.config.cross_encoder_weight * cross_scores[node_id]
                + self.config.keyword_weight * keyword_scores.get(node_id, 0.0) / max_keyword
                + self.config.semantic_weight * semantic_scores.get(node_id, 0.0) / max_semantic
                + self.config.fusion_weight * fusion[node_id] / max_fusion
            )
            rows.append((node_id, score))
        return _sort(rows)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.lower()))


def _normalize(array: Any, np: Any) -> Any:
    if getattr(array, "ndim", None) != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("normalization requires a non-empty two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _rrf(
    keyword: Sequence[tuple[str, float]],
    semantic: Sequence[tuple[str, float]],
    fusion_k: int,
) -> dict[str, float]:
    if fusion_k <= 0:
        raise ValueError("fusion_k must be positive")
    if len(keyword) + len(semantic) > MAX_RERANK_CANDIDATES:
        raise ValueError(f"fusion accepts at most {MAX_RERANK_CANDIDATES} ranked candidates")
    scores: dict[str, float] = {}
    for ranked in (keyword, semantic):
        for rank, (node_id, _) in enumerate(ranked, 1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (fusion_k + rank)
    return scores


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _sort(rows: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def _validate_memories(memories: Sequence[MemoryRecord]) -> None:
    if not 1 <= len(memories) <= MAX_MEMORY_COUNT:
        raise ValueError(f"baseline ingest accepts 1 to {MAX_MEMORY_COUNT} memories")
    benchmark_ids = [memory.benchmark_id for memory in memories]
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("baseline memory IDs must be unique")
    if any(not memory.content.strip() or len(memory.content) > MAX_RECORD_CHARS for memory in memories):
        raise ValueError("baseline memories require bounded non-empty content")


def _validate_query(text: str, top_k: int) -> None:
    if not text.strip() or len(text) > MAX_RECORD_CHARS:
        raise ValueError("recall text must be bounded and non-empty")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
