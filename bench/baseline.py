"""Locked historical BM25 + FAISS + cross-encoder baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Sequence

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LockedBaseline:
    """The exact baseline configuration used by the canonical archived run."""

    name = "baseline"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self._ready = False

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
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
        )
        self._ready = True

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        if not self._ready:
            raise RuntimeError("baseline must be ingested before recall")
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
        del retrieval, positive

    def _keyword(self, text: str) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(list(_tokenize(text)))
        ranked = [
            (self._doc_ids[index], float(score))
            for index, score in enumerate(scores)
            if score > 0
        ]
        return _sort(ranked)[: self.config.keyword_top_k]

    def _semantic(self, text: str) -> list[tuple[str, float]]:
        vector = self._vectorizer.transform([text])
        query = self._np.asarray(vector.toarray(), dtype="float32")
        query = _normalize(query, self._np)
        count = min(self.config.semantic_top_k, len(self._doc_ids))
        scores, indices = self._index.search(query, count)
        ranked = [
            (self._doc_ids[int(index)], float(score))
            for index, score in zip(indices[0], scores[0])
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
        cross_scores = {node_id: _sigmoid(float(score)) for node_id, score in zip(node_ids, raw)}
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
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _rrf(
    keyword: Sequence[tuple[str, float]],
    semantic: Sequence[tuple[str, float]],
    fusion_k: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in (keyword, semantic):
        for rank, (node_id, _) in enumerate(ranked, 1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (fusion_k + rank)
    return scores


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _sort(rows: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(rows, key=lambda row: (-row[1], row[0]))
