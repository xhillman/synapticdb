from datetime import datetime, timezone
from uuid import uuid4

import pydantic
import pytest

from synapticdb import (
    EmbeddingError,
    InvalidArgumentError,
    Memory,
    NotFoundError,
    Recalled,
    RecallResult,
    Stats,
    SynapticError,
)
from synapticdb.models import RecallSource


def make_memory(content: str = "fact") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(id=uuid4(), content=content, created_at=now, last_accessed_at=now)


def make_recalled(via: RecallSource, score: float = 0.5, confidence: float = 0.5) -> Recalled:
    return Recalled(memory=make_memory(), score=score, confidence=confidence, via=via)


def test_recalled_separates_ranking_strength_from_confidence() -> None:
    # The two answer different questions, so they are allowed to disagree: an
    # association can rank well on graph evidence while matching the query text
    # poorly. Collapsing them into one number is what made `score` unusable.
    recalled = make_recalled("association", score=0.9, confidence=0.1)
    assert (recalled.score, recalled.confidence) == (0.9, 0.1)
    with pytest.raises(pydantic.ValidationError):
        make_recalled("search", confidence=1.5)


def test_memory_defaults() -> None:
    m = make_memory()
    assert m.metadata == {}
    assert m.access_count == 0


def test_memory_metadata_default_not_shared() -> None:
    a, b = make_memory("a"), make_memory("b")
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_memory_rejects_non_utc_timestamps() -> None:
    naive = datetime.now()
    with pytest.raises(pydantic.ValidationError, match="timestamp must use UTC"):
        Memory(id=uuid4(), content="fact", created_at=naive, last_accessed_at=naive)


def test_memory_rejects_negative_access_count() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(pydantic.ValidationError):
        Memory(id=uuid4(), content="fact", created_at=now, last_accessed_at=now, access_count=-1)


def test_recalled_rejects_unknown_via() -> None:
    with pytest.raises(pydantic.ValidationError):
        Recalled.model_validate({"memory": make_memory(), "score": 0.5, "via": "telepathy"})


@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_recalled_rejects_score_outside_unit_range(score: float) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_recalled("search", score)


def test_recall_result_associative_filters_by_via() -> None:
    ranked = [make_recalled("search"), make_recalled("association"), make_recalled("both")]
    result = RecallResult(query_id=uuid4(), memories=ranked, maturity=0.0, latency_ms=1.0)
    assert result.associative == [ranked[1]]
    assert result.memories == ranked


@pytest.mark.parametrize(
    ("maturity", "latency_ms"),
    [(-0.01, 1.0), (1.01, 1.0), (0.5, -0.01), (0.5, float("inf"))],
)
def test_recall_result_rejects_invalid_measurements(maturity: float, latency_ms: float) -> None:
    with pytest.raises(pydantic.ValidationError):
        RecallResult(query_id=uuid4(), memories=[], maturity=maturity, latency_ms=latency_ms)


def test_stats_accepts_valid_counts() -> None:
    stats = Stats(
        memories=2,
        edges=1,
        edges_by_origin={"semantic": 1, "temporal": 0, "co_retrieval": 0, "explicit": 0},
        maturity=0.5,
        db_path="agent.db",
    )
    assert stats.edges_by_origin["semantic"] == 1


@pytest.mark.parametrize(
    "values",
    [
        {"memories": -1, "edges": 0, "edges_by_origin": {}, "maturity": 0.5},
        {"memories": 0, "edges": -1, "edges_by_origin": {}, "maturity": 0.5},
        {"memories": 0, "edges": 0, "edges_by_origin": {"semantic": -1}, "maturity": 0.5},
        {"memories": 0, "edges": 0, "edges_by_origin": {"unknown": 1}, "maturity": 0.5},
        {"memories": 0, "edges": 0, "edges_by_origin": {}, "maturity": 1.01},
    ],
)
def test_stats_rejects_invalid_counts(values: dict[str, object]) -> None:
    with pytest.raises(pydantic.ValidationError):
        Stats.model_validate({**values, "db_path": "agent.db"})


def test_exception_hierarchy() -> None:
    for exc in (NotFoundError, InvalidArgumentError, EmbeddingError):
        assert issubclass(exc, SynapticError)
    assert issubclass(SynapticError, Exception)
