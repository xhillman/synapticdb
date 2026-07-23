from collections.abc import Sequence
from typing import cast
from uuid import uuid4

import pytest

from synapticdb import InvalidArgumentError, NotFoundError, Synaptic
from synapticdb.retrieval import min_max_normalize, reciprocal_rank_fusion


def embedding(text: str) -> Sequence[float]:
    lowered = text.lower()
    return (
        float(lowered.count("alpha") + lowered.count("beta")),
        float(lowered.count("gamma") + lowered.count("delta")),
    )


def test_remember_recall_and_dedupe_form_a_walking_skeleton() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory.remember("alpha beta", {"source": "call"})
        duplicate = memory.remember(" alpha beta ", {"source": "other"})
        memory.remember("gamma delta", {"source": "email"})
        result = memory.recall("alpha", top_k=2)
        assert duplicate == first
        assert result.memories[0].memory.id == first.id
        assert result.memories[0].memory.access_count == 1
        assert all(recalled.via == "search" for recalled in result.memories)
        assert result.associative == []
        assert result.maturity == 0.0


def test_recall_alpha_zero_scores_equal_normalized_rrf() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory.remember("alpha alpha alpha alpha")
        second = memory.remember("alpha gamma")
        result = memory.recall("alpha", top_k=2)
        expected_fusion = reciprocal_rank_fusion(((first.id, second.id), (first.id, second.id)))
        expected = min_max_normalize({hit.memory_id: hit.score for hit in expected_fusion})
        assert [recalled.memory.id for recalled in result.memories] == [first.id, second.id]
        assert [recalled.score for recalled in result.memories] == pytest.approx(
            [expected[first.id], expected[second.id]]
        )


def test_recall_persists_unit_energies_for_fusion_only_results() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory.remember("alpha beta")
        second = memory.remember("alpha gamma")
        result = memory.recall("alpha", top_k=2)
        stored = memory._store.get_query(result.query_id)
        assert stored.energies == {first.id: 1.0, second.id: 1.0}


def test_where_filter_requires_present_equal_metadata() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        call = memory.remember("alpha call", {"source": "call", "optional": None})
        memory.remember("alpha email", {"source": "email"})
        matching = memory.recall("alpha", where={"source": "call"})
        missing = memory.recall("alpha", where={"missing": None})
        assert [item.memory.id for item in matching.memories] == [call.id]
        assert missing.memories == []


def test_empty_database_recall_is_persisted_and_returns_no_results() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        result = memory.recall("alpha")
        assert result.memories == []
        assert result.maturity == 0.0
        assert memory.stats().memories == 0


def test_forget_stats_and_context_manager() -> None:
    memory = Synaptic(":memory:", embedding_fn=embedding)
    with memory:
        remembered = memory.remember("alpha")
        assert memory.stats().memories == 1
        memory.forget(remembered.id)
        assert memory.stats().memories == 0
        with pytest.raises(NotFoundError):
            memory.forget(uuid4())
    with pytest.raises(RuntimeError, match="closed"):
        memory.stats()


@pytest.mark.parametrize("top_k", [0, 101, True])
def test_recall_rejects_invalid_top_k(top_k: object) -> None:
    with (
        Synaptic(":memory:", embedding_fn=embedding) as memory,
        pytest.raises(InvalidArgumentError, match="top_k"),
    ):
        memory.recall("alpha", top_k=cast(int, top_k))
