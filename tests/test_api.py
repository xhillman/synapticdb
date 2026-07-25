from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest

from synapticdb import InvalidArgumentError, NotFoundError, Synaptic
from synapticdb.learning import SEMANTIC_SEED_CALIBRATION
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
        second = memory.remember("alpha delta delta delta")
        result = memory.recall("alpha", top_k=2)
        expected_fusion = reciprocal_rank_fusion(((first.id, second.id), (first.id, second.id)))
        expected = min_max_normalize({hit.memory_id: hit.score for hit in expected_fusion})
        assert [recalled.memory.id for recalled in result.memories] == [first.id, second.id]
        assert [recalled.score for recalled in result.memories] == pytest.approx(
            [expected[first.id], expected[second.id]]
        )


def test_semantic_seeding_is_disabled_by_default() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha first", None, started)
        # Far outside the temporal window, so any edge would be semantic.
        second = memory._remember_at("alpha second", None, started + timedelta(seconds=3600))
        assert memory._store.get_edge_between(first.id, second.id) is None
        assert memory.stats().edges_by_origin["semantic"] == 0


def test_remember_seeds_top_three_semantic_edges() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["semantic_seed"] = SEMANTIC_SEED_CALIBRATION
        existing = tuple(
            memory._remember_at(
                f"alpha memory {index}",
                None,
                started + timedelta(seconds=601 * index),
            )
            for index in range(4)
        )
        newest = memory._remember_at("alpha newest", None, started + timedelta(seconds=601 * 4))
        edges = memory._store.list_edges_for_node(newest.id)
        assert len(edges) == 3
        assert {edge.a if edge.b == newest.id else edge.b for edge in edges} == {
            item.id for item in existing[:3]
        }
        assert all(edge.origin == "semantic" and edge.weight == 0.25 for edge in edges)


def test_private_semantic_parameters_support_benchmark_calibration() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["semantic_seed"] = (0.8, 1, 0.4)
        first = memory._remember_at("alpha first", None, started)
        second = memory._remember_at("alpha second", None, started + timedelta(seconds=601))
        edge = memory._store.get_edge_between(first.id, second.id)
        assert edge is not None
        assert edge.weight == 0.4


def test_remember_links_temporal_boundary_and_skips_outside_window() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha first", None, started)
        boundary = memory._remember_at("gamma boundary", None, started + timedelta(seconds=600))
        outside = memory._remember_at("alpha outside", None, started + timedelta(seconds=1201))
        edge = memory._store.get_edge_between(first.id, boundary.id)
        assert edge is not None
        assert edge.origin == "temporal"
        assert edge.weight == 0.2
        assert memory._store.get_edge_between(boundary.id, outside.id) is None


def test_semantic_and_temporal_overlap_reinforces_one_edge() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["semantic_seed"] = SEMANTIC_SEED_CALIBRATION
        first = memory._remember_at("alpha first", None, started)
        second = memory._remember_at("alpha second", None, started + timedelta(seconds=1))
        edge = memory._store.get_edge_between(first.id, second.id)
        assert edge is not None
        assert edge.origin == "semantic"
        assert edge.weight == pytest.approx(0.2875)
        assert edge.reinforcement_count == 1


def test_recall_persists_unit_energies_for_fusion_only_results() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha beta", None, started)
        second = memory._remember_at(
            "alpha delta delta delta",
            None,
            started + timedelta(seconds=601),
        )
        result = memory.recall("alpha", top_k=2)
        stored = memory._store.get_query(result.query_id)
        assert stored.energies == {first.id: 1.0, second.id: 1.0}


def test_recall_blends_association_and_persists_activation_evidence() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        anchor = memory.remember("alpha anchor")
        for index in range(40):
            memory.remember(f"gamma filler {index}")
        associated = memory.remember("delta hidden")
        edge = memory._store.insert_edge(anchor.id, associated.id, 1.0, "explicit")

        result = memory.recall("alpha", top_k=50)
        recalled = {item.memory.id: item for item in result.memories}
        stored = memory._store.get_query(result.query_id)

        assert result.maturity > 0.0
        assert recalled[associated.id].via == "association"
        assert recalled[associated.id].score > 0.0
        assert stored.energies[associated.id] == pytest.approx(0.8)
        assert edge.id in stored.path_edge_ids


def test_recall_links_its_top_results_and_reinforces_on_repeat() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        second = memory._remember_at("alpha two", None, started + timedelta(seconds=601))

        memory._recall_at("alpha", 10, None, started + timedelta(seconds=1200))
        edge = memory._store.get_edge_between(first.id, second.id)
        assert edge is not None
        assert edge.origin == "co_retrieval"
        assert edge.weight == pytest.approx(0.05)

        memory._recall_at("alpha", 10, None, started + timedelta(seconds=1300))
        reinforced = memory._store.get_edge_between(first.id, second.id)
        assert reinforced is not None
        # 0.05 + 0.05 * (1 - 0.05), less the decay accrued over the 100 seconds
        # between the two recalls: reinforcement always builds on the aged
        # weight, so the result sits just below a clean 0.0975.
        assert reinforced.weight == pytest.approx(0.0975, rel=1e-3)
        assert reinforced.weight < 0.0975
        assert reinforced.reinforcement_count == 1
        # stats() reads through the confidence cache; a stale cache here would
        # mean the co-retrieval write skipped its invalidation.
        assert memory.stats().edges_by_origin["co_retrieval"] == 1


def test_recall_below_two_results_learns_nothing() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._remember_at("alpha only", None, started)
        memory._recall_at("alpha", 10, None, started + timedelta(seconds=601))
        assert memory.stats().edges == 0


def test_a_new_co_retrieval_edge_carries_no_energy_until_reinforced() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        anchor = memory._remember_at("alpha anchor", None, started)
        far = memory._remember_at("delta far", None, started + timedelta(seconds=601))
        # A single 0.05 edge: activation would carry 1.0 * 0.05 * 0.8 = 0.04,
        # under the 0.05 minimum energy, so the pair cannot yet spread.
        edge = memory._store.insert_edge(anchor.id, far.id, 0.05, "co_retrieval")
        result = memory._recall_at("alpha", 10, None, started + timedelta(seconds=1200))
        assert edge.id not in memory._store.get_query(result.query_id).path_edge_ids

        # One reinforcement lifts it to 0.0975, above the 0.0625 an edge needs
        # to propagate a full-energy seed. This threshold is why co-retrieval
        # only matters for pairs that recur.
        memory._store.bulk_update_edge_weights(((edge.id, 0.0975),))
        after = memory._recall_at("alpha", 10, None, started + timedelta(seconds=1300))
        assert edge.id in memory._store.get_query(after.query_id).path_edge_ids


def test_activation_energy_falls_as_the_connecting_edge_ages() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        anchor = memory._remember_at("alpha anchor", None, started)
        for index in range(40):
            memory._remember_at(f"gamma filler {index}", None, started)
        associated = memory._remember_at("delta hidden", None, started)
        memory._store.insert_edge(anchor.id, associated.id, 1.0, "explicit", created_at=started)

        fresh = memory._recall_at("alpha", 50, None, started)
        aged = memory._recall_at("alpha", 50, None, started + timedelta(days=30))

        fresh_energy = memory._store.get_query(fresh.query_id).energies[associated.id]
        aged_energy = memory._store.get_query(aged.query_id).energies[associated.id]
        # A full seed spreads 1.0 * weight * (1 - 0.2). One half-life takes the
        # edge from 1.0 to 0.5, so the energy it carries halves with it.
        assert fresh_energy == pytest.approx(0.8)
        assert aged_energy == pytest.approx(0.4)


def test_recall_reads_every_edge_at_one_instant() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        second = memory._remember_at("alpha two", None, started)
        result = memory._recall_at("alpha", 2, None, started)
        stored = memory._store.get_query(result.query_id)
        assert stored.created_at == started
        assert set(stored.energies) == {first.id, second.id}


def _two_memory_recall(memory: Synaptic, started: datetime) -> tuple[UUID, UUID, UUID]:
    first = memory._remember_at("alpha one", None, started)
    second = memory._remember_at("alpha two", None, started + timedelta(seconds=601))
    result = memory._recall_at("alpha", 10, None, started + timedelta(seconds=1200))
    return first.id, second.id, result.query_id


def test_positive_feedback_reinforces_the_edge_between_results() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first_id, second_id, query_id = _two_memory_recall(memory, started)
        before = memory._store.get_edge_between(first_id, second_id)
        assert before is not None and before.weight == pytest.approx(0.05)

        memory._feedback_at(query_id, True, started + timedelta(seconds=1200))
        after = memory._store.get_edge_between(first_id, second_id)
        assert after is not None
        # Both results are fusion-only, so e_i = e_j = 1.0 and the update is
        # the full 0.15 * (1 - w) on top of the co-retrieval edge.
        assert after.weight == pytest.approx(0.05 + 0.15 * 0.95, rel=1e-3)
        assert after.reinforcement_count == before.reinforcement_count + 1


def test_negative_feedback_weakens_without_creating_edges() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first_id, second_id, query_id = _two_memory_recall(memory, started)
        before = memory._store.get_edge_between(first_id, second_id)
        assert before is not None
        edges_before = memory.stats().edges

        memory._feedback_at(query_id, False, started + timedelta(seconds=1200))
        after = memory._store.get_edge_between(first_id, second_id)
        assert after is not None
        assert after.weight == pytest.approx(before.weight * 0.85, rel=1e-3)
        assert memory.stats().edges == edges_before


def test_positive_feedback_creates_a_missing_result_pair_edge() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        second = memory._remember_at("alpha two", None, started + timedelta(seconds=601))
        # A saved query without the co-retrieval pass, so no edge exists yet.
        query = memory._store.save_query(
            "alpha",
            (first.id, second.id),
            {first.id: 1.0, second.id: 0.4},
            (),
        )
        assert memory._store.get_edge_between(first.id, second.id) is None

        memory._feedback_at(query.id, True, started + timedelta(seconds=1200))
        created = memory._store.get_edge_between(first.id, second.id)
        assert created is not None
        assert created.origin == "co_retrieval"
        assert created.weight == pytest.approx(0.05 * 0.4)


def test_repeated_feedback_raises_and_unknown_query_raises() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        _, _, query_id = _two_memory_recall(memory, started)
        memory.feedback(query_id, positive=True)
        with pytest.raises(InvalidArgumentError, match="already recorded"):
            memory.feedback(query_id, positive=True)
        with pytest.raises(NotFoundError):
            memory.feedback(uuid4(), positive=True)


def test_feedback_skips_pairs_whose_memory_was_forgotten() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first_id, second_id, query_id = _two_memory_recall(memory, started)
        memory.forget(second_id)
        # The query row still names the forgotten memory; feedback must not
        # raise, and must leave the survivor's own edges alone.
        memory.feedback(query_id, positive=True)
        assert memory._store.get_memory(first_id).id == first_id
        assert memory.stats().edges == 0


def test_recall_persists_energies_for_activated_non_results() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        anchor = memory._remember_at("alpha anchor", None, started)
        hidden = memory._remember_at("delta hidden", None, started + timedelta(seconds=601))
        memory._store.insert_edge(anchor.id, hidden.id, 1.0, "explicit")

        result = memory._recall_at("alpha", 1, None, started + timedelta(seconds=1200))
        stored = memory._store.get_query(result.query_id)
        # top_k=1 returns only the anchor, but activation energized `hidden`,
        # and feedback on the path edge needs both endpoint energies.
        assert [item.memory.id for item in result.memories] == [anchor.id]
        assert hidden.id in stored.energies
        assert stored.energies[hidden.id] == pytest.approx(0.8)


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
