import inspect
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest

from synapticdb import InvalidArgumentError, NotFoundError, Synaptic
from synapticdb.learning import LINKED_RESULT_COUNT, SEMANTIC_SEED_CALIBRATION, default_parameters
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
        assert {edge.a if edge.b == newest.id else edge.b for edge in edges} == {item.id for item in existing[:3]}
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


def test_feedback_edge_creation_is_bounded_by_the_linked_result_count() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        # Ten results, each far enough apart that no temporal edge forms, so
        # every edge afterwards is one feedback created.
        ids = [
            memory._remember_at(f"alpha memory {index}", None, started + timedelta(seconds=3600 * index)).id
            for index in range(10)
        ]
        query = memory._store.save_query("alpha", tuple(ids), dict.fromkeys(ids, 1.0), ())
        assert memory.stats().edges == 0

        memory._feedback_at(query.id, True, started + timedelta(seconds=36_000))

        # PRD §6.6 as written pairs every result: C(10, 2) = 45 edges from one
        # call, scaling to 4,950 at the maximum top_k. Bounded to the top five
        # it is C(5, 2) = 10, matching co-retrieval.
        expected = LINKED_RESULT_COUNT * (LINKED_RESULT_COUNT - 1) // 2
        assert memory.stats().edges == expected
        # The edges are among the top results, not spread across the tail.
        assert memory._store.get_edge_between(ids[0], ids[4]) is not None
        assert memory._store.get_edge_between(ids[0], ids[9]) is None


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


def test_confidence_reports_similarity_not_rank_position() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        close = memory._remember_at("alpha alpha alpha", None, started)
        far = memory._remember_at("gamma gamma gamma", None, started + timedelta(seconds=3600))

        result = memory._recall_at("alpha", 10, None, started + timedelta(seconds=7200))
        by_id = {item.memory.id: item for item in result.memories}
        # The query is pure alpha: the alpha memory matches it exactly, the
        # gamma one is orthogonal. That is what confidence should say, whatever
        # the ranking does.
        assert by_id[close.id].confidence == pytest.approx(1.0)
        assert by_id[far.id].confidence == pytest.approx(0.0)


def test_confidence_does_not_drift_with_graph_state() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        anchor = memory._remember_at("alpha one", None, started)
        memory._remember_at("alpha two", None, started + timedelta(seconds=601))

        first = memory._recall_at("alpha", 10, None, started + timedelta(seconds=1200))
        # Several recalls later the graph has grown and maturity has moved. The
        # old `score` field tracked exactly that; confidence must not.
        for offset in range(1300, 1800, 100):
            memory._recall_at("alpha", 10, None, started + timedelta(seconds=offset))
        last = memory._recall_at("alpha", 10, None, started + timedelta(seconds=1900))

        before = {r.memory.id: r.confidence for r in first.memories}[anchor.id]
        after = {r.memory.id: r.confidence for r in last.memories}[anchor.id]
        assert before == pytest.approx(after)


def test_confidence_is_clamped_to_the_unit_scale() -> None:
    def signed(text: str) -> Sequence[float]:
        # Opposed vectors give a negative cosine, which must not surface as a
        # negative confidence: unrelated is 0.0, never anti-relevant.
        return (-1.0, 0.0) if "gamma" in text else (1.0, 0.0)

    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=signed) as memory:
        opposed = memory._remember_at("gamma opposite", None, started)
        result = memory._recall_at("alpha", 10, None, started + timedelta(seconds=3600))
        confidence = {r.memory.id: r.confidence for r in result.memories}[opposed.id]
        assert confidence == 0.0


def test_min_confidence_can_return_nothing_at_all() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._remember_at("gamma one", None, started)
        memory._remember_at("gamma two", None, started + timedelta(seconds=3600))

        # "alpha" is orthogonal to everything stored. Without a floor the
        # caller gets two confident-looking results for a question the corpus
        # cannot answer; with one it gets an honest empty list.
        unfiltered = memory._recall_at("alpha", 10, None, started + timedelta(seconds=7200))
        assert len(unfiltered.memories) == 2

        filtered = memory._recall_at("alpha", 10, None, started + timedelta(seconds=7200), 0.6)
        assert filtered.memories == []
        assert filtered.query_id is not None


def test_min_confidence_keeps_strong_matches_and_drops_weak_ones() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        strong = memory._remember_at("alpha alpha alpha", None, started)
        memory._remember_at("gamma gamma gamma", None, started + timedelta(seconds=3600))

        result = memory._recall_at("alpha", 10, None, started + timedelta(seconds=7200), 0.6)
        assert [item.memory.id for item in result.memories] == [strong.id]
        assert all(item.confidence >= 0.6 for item in result.memories)


def test_filtered_results_are_not_learned_from() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        second = memory._remember_at("gamma two", None, started + timedelta(seconds=3600))

        # Both would be returned unfiltered, and co-retrieval would link them.
        # A result the caller rejected as too weak must not also be reinforced
        # as though it were useful.
        memory._recall_at("alpha", 10, None, started + timedelta(seconds=7200), 0.6)
        assert memory._store.get_edge_between(first.id, second.id) is None


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), True, "high"])
def test_min_confidence_rejects_invalid_values(value: object) -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory.remember("alpha")
        with pytest.raises(InvalidArgumentError, match="min_confidence"):
            memory.recall("alpha", min_confidence=cast(float, value))


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


def test_connect_creates_the_only_user_authored_edge() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        # Outside the 600 s temporal window, so nothing links these two yet.
        second = memory._remember_at("gamma two", None, started + timedelta(seconds=3600))
        assert memory._store.get_edge_between(first.id, second.id) is None

        memory._connect_at(first.id, second.id, started + timedelta(seconds=3600))
        edge = memory._store.get_edge_between(first.id, second.id)
        assert edge is not None
        assert edge.origin == "explicit"
        assert edge.weight == pytest.approx(0.5)
        # Reading through stats() proves the confidence cache was invalidated.
        assert memory.stats().edges_by_origin["explicit"] == 1


def test_connect_claims_an_edge_the_graph_inferred() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        # Inside the window, so temporal proximity already linked them at 0.2.
        second = memory._remember_at("gamma two", None, started + timedelta(seconds=60))
        inferred = memory._store.get_edge_between(first.id, second.id)
        assert inferred is not None and inferred.origin == "temporal"

        memory._connect_at(first.id, second.id, started + timedelta(seconds=60))
        claimed = memory._store.get_edge_between(first.id, second.id)
        assert claimed is not None
        assert claimed.origin == "explicit"
        assert claimed.weight == pytest.approx(0.5)
        assert memory.stats().edges_by_origin == {
            "semantic": 0,
            "temporal": 0,
            "co_retrieval": 0,
            "explicit": 1,
        }


def test_connect_is_idempotent() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory._remember_at("alpha one", None, started)
        second = memory._remember_at("gamma two", None, started + timedelta(seconds=3600))
        memory._connect_at(first.id, second.id, started + timedelta(seconds=3600))
        memory._connect_at(first.id, second.id, started + timedelta(seconds=3600))

        edge = memory._store.get_edge_between(first.id, second.id)
        assert edge is not None
        assert edge.weight == pytest.approx(0.5)
        # Repeating an assertion must not accumulate confidence evidence.
        assert edge.reinforcement_count == 0
        assert memory.stats().edges == 1


def test_connect_rejects_non_uuid_arguments() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory.remember("alpha one")
        with pytest.raises(InvalidArgumentError, match="two UUID"):
            memory.connect(first.id, cast(UUID, "not-a-uuid"))


def test_forget_removes_a_connected_memory_and_its_edge() -> None:
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        first = memory.remember("alpha one")
        second = memory.remember("gamma two")
        memory.connect(first.id, second.id)
        assert memory.stats().edges == 1

        memory.forget(second.id)
        after = memory.stats()
        assert after.memories == 1
        assert after.edges == 0
        assert memory._store.list_edges_for_node(first.id) == ()


def test_maintenance_fires_on_the_interval_and_not_before() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["maintenance_interval"] = 4
        anchor = memory._remember_at("alpha anchor", None, started)
        stale = memory._remember_at("gamma stale", None, started)
        memory._connect_at(anchor.id, stale.id, started)
        # An explicit 0.5 edge is worth ~0.0004 after 300 days, well under the
        # 0.02 floor: exactly what maintenance is meant to collect.
        assert memory.stats().edges == 1

        much_later = started + timedelta(days=300)
        memory._remember_at("delta third", None, much_later)
        # Three inserts so far; the interval has not come round.
        assert memory._store.get_edge_between(anchor.id, stale.id) is not None

        memory._remember_at("delta fourth", None, much_later)
        assert memory._store.get_edge_between(anchor.id, stale.id) is None


def test_maintenance_keeps_edges_decay_has_not_worn_out() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["maintenance_interval"] = 2
        anchor = memory._remember_at("alpha anchor", None, started)
        fresh = memory._remember_at("gamma fresh", None, started)
        memory._connect_at(anchor.id, fresh.id, started)

        # Same instant, so nothing has aged: a maintenance pass must not touch
        # a healthy graph.
        assert memory.stats().edges == 1
        assert memory._store.get_edge_between(anchor.id, fresh.id) is not None


def test_maintenance_reports_the_smaller_graph_through_stats() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["maintenance_interval"] = 4
        first = memory._remember_at("alpha one", None, started)
        second = memory._remember_at("gamma two", None, started)
        memory._connect_at(first.id, second.id, started)
        assert memory.stats().edges == 1

        much_later = started + timedelta(days=300)
        memory._remember_at("delta three", None, much_later)
        memory._remember_at("delta four", None, much_later)
        # stats() reads through the confidence cache, so a stale cache here
        # would mean the prune skipped its invalidation. The worn-out explicit
        # edge is gone; the temporal edge those two inserts just created is
        # fresh, so it correctly survives its own maintenance pass.
        after = memory.stats()
        assert after.edges_by_origin["explicit"] == 0
        assert after.edges_by_origin["temporal"] == 1


def test_a_deduplicated_remember_does_not_advance_maintenance() -> None:
    started = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    with Synaptic(":memory:", embedding_fn=embedding) as memory:
        memory._params["maintenance_interval"] = 2
        anchor = memory._remember_at("alpha anchor", None, started)
        stale = memory._remember_at("gamma stale", None, started)
        memory._connect_at(anchor.id, stale.id, started)

        much_later = started + timedelta(days=300)
        # Re-remembering existing content stores nothing, so it grew no graph
        # and must not count toward the interval.
        memory._remember_at("alpha anchor", None, much_later)
        memory._remember_at("gamma stale", None, much_later)
        assert memory._store.get_edge_between(anchor.id, stale.id) is not None


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


def test_constructor_exposes_no_tuning_parameters() -> None:
    """PRD section 9: the 17 parameter groups stay private in v0.

    A knob added to the constructor is a public commitment that v0 does not
    intend to make. This fails the moment one leaks out, rather than at the
    point someone depends on it.
    """
    signature = inspect.signature(Synaptic.__init__)
    assert list(signature.parameters) == ["self", "db_path", "embedding_fn"]
    tunable = set(default_parameters()) - {"top_k"}
    assert tunable.isdisjoint(signature.parameters)


@pytest.mark.parametrize(
    "name",
    ["remember", "recall", "feedback", "connect", "forget", "stats", "close", "__enter__", "__exit__"],
)
def test_public_methods_are_documented(name: str) -> None:
    assert inspect.getdoc(getattr(Synaptic, name))
