import hashlib
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from synapticdb import EmbeddingError, InvalidArgumentError, NotFoundError
from synapticdb.store import EdgeSeed, Store


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "store.db") as opened:
        yield opened


def remember(
    store: Store,
    content: str,
    embedding: tuple[float, ...] = (1.0, 0.0),
    *,
    created_at: datetime | None = None,
) -> UUID:
    memory = store.insert_memory(content, {}, embedding, created_at=created_at)
    return memory.id


def test_schema_initializes_with_wal_fts_and_version(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"
    with Store(db_path):
        pass
    with sqlite3.connect(db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()
        objects = connection.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'",
        ).fetchall()
    names = {name for _, name in objects}
    assert journal_mode == ("wal",)
    assert version == (1,)
    assert {"meta", "memories", "memories_fts", "edges", "queries"} <= names
    assert {"memories_ai", "memories_ad", "memories_au"} <= names


def test_store_supports_memory_database_and_context_manager() -> None:
    with Store(":memory:") as store:
        memory_id = remember(store, "in memory")
        assert store.get_memory(memory_id).content == "in memory"
    with pytest.raises(RuntimeError, match="closed"):
        store.get_memory(memory_id)


def test_memory_roundtrip_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "roundtrip.db"
    created_at = datetime(2026, 7, 23, 12, 30, tzinfo=timezone.utc)
    with Store(db_path) as first:
        memory = first.insert_memory(
            "Client requirement",
            {"source": "call", "priority": 2},
            (0.25, 0.75),
            created_at=created_at,
        )
    with Store(db_path) as second:
        loaded = second.get_memory(memory.id)
        assert loaded == memory
        assert second.embedding_dimension() == 2


def test_content_hash_dedupe_returns_existing_memory(store: Store) -> None:
    first = store.insert_memory("  Fact\r\n", {"version": 1}, (1.0, 0.0))
    duplicate = store.insert_memory("Fact", {"version": 2}, (0.0, 1.0))
    assert duplicate == first
    assert duplicate.content == "  Fact\r\n"
    assert duplicate.metadata == {"version": 1}


def test_memory_and_initial_edges_commit_atomically(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    result = store.insert_memory_with_edges(
        "linked memory",
        {},
        (1.0, 0.0),
        (
            EdgeSeed(first_id, 0.25, "semantic"),
            EdgeSeed(second_id, 0.25, "semantic"),
        ),
    )
    assert result.inserted is True
    edges = store.list_edges_for_node(result.memory.id)
    assert len(edges) == 2
    assert {edge.origin for edge in edges} == {"semantic"}
    assert {edge.weight for edge in edges} == {0.25}


def test_deduplicated_memory_does_not_create_new_edges(store: Store) -> None:
    first_id = remember(store, "first")
    original = store.insert_memory("duplicate", {}, (1.0, 0.0))
    result = store.insert_memory_with_edges(
        " duplicate ",
        {},
        (1.0, 0.0),
        (EdgeSeed(first_id, 0.25, "semantic"),),
    )
    assert result.memory == original
    assert result.inserted is False
    assert store.graph_summary().edge_count == 0


def test_unknown_edge_seed_rolls_back_memory_insert(store: Store) -> None:
    remember(store, "existing")
    before = store.graph_summary()
    with pytest.raises(NotFoundError):
        store.insert_memory_with_edges(
            "must roll back",
            {},
            (1.0, 0.0),
            (EdgeSeed(uuid4(), 0.25, "semantic"),),
        )
    after = store.graph_summary()
    assert after == before
    assert store.keyword_search("rollback") == ()


@pytest.mark.parametrize(
    "embedding",
    [(), (0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0), (1e100, 1.0)],
)
def test_insert_rejects_invalid_embeddings(store: Store, embedding: tuple[float, ...]) -> None:
    with pytest.raises(EmbeddingError):
        store.insert_memory("fact", {}, embedding)


def test_insert_rejects_embedding_dimension_mismatch(store: Store) -> None:
    remember(store, "first", (1.0, 0.0))
    with pytest.raises(EmbeddingError, match="does not match"):
        remember(store, "second", (1.0, 0.0, 0.0))


def test_insert_rejects_invalid_content_and_metadata(store: Store) -> None:
    with pytest.raises(InvalidArgumentError, match="blank"):
        store.insert_memory(" \n ", {}, (1.0, 0.0))
    with pytest.raises(InvalidArgumentError, match="JSON-serializable"):
        store.insert_memory("fact", {"bad": object()}, (1.0, 0.0))
    with pytest.raises(InvalidArgumentError, match="keys must be strings"):
        store.insert_memory("fact", {cast(str, 1): "value"}, (1.0, 0.0))


def test_get_memories_preserves_order_and_rejects_unknown_ids(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    assert [memory.id for memory in store.get_memories((second_id, first_id))] == [second_id, first_id]
    with pytest.raises(NotFoundError):
        store.get_memories((first_id, uuid4()))


def test_recent_memory_ids_applies_timestamp_window_order_and_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "recent.db"
    started = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
    with Store(db_path) as first:
        boundary_id = remember(first, "boundary", created_at=started)
        second_id = remember(first, "second", created_at=started + timedelta(seconds=100))
        third_id = remember(first, "third", created_at=started + timedelta(seconds=200))
        remember(first, "future", created_at=started + timedelta(seconds=601))
    with Store(db_path) as reopened:
        recent = reopened.recent_memory_ids(
            before=started + timedelta(seconds=600),
            window_seconds=600,
            limit=2,
        )
        assert recent == (third_id, second_id)
        all_in_window = reopened.recent_memory_ids(
            before=started + timedelta(seconds=600),
            window_seconds=600,
            limit=3,
        )
        assert all_in_window == (third_id, second_id, boundary_id)
        with pytest.raises(InvalidArgumentError, match="bounded limits"):
            reopened.recent_memory_ids(before=started, window_seconds=86_401, limit=3)


def test_bump_access_updates_all_rows_atomically(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    accessed_at = datetime(2026, 7, 23, 13, 0, tzinfo=timezone.utc)
    store.bump_access((first_id, second_id), accessed_at=accessed_at)
    first, second = store.get_memories((first_id, second_id))
    assert (first.access_count, second.access_count) == (1, 1)
    assert (first.last_accessed_at, second.last_accessed_at) == (accessed_at, accessed_at)
    with pytest.raises(NotFoundError):
        store.bump_access((first_id, uuid4()))
    assert store.get_memory(first_id).access_count == 1


def test_insert_edge_is_canonical_and_conflict_returns_existing(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    created_at = datetime(2026, 7, 23, 14, 0, tzinfo=timezone.utc)
    edge = store.insert_edge(second_id, first_id, 0.25, "semantic", created_at=created_at)
    existing = store.insert_edge(first_id, second_id, 0.5, "explicit")
    a, b = sorted((str(first_id), str(second_id)))
    expected_id = hashlib.sha256(f"{a}{b}".encode("ascii")).hexdigest()
    assert edge.id == expected_id
    assert (str(edge.a), str(edge.b)) == (a, b)
    assert existing == edge
    assert existing.weight == 0.25
    assert existing.origin == "semantic"
    assert existing.last_reinforced_at == created_at
    assert existing.reinforcement_count == 0


def test_get_edge_between_normalizes_pair_order(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    assert store.get_edge_between(first_id, second_id) is None
    edge = store.insert_edge(first_id, second_id, 0.25, "semantic")
    assert store.get_edge_between(second_id, first_id) == edge


def test_edge_operations_validate_endpoints(store: Store) -> None:
    memory_id = remember(store, "first")
    with pytest.raises(InvalidArgumentError, match="different"):
        store.insert_edge(memory_id, memory_id, 0.5, "explicit")
    with pytest.raises(NotFoundError):
        store.insert_edge(memory_id, uuid4(), 0.5, "explicit")
    with pytest.raises(NotFoundError):
        store.list_edges_for_node(uuid4())


def test_list_edges_and_forget_cascade(store: Store) -> None:
    center_id = remember(store, "center")
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    first_edge = store.insert_edge(center_id, first_id, 0.25, "semantic")
    second_edge = store.insert_edge(center_id, second_id, 0.2, "temporal")
    assert {edge.id for edge in store.list_edges_for_node(center_id)} == {first_edge.id, second_edge.id}
    store.forget_memory(center_id)
    with pytest.raises(NotFoundError):
        store.get_edge(first_edge.id)
    with pytest.raises(NotFoundError):
        store.get_edge(second_edge.id)
    with pytest.raises(NotFoundError):
        store.get_memory(center_id)


def test_list_edges_on_a_dense_hub_returns_strongest_edges(store: Store) -> None:
    center_id = remember(store, "center")
    neighbor_ids = tuple(remember(store, f"neighbor {index}") for index in range(401))
    strong_edge = store.insert_edge(center_id, neighbor_ids[0], 0.9, "explicit")
    for neighbor_id in neighbor_ids[1:]:
        store.insert_edge(center_id, neighbor_id, 0.25, "semantic")

    edges = store.list_edges_for_node(center_id)
    assert len(edges) == 400
    assert edges[0] == strong_edge
    assert all(edges[index].weight >= edges[index + 1].weight for index in range(len(edges) - 1))


def test_fresh_edge_reads_its_stored_weight_undecayed(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    edge = store.insert_edge(first_id, second_id, 0.4, "semantic", created_at=created_at)
    assert edge.effective_weight == pytest.approx(edge.weight)

    aged = store.get_edge(edge.id, now=created_at + timedelta(days=60))
    assert aged.weight == 0.4
    assert aged.effective_weight == pytest.approx(0.1)


def test_hub_ranking_prefers_a_fresh_edge_over_a_stale_heavier_one(store: Store) -> None:
    center_id = remember(store, "center")
    stale_id = remember(store, "stale neighbor")
    fresh_id = remember(store, "fresh neighbor")
    long_ago = datetime(2026, 1, 1, tzinfo=timezone.utc)
    read_time = long_ago + timedelta(days=300)
    store.insert_edge(center_id, stale_id, 0.9, "explicit", created_at=long_ago)
    store.insert_edge(center_id, fresh_id, 0.3, "semantic", created_at=read_time)

    edges = store.list_edges_for_node(center_id, now=read_time)
    # 0.9 decayed over 300 days (10 half-lives) is far below a fresh 0.3, so
    # ordering by stored weight would put the dead edge first.
    neighbors = [edge.b if edge.a == center_id else edge.a for edge in edges]
    assert neighbors == [fresh_id, stale_id]
    assert edges[0].effective_weight > edges[1].effective_weight


def test_duplicate_seeds_reinforce_the_same_edge_once(store: Store) -> None:
    neighbor_id = remember(store, "neighbor")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = store.insert_memory_with_edges(
        "new memory",
        {},
        (1.0, 0.0),
        (
            EdgeSeed(neighbor_id, 0.25, "semantic", 0.05),
            EdgeSeed(neighbor_id, 0.2, "temporal", 0.05),
        ),
        created_at=created_at,
    )

    (edge,) = store.list_edges_for_node(result.memory.id, now=created_at)
    # The second seed reinforces rather than duplicating: 0.25 + 0.05 * 0.75.
    assert edge.weight == pytest.approx(0.2875)
    assert edge.reinforcement_count == 1


def test_reinforcement_decays_the_stored_weight_before_raising_it(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = created_at + timedelta(days=30)
    edge = store.insert_edge(first_id, second_id, 0.4, "semantic", created_at=created_at)

    # _write_edge is driven directly because every public caller today
    # reinforces an edge created in the same call, where no time has passed.
    # Phase 5.4 co-retrieval is the first mechanism to reinforce an aged edge.
    with store._connection:
        store._write_edge(first_id, EdgeSeed(second_id, 0.2, "temporal", 0.05), later, 30.0)

    reinforced = store.get_edge(edge.id, now=later)
    # One half-life takes the stored 0.4 to 0.2; reinforcement then adds
    # 0.05 * (1 - 0.2). Reinforcing the undecayed 0.4 would have given 0.43.
    assert reinforced.weight == pytest.approx(0.24)
    assert reinforced.last_reinforced_at == later
    assert reinforced.reinforcement_count == 1
    assert reinforced.effective_weight == pytest.approx(0.24)


def test_graph_summary_average_weight_falls_as_edges_age(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.insert_edge(first_id, second_id, 0.6, "explicit", created_at=created_at)

    fresh = store.graph_summary(now=created_at)
    aged = store.graph_summary(now=created_at + timedelta(days=30))
    assert fresh.average_edge_weight == pytest.approx(0.6)
    assert aged.average_edge_weight == pytest.approx(0.3)


def test_bulk_edge_update_rolls_back_on_unknown_edge(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    edge = store.insert_edge(first_id, second_id, 0.25, "semantic")
    with pytest.raises(NotFoundError):
        store.bulk_update_edge_weights(((edge.id, 0.8), ("missing", 0.6)))
    unchanged = store.get_edge(edge.id)
    assert unchanged.weight == 0.25
    assert unchanged.reinforcement_count == 0
    (updated,) = store.bulk_update_edge_weights(((edge.id, 0.8),))
    assert updated.weight == 0.8
    assert updated.reinforcement_count == 1


def test_graph_summary_reports_counts_averages_and_origins(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    third_id = remember(store, "third")
    semantic = store.insert_edge(first_id, second_id, 0.25, "semantic")
    store.insert_edge(second_id, third_id, 0.75, "explicit")
    store.bulk_update_edge_weights(((semantic.id, 0.5),))
    summary = store.graph_summary()
    assert summary.memory_count == 3
    assert summary.edge_count == 2
    assert summary.average_edge_weight == pytest.approx(0.625)
    assert summary.average_reinforcement_count == pytest.approx(0.5)
    assert summary.edges_by_origin() == {
        "semantic": 1,
        "temporal": 0,
        "co_retrieval": 0,
        "explicit": 1,
    }


def test_query_roundtrip_and_feedback(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    edge = store.insert_edge(first_id, second_id, 0.25, "semantic")
    created_at = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    query = store.save_query(
        "related facts",
        (first_id, second_id),
        {first_id: 1.0, second_id: 0.4},
        (edge.id,),
        created_at=created_at,
    )
    assert store.get_query(query.id) == query
    assert query.result_ids == (first_id, second_id)
    assert query.energies == {first_id: 1.0, second_id: 0.4}
    assert query.path_edge_ids == (edge.id,)
    assert query.feedback is None
    updated = store.set_query_feedback(query.id, 1)
    assert updated.feedback == 1
    with pytest.raises(InvalidArgumentError, match="already recorded"):
        store.set_query_feedback(query.id, -1)


def test_record_recall_persists_query_and_bumps_access_atomically(store: Store) -> None:
    first_id = remember(store, "first")
    second_id = remember(store, "second")
    recorded_at = datetime(2026, 7, 23, 15, 30, tzinfo=timezone.utc)
    query, memories = store.record_recall(
        "related facts",
        (second_id, first_id),
        {second_id: 1.0, first_id: 0.4},
        (),
        recorded_at=recorded_at,
    )
    assert query.result_ids == (second_id, first_id)
    assert [memory.id for memory in memories] == [second_id, first_id]
    assert all(memory.access_count == 1 for memory in memories)
    assert all(memory.last_accessed_at == recorded_at for memory in memories)


def test_record_recall_rolls_back_when_a_memory_is_unknown(store: Store) -> None:
    memory_id = remember(store, "first")
    unknown_id = uuid4()
    query_id = uuid4()
    with pytest.raises(NotFoundError):
        store.record_recall(
            "related facts",
            (memory_id, unknown_id),
            {memory_id: 1.0, unknown_id: 0.5},
            (),
            query_id=query_id,
        )
    with pytest.raises(NotFoundError):
        store.get_query(query_id)
    assert store.get_memory(memory_id).access_count == 0


def test_query_validation_rejects_incomplete_data(store: Store) -> None:
    memory_id = remember(store, "first")
    with pytest.raises(InvalidArgumentError, match="missing energy"):
        store.save_query("query", (memory_id,), {}, ())
    with pytest.raises(NotFoundError):
        store.get_query(uuid4())


def test_expire_queries_applies_age_rules_and_limit(store: Store) -> None:
    now = datetime(2026, 7, 23, 16, 0, tzinfo=timezone.utc)
    first_old = store.save_query("old pending one", (), {}, (), created_at=now - timedelta(days=8))
    second_old = store.save_query("old pending two", (), {}, (), created_at=now - timedelta(days=9))
    boundary = store.save_query("pending boundary", (), {}, (), created_at=now - timedelta(days=7))
    completed_old = store.save_query("old completed", (), {}, (), created_at=now - timedelta(days=31))
    completed_recent = store.save_query("recent completed", (), {}, (), created_at=now - timedelta(days=29))
    store.set_query_feedback(completed_old.id, 1)
    store.set_query_feedback(completed_recent.id, -1)
    assert store.expire_queries(now=now, limit=2) == 2
    assert store.expire_queries(now=now, limit=2) == 1
    for query_id in (first_old.id, second_old.id, completed_old.id):
        with pytest.raises(NotFoundError):
            store.get_query(query_id)
    assert store.get_query(boundary.id).id == boundary.id
    assert store.get_query(completed_recent.id).id == completed_recent.id
