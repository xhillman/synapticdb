import sqlite3
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest

from synapticdb import EmbeddingError, InvalidArgumentError
from synapticdb.store import Store


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "search.db"


@pytest.fixture
def store(store_path: Path) -> Iterator[Store]:
    with Store(store_path) as opened:
        yield opened


def store_memory(store: Store, content: str, embedding: tuple[float, ...]) -> UUID:
    return store.insert_memory(content, {}, embedding).id


def test_keyword_search_uses_bm25_ordering(store: Store) -> None:
    frequent_id = store_memory(store, "alpha alpha alpha alpha", (1.0, 0.0))
    sparse_id = store_memory(store, "alpha beta gamma delta epsilon zeta eta theta", (0.8, 0.2))
    store_memory(store, "beta only", (0.0, 1.0))
    hits = store.keyword_search("alpha")
    assert [hit.memory_id for hit in hits] == [frequent_id, sparse_id]
    assert hits[0].score > hits[1].score


def test_keyword_search_handles_plain_punctuation(store: Store) -> None:
    alpha_id = store_memory(store, "alpha record", (1.0, 0.0))
    beta_id = store_memory(store, "beta record", (0.0, 1.0))
    hits = store.keyword_search("alpha + (beta)?")
    assert {hit.memory_id for hit in hits} == {alpha_id, beta_id}
    assert store.keyword_search(" + ? ") == ()


def test_fts_stays_synced_after_insert_update_and_delete(store: Store, store_path: Path) -> None:
    memory_id = store_memory(store, "original token", (1.0, 0.0))
    assert store.keyword_search("original")[0].memory_id == memory_id
    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "UPDATE memories SET content = ? WHERE id = ?",
            ("replacement token", str(memory_id)),
        )
    assert store.keyword_search("original") == ()
    assert store.keyword_search("replacement")[0].memory_id == memory_id
    store.forget_memory(memory_id)
    assert store.keyword_search("replacement") == ()


def test_search_limits_must_be_positive(store: Store) -> None:
    store_memory(store, "alpha", (1.0, 0.0))
    with pytest.raises(InvalidArgumentError, match="positive"):
        store.keyword_search("alpha", limit=0)
    with pytest.raises(InvalidArgumentError, match="positive"):
        store.semantic_search((1.0, 0.0), limit=0)


def test_keyword_search_breaks_bm25_ties_by_insertion_order() -> None:
    contents = ("alpha one", "alpha two", "alpha three")

    def ranked_positions() -> list[int]:
        with Store(":memory:") as store:
            ids = [store.insert_memory(content, {}, (1.0, 0.0)).id for content in contents]
            return [ids.index(hit.memory_id) for hit in store.keyword_search("alpha")]

    assert ranked_positions() == [0, 1, 2]
    assert ranked_positions() == [0, 1, 2]


def test_keyword_search_truncates_very_long_queries(store: Store) -> None:
    needle_id = store_memory(store, "needle document", (1.0, 0.0))
    filler = " ".join(f"filler{index}" for index in range(80))
    hits = store.keyword_search(f"needle {filler}")
    assert needle_id in {hit.memory_id for hit in hits}


def test_semantic_search_orders_by_cosine_similarity(store: Store) -> None:
    exact_id = store_memory(store, "exact", (1.0, 0.0))
    near_id = store_memory(store, "near", (0.8, 0.2))
    orthogonal_id = store_memory(store, "orthogonal", (0.0, 1.0))
    hits = store.semantic_search((1.0, 0.0))
    assert [hit.memory_id for hit in hits] == [exact_id, near_id, orthogonal_id]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score > hits[2].score


def test_vector_cache_updates_after_insert_and_forget(store: Store) -> None:
    positive_id = store_memory(store, "positive", (1.0, 0.0))
    store.semantic_search((1.0, 0.0))
    negative_id = store_memory(store, "negative", (-1.0, 0.0))
    assert store.semantic_search((-1.0, 0.0))[0].memory_id == negative_id
    store.forget_memory(negative_id)
    remaining = store.semantic_search((-1.0, 0.0))
    assert [hit.memory_id for hit in remaining] == [positive_id]
    assert remaining[0].score == pytest.approx(-1.0)


def test_semantic_search_handles_empty_store() -> None:
    with Store(":memory:") as store:
        assert store.semantic_search((1.0, 0.0)) == ()


def test_semantic_search_rejects_invalid_query_vectors(store: Store) -> None:
    store_memory(store, "first", (1.0, 0.0))
    with pytest.raises(EmbeddingError, match="zero vector"):
        store.semantic_search((0.0, 0.0))
    with pytest.raises(EmbeddingError, match="does not match"):
        store.semantic_search((1.0, 0.0, 0.0))
