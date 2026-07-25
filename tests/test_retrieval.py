from collections.abc import Sequence
from typing import cast
from uuid import UUID, uuid4

import pytest

from synapticdb import InvalidArgumentError
from synapticdb.retrieval import blend_rankings, min_max_normalize, reciprocal_rank_fusion


def test_rrf_rewards_overlap_without_an_extra_bonus() -> None:
    first, overlap, last = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion(((first, overlap), (overlap, last)))
    assert [hit.memory_id for hit in fused] == [overlap, first, last]
    expected = 1 / 62 + 1 / 61
    assert fused[0].score == pytest.approx(expected)


def test_rrf_preserves_first_seen_order_for_equal_scores() -> None:
    first, second = uuid4(), uuid4()
    fused = reciprocal_rank_fusion(((first,), (second,)))
    assert [hit.memory_id for hit in fused] == [first, second]


def test_rrf_rejects_duplicate_ids_within_one_source() -> None:
    memory_id = uuid4()
    with pytest.raises(InvalidArgumentError, match="unique"):
        reciprocal_rank_fusion(((memory_id, memory_id),))


def test_rrf_rejects_non_uuid_ids_and_boolean_k() -> None:
    invalid_ranking = cast(Sequence[UUID], ("memory",))
    with pytest.raises(InvalidArgumentError, match="UUID"):
        reciprocal_rank_fusion((invalid_ranking,))
    with pytest.raises(InvalidArgumentError, match="positive"):
        reciprocal_rank_fusion(((uuid4(),),), k=True)


def test_min_max_normalize_handles_empty_equal_and_varied_scores() -> None:
    low, middle, high = uuid4(), uuid4(), uuid4()
    assert min_max_normalize({}) == {}
    assert min_max_normalize({low: 2.0}) == {low: 1.0}
    assert min_max_normalize({low: 2.0, high: 2.0}) == {low: 1.0, high: 1.0}
    normalized = min_max_normalize({low: 2.0, middle: 3.0, high: 4.0})
    assert normalized == {low: 0.0, middle: 0.5, high: 1.0}


def test_blend_at_zero_confidence_is_exact_fusion() -> None:
    first, second, association = uuid4(), uuid4(), uuid4()

    blended = blend_rankings(
        {first: 2.0, second: 1.0},
        {second: 0.5, association: 1.0},
        0.0,
    )

    assert [(hit.memory_id, hit.score, hit.via) for hit in blended] == [
        (first, 1.0, "search"),
        (second, 0.0, "search"),
    ]


def test_blend_combines_sources_and_attributes_membership() -> None:
    search, both, association = uuid4(), uuid4(), uuid4()

    blended = blend_rankings(
        {search: 2.0, both: 1.0},
        {both: 1.0, association: 2.0},
        1.0,
    )

    assert [(hit.memory_id, hit.score, hit.via) for hit in blended] == [
        (search, 0.55, "search"),
        (association, 0.45, "association"),
        (both, 0.0, "both"),
    ]


def test_seeds_are_excluded_from_the_discovery_scale() -> None:
    seed, near, far = uuid4(), uuid4(), uuid4()
    # The seed holds full energy, so leaving it in the distribution pins the
    # scale at 1.0 and compresses both discoveries toward zero.
    activation = {seed: 1.0, near: 0.2, far: 0.1}

    with_seed = {hit.memory_id: hit.score for hit in blend_rankings({}, activation, 1.0, 1.0)}
    without_seed = {hit.memory_id: hit.score for hit in blend_rankings({}, activation, 1.0, 1.0, seed_ids=(seed,))}

    assert with_seed[near] == pytest.approx(0.1111, abs=1e-3)
    # The strongest discovery now reaches the top of the scale.
    assert without_seed[near] == pytest.approx(1.0)
    assert without_seed[far] == pytest.approx(0.0)


def test_an_excluded_seed_keeps_its_search_attribution() -> None:
    seed, discovery = uuid4(), uuid4()

    blended = blend_rankings(
        {seed: 1.0},
        {seed: 1.0, discovery: 0.5},
        1.0,
        0.45,
        seed_ids=(seed,),
    )
    attribution = {hit.memory_id: hit.via for hit in blended}

    # A seed came from search; only a memory the graph reached on its own is
    # attributed to association.
    assert attribution[seed] == "search"
    assert attribution[discovery] == "association"


def test_a_lone_discovery_reaches_the_top_of_the_scale() -> None:
    seed, discovery = uuid4(), uuid4()

    blended = blend_rankings({seed: 1.0}, {seed: 1.0, discovery: 0.3}, 1.0, 1.0, seed_ids=(seed,))
    scores = {hit.memory_id: hit.score for hit in blended}

    # One discovery is an equal band, which min_max_normalize treats as fully
    # relevant, so at alpha 1.0 it ties the best search result.
    assert scores[discovery] == pytest.approx(1.0)


def test_blend_weight_scales_the_activation_share() -> None:
    search, both, association = uuid4(), uuid4(), uuid4()
    scores = ({search: 2.0, both: 1.0}, {both: 1.0, association: 2.0})

    # Alpha is blend_weight * confidence, so halving the weight at full
    # confidence gives the same ranking as half the confidence would.
    halved = blend_rankings(*scores, 1.0, 0.225)
    lower_confidence = blend_rankings(*scores, 0.5, 0.45)
    assert [(hit.memory_id, hit.score) for hit in halved] == pytest.approx(
        [(hit.memory_id, hit.score) for hit in lower_confidence]
    )


def test_blend_weight_of_zero_is_pure_fusion_even_at_full_confidence() -> None:
    first, second, association = uuid4(), uuid4(), uuid4()

    blended = blend_rankings(
        {first: 2.0, second: 1.0},
        {second: 0.5, association: 1.0},
        1.0,
        0.0,
    )

    # A swept weight of zero must also drop the association attribution, not
    # merely zero its contribution to the score.
    assert [(hit.memory_id, hit.score, hit.via) for hit in blended] == [
        (first, 1.0, "search"),
        (second, 0.0, "search"),
    ]


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_blend_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(InvalidArgumentError, match="graph confidence"):
        blend_rankings({}, {}, confidence)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_blend_rejects_an_invalid_weight(weight: float) -> None:
    with pytest.raises(InvalidArgumentError, match="blend weight"):
        blend_rankings({}, {}, 1.0, weight)
