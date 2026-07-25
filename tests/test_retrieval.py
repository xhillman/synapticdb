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
