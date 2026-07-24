from uuid import uuid4

import pytest

from synapticdb import InvalidArgumentError
from synapticdb.learning import (
    SEMANTIC_SEED_CALIBRATION,
    SemanticSeedConfig,
    TemporalLinkConfig,
    default_parameters,
    passive_reinforcement_rate,
    reinforce_weight,
    semantic_seed_config,
    semantic_seed_ids,
    temporal_link_config,
)


def test_defaults_define_exact_parameter_budget_and_semantic_group() -> None:
    params = default_parameters()
    assert len(params) == 17
    # Semantic seeding ships disabled (benchmark evidence); it is still one of
    # the 17 parameter groups, just None-valued.
    assert semantic_seed_config(params) is None
    assert temporal_link_config(params) == TemporalLinkConfig(600, 3, 0.2)
    assert passive_reinforcement_rate(params) == 0.05


def test_semantic_seed_calibration_reenables_the_mechanism() -> None:
    params = default_parameters()
    params["semantic_seed"] = SEMANTIC_SEED_CALIBRATION
    assert semantic_seed_config(params) == SemanticSeedConfig(0.6, 3, 0.25)


def test_semantic_seed_selection_applies_threshold_and_preserves_order() -> None:
    first = uuid4()
    second = uuid4()
    third = uuid4()
    selected = semantic_seed_ids(
        ((first, 0.9), (second, 0.6), (third, -0.2)),
        SemanticSeedConfig(0.6, 3, 0.25),
    )
    assert selected == (first, second)


def test_semantic_seed_selection_rejects_unbounded_candidates() -> None:
    config = SemanticSeedConfig(0.6, 1, 0.25)
    with pytest.raises(InvalidArgumentError, match="configured maximum"):
        semantic_seed_ids(((uuid4(), 0.9), (uuid4(), 0.8)), config)


def test_passive_reinforcement_moves_weight_toward_one() -> None:
    assert reinforce_weight(0.25, 0.05) == pytest.approx(0.2875)
