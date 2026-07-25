from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from synapticdb import InvalidArgumentError
from synapticdb.learning import (
    SEMANTIC_SEED_CALIBRATION,
    DecayConfig,
    SemanticSeedConfig,
    TemporalLinkConfig,
    decay_config,
    decayed_weight,
    default_parameters,
    effective_weight,
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


def test_decay_config_reads_the_specified_half_life_and_threshold() -> None:
    assert decay_config(default_parameters()) == DecayConfig(30, 0.02)


@pytest.mark.parametrize(
    "group",
    [(0, 0.02), (3651, 0.02), (30, 1.5), (30,), 30, None],
)
def test_decay_config_rejects_a_malformed_group(group: object) -> None:
    params = default_parameters()
    params["decay_and_prune"] = group  # type: ignore[assignment]
    with pytest.raises(InvalidArgumentError):
        decay_config(params)


@pytest.mark.parametrize(
    ("days_elapsed", "expected"),
    [(0.0, 0.8), (30.0, 0.4), (60.0, 0.2), (90.0, 0.1)],
)
def test_decay_halves_the_weight_every_half_life(days_elapsed: float, expected: float) -> None:
    assert decayed_weight(0.8, days_elapsed, 30.0) == pytest.approx(expected)


def test_decay_never_raises_a_weight_for_a_future_timestamp() -> None:
    # The benchmark ingests at a 2030 epoch, so edges legitimately carry
    # timestamps ahead of the read time. Amplifying them would push the weight
    # above 1.0 and make spreading activation reject the neighbor outright.
    assert decayed_weight(0.8, -1250.0, 30.0) == 0.8


def test_decay_is_monotonic_and_stays_within_the_unit_scale() -> None:
    weights = [decayed_weight(1.0, float(days), 30.0) for days in range(0, 400, 20)]
    assert weights == sorted(weights, reverse=True)
    assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_effective_weight_measures_the_span_between_two_timestamps() -> None:
    reinforced = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert effective_weight(0.6, reinforced, reinforced + timedelta(days=30), 30.0) == pytest.approx(0.3)


def test_effective_weight_requires_utc_timestamps() -> None:
    naive = datetime(2026, 1, 1)
    with pytest.raises(InvalidArgumentError, match="must use UTC"):
        effective_weight(0.6, naive, datetime.now(timezone.utc), 30.0)


@pytest.mark.parametrize("half_life", [0.0, -1.0, 4000.0])
def test_decay_rejects_an_unusable_half_life(half_life: float) -> None:
    with pytest.raises(InvalidArgumentError, match="half-life"):
        decayed_weight(0.5, 10.0, half_life)
