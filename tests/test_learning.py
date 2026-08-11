from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from synapticdb import InvalidArgumentError
from synapticdb.activation import ActivationConfig
from synapticdb.learning import (
    SEMANTIC_SEED_CALIBRATION,
    CoRetrievalConfig,
    DecayConfig,
    FusionConfig,
    SemanticSeedConfig,
    TemporalLinkConfig,
    co_retrieval_pairs,
    decayed_weight,
    default_parameters,
    effective_weight,
    negative_feedback_weight,
    positive_feedback_seed,
    reinforce_weight,
    runtime_policy,
    semantic_seed_ids,
    unordered_pairs,
)


def test_defaults_define_exact_parameter_budget_and_semantic_group() -> None:
    params = default_parameters()
    assert params == {
        "top_k": 10,
        "candidate_depth": 40,
        "rrf_k": 60,
        "activation_seeds": 5,
        "activation_max_steps": 5,
        "activation_decay": 0.2,
        "activation_min_energy": 0.05,
        "hop_bonus": 0.15,
        "seed_penalty": 0.2,
        "activation_blend_weight": 0.45,
        "semantic_seed": None,
        "temporal_link": (600, 3, 0.2),
        "co_retrieval": (0.05, 0.05),
        "feedback_rate": 0.15,
        "connect_weight": 0.5,
        "decay_and_prune": (30, 0.02),
        "maintenance_interval": 100,
    }
    policy = runtime_policy()
    assert policy.semantic_seed is None
    assert policy.temporal_link == TemporalLinkConfig(600, 3, 0.2)
    assert policy.co_retrieval.reinforcement_rate == 0.05


def test_semantic_seed_calibration_reenables_the_mechanism() -> None:
    policy = runtime_policy({"semantic_seed": SEMANTIC_SEED_CALIBRATION})
    assert policy.semantic_seed == SemanticSeedConfig(0.6, 3, 0.25)


def test_runtime_policy_is_immutable() -> None:
    policy = runtime_policy()
    with pytest.raises(FrozenInstanceError):
        policy.feedback_rate = 0.2


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


def test_runtime_policy_reads_the_co_retrieval_weight_and_rate() -> None:
    assert runtime_policy().co_retrieval == CoRetrievalConfig(0.05, 0.05)


@pytest.mark.parametrize("group", [(0.05,), (0.05, 1.5), (-0.1, 0.05), 0.05, None])
def test_runtime_policy_rejects_a_malformed_co_retrieval_group(group: object) -> None:
    with pytest.raises(InvalidArgumentError):
        runtime_policy({"co_retrieval": group})


def test_co_retrieval_pairs_links_every_top_result_once() -> None:
    ids = tuple(uuid4() for _ in range(5))
    pairs = co_retrieval_pairs(ids)
    assert len(pairs) == 10
    assert len({frozenset(pair) for pair in pairs}) == 10
    assert all(first != second for first, second in pairs)


def test_co_retrieval_pairs_are_bounded_to_the_top_five_results() -> None:
    ids = tuple(uuid4() for _ in range(20))
    pairs = co_retrieval_pairs(ids)
    # The slice happens before pairing, so 20 results cannot yield 190 edges.
    assert len(pairs) == 10
    assert set(ids[5:]).isdisjoint({value for pair in pairs for value in pair})


def test_co_retrieval_pairs_are_deterministic() -> None:
    ids = tuple(uuid4() for _ in range(5))
    assert co_retrieval_pairs(ids) == co_retrieval_pairs(ids)


@pytest.mark.parametrize("count", [0, 1])
def test_co_retrieval_needs_two_results_to_link_anything(count: int) -> None:
    assert co_retrieval_pairs(tuple(uuid4() for _ in range(count))) == ()


def test_co_retrieval_pairs_reject_repeated_results() -> None:
    repeated = uuid4()
    with pytest.raises(InvalidArgumentError, match="unique"):
        co_retrieval_pairs((repeated, repeated))


def test_feedback_rate_reads_the_specified_value() -> None:
    assert runtime_policy().feedback_rate == 0.15


def test_connect_weight_reads_the_specified_value() -> None:
    assert runtime_policy().connect_weight == 0.5


def test_maintenance_interval_reads_the_specified_value() -> None:
    assert runtime_policy().maintenance_interval == 100


@pytest.mark.parametrize("value", [0, -1, None, (100,), 100_001])
def test_maintenance_interval_rejects_a_malformed_value(value: object) -> None:
    with pytest.raises(InvalidArgumentError):
        runtime_policy({"maintenance_interval": value})


def test_the_whole_parameter_budget_is_readable() -> None:
    policy = runtime_policy()
    assert policy.fusion == FusionConfig(40, 60)
    assert policy.activation_blend_weight == 0.45
    # The last group to be wired: every parameter PRD §9 promises the harness
    # can sweep is now actually read.
    assert policy.maintenance_interval == 100
    assert policy.activation == ActivationConfig(
        seeds=5, max_steps=5, decay=0.2, min_energy=0.05, hop_bonus=0.15, seed_penalty=0.2
    )


def test_runtime_policy_uses_the_activation_module_defaults() -> None:
    # The dataclass defaults are the locked constants, so reading the shipped
    # parameters must reproduce them exactly.
    assert runtime_policy().activation == ActivationConfig()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("candidate_depth", 101),  # retrieval._MAX_RANKING_LENGTH caps this
        ("candidate_depth", 0),
        ("rrf_k", 0),
        ("activation_seeds", 0),
        ("activation_max_steps", 21),
        ("activation_decay", 1.5),
        ("activation_min_energy", -0.1),
        ("hop_bonus", 1.5),
        ("seed_penalty", 1.5),
        ("activation_blend_weight", 1.5),
    ],
)
def test_retrieval_parameters_reject_out_of_range_values(key: str, value: object) -> None:
    with pytest.raises(InvalidArgumentError):
        runtime_policy({key: value})


@pytest.mark.parametrize(
    "key",
    [
        "candidate_depth",
        "rrf_k",
        "activation_seeds",
        "activation_max_steps",
        "activation_decay",
        "activation_min_energy",
        "hop_bonus",
        "seed_penalty",
        "activation_blend_weight",
    ],
)
def test_scalar_retrieval_parameters_reject_a_tuple(key: str) -> None:
    with pytest.raises(InvalidArgumentError, match="single number"):
        runtime_policy({key: (1, 2)})


@pytest.mark.parametrize("value", [None, (0.5, 0.5), 1.5, -0.1])
def test_connect_weight_rejects_a_malformed_value(value: object) -> None:
    with pytest.raises(InvalidArgumentError):
        runtime_policy({"connect_weight": value})


def test_positive_feedback_scales_by_the_energy_product() -> None:
    weight, rate = positive_feedback_seed(1.0, 1.0, 0.15)
    assert (weight, rate) == pytest.approx((0.05, 0.15))
    faint_weight, faint_rate = positive_feedback_seed(0.5, 0.4, 0.15)
    assert faint_rate == pytest.approx(0.15 * 0.2)
    # 0.05 * 0.2 = 0.01 would be born below the 0.02 prune threshold.
    assert faint_weight == pytest.approx(0.02)


def test_negative_feedback_shrinks_by_the_energy_product() -> None:
    assert negative_feedback_weight(0.4, 1.0, 1.0, 0.15) == pytest.approx(0.34)
    assert negative_feedback_weight(0.4, 0.5, 0.4, 0.15) == pytest.approx(0.4 * (1 - 0.03))


def test_negative_feedback_never_crosses_zero() -> None:
    weight = 1.0
    for _ in range(200):
        weight = negative_feedback_weight(weight, 1.0, 1.0, 0.15)
    assert 0.0 < weight < 1e-12


def test_feedback_arithmetic_rejects_out_of_range_energies() -> None:
    with pytest.raises(InvalidArgumentError, match="energy"):
        positive_feedback_seed(1.5, 1.0, 0.15)
    with pytest.raises(InvalidArgumentError, match="energy"):
        negative_feedback_weight(0.4, -0.1, 1.0, 0.15)


def test_unordered_pairs_covers_every_result_for_feedback() -> None:
    ids = tuple(uuid4() for _ in range(6))
    # Feedback pairs every result, unlike co-retrieval's top-five slice.
    assert len(unordered_pairs(ids)) == 15
    assert len(co_retrieval_pairs(ids)) == 10


def test_runtime_policy_reads_the_decay_half_life_and_threshold() -> None:
    assert runtime_policy().decay == DecayConfig(30, 0.02)


@pytest.mark.parametrize(
    "group",
    [(0, 0.02), (3651, 0.02), (30, 1.5), (30,), 30, None],
)
def test_runtime_policy_rejects_a_malformed_decay_group(group: object) -> None:
    with pytest.raises(InvalidArgumentError):
        runtime_policy({"decay_and_prune": group})


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
