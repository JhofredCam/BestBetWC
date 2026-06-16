from __future__ import annotations

import numpy as np
import pytest

from src.models.dixon_coles import DixonColes, MatchPrediction
from src.simulation.participants import (
    ParticipantSimulator,
    SimulatedParticipant,
    StrategyMode,
)


@pytest.fixture
def sample_pred() -> MatchPrediction:
    model = DixonColes(max_goals=7)
    return model.predict_from_params(lambda_h=1.5, mu_a=1.0)


@pytest.fixture
def sample_predictions(sample_pred: MatchPrediction) -> dict[str, MatchPrediction]:
    return {f"match-{i}": sample_pred for i in range(10)}


@pytest.fixture
def simulator() -> ParticipantSimulator:
    return ParticipantSimulator(seed=42)


def test_strategy_mode_values() -> None:
    assert StrategyMode.CONSERVATIVE.value == "conservative"
    assert StrategyMode.AGGRESSIVE.value == "aggressive"
    assert StrategyMode.MARKET_FOLLOWER.value == "market_follower"
    assert StrategyMode.RANDOM.value == "random"
    assert StrategyMode.PERFECT.value == "perfect"


def test_simulated_participant_defaults() -> None:
    p = SimulatedParticipant(name="TestPlayer")
    assert p.name == "TestPlayer"
    assert p.strategy_mode == StrategyMode.MARKET_FOLLOWER


def test_simulated_participant_custom() -> None:
    p = SimulatedParticipant(
        name="Aggro", strategy_mode=StrategyMode.AGGRESSIVE
    )
    assert p.strategy_mode == StrategyMode.AGGRESSIVE


def test_conservative_strategy(sample_pred: MatchPrediction) -> None:
    result = ParticipantSimulator.conservative_strategy(sample_pred)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert 0 <= result[0] <= 7
    assert 0 <= result[1] <= 7


def test_aggressive_strategy(sample_pred: MatchPrediction) -> None:
    result = ParticipantSimulator.aggressive_strategy(sample_pred)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert 0 <= result[0] <= 7
    assert 0 <= result[1] <= 7


def test_market_follower_strategy(sample_pred: MatchPrediction) -> None:
    result = ParticipantSimulator.market_follower_strategy(sample_pred)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert 0 <= result[0] <= 7
    assert 0 <= result[1] <= 7


def test_random_strategy(sample_pred: MatchPrediction) -> None:
    result = ParticipantSimulator.random_strategy(sample_pred)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert 0 <= result[0] <= 7
    assert 0 <= result[1] <= 7


def test_simulate_predictions_conservative(
    simulator: ParticipantSimulator,
    sample_predictions: dict[str, MatchPrediction],
) -> None:
    p = SimulatedParticipant(
        name="Conservative", strategy_mode=StrategyMode.CONSERVATIVE
    )
    preds = simulator.simulate_predictions(p, sample_predictions)
    assert len(preds) == len(sample_predictions)
    for mid, val in preds.items():
        assert isinstance(val, tuple)
        assert len(val) == 2


def test_simulate_predictions_aggressive(
    simulator: ParticipantSimulator,
    sample_predictions: dict[str, MatchPrediction],
) -> None:
    p = SimulatedParticipant(
        name="Aggressive", strategy_mode=StrategyMode.AGGRESSIVE
    )
    preds = simulator.simulate_predictions(p, sample_predictions)
    assert len(preds) == len(sample_predictions)


def test_simulate_predictions_random(
    simulator: ParticipantSimulator,
    sample_predictions: dict[str, MatchPrediction],
) -> None:
    p = SimulatedParticipant(
        name="Random", strategy_mode=StrategyMode.RANDOM
    )
    preds = simulator.simulate_predictions(p, sample_predictions)
    assert len(preds) == len(sample_predictions)


def test_simulate_predictions_perfect(
    simulator: ParticipantSimulator,
    sample_predictions: dict[str, MatchPrediction],
) -> None:
    p = SimulatedParticipant(
        name="Perfect", strategy_mode=StrategyMode.PERFECT
    )
    preds = simulator.simulate_predictions(p, sample_predictions)
    assert len(preds) == len(sample_predictions)
    for mid, val in preds.items():
        assert val == sample_predictions[mid].most_likely_score


def test_simulate_all(
    simulator: ParticipantSimulator,
    sample_predictions: dict[str, MatchPrediction],
) -> None:
    participants = [
        SimulatedParticipant(name="P1", strategy_mode=StrategyMode.CONSERVATIVE),
        SimulatedParticipant(name="P2", strategy_mode=StrategyMode.AGGRESSIVE),
        SimulatedParticipant(name="P3", strategy_mode=StrategyMode.RANDOM),
    ]
    result = simulator.simulate_all(participants, sample_predictions)
    assert "P1" in result
    assert "P2" in result
    assert "P3" in result
    assert len(result["P1"]) == len(sample_predictions)


def test_conservative_vs_aggressive(
    sample_pred: MatchPrediction,
) -> None:
    cons = ParticipantSimulator.conservative_strategy(sample_pred)
    aggr = ParticipantSimulator.aggressive_strategy(sample_pred)
    assert isinstance(cons, tuple)
    assert isinstance(aggr, tuple)


def test_market_follower_deterministic_with_seed(
    sample_pred: MatchPrediction,
) -> None:
    a = ParticipantSimulator.market_follower_strategy(sample_pred)
    b = ParticipantSimulator.market_follower_strategy(sample_pred)
    assert a == b


def test_simulated_participant_with_profile() -> None:
    profile = type("FakeProfile", (), {"dominant_archetype": "CONSERVATIVE"})()
    p = SimulatedParticipant(
        name="ProfiledPlayer",
        profile=profile,
        strategy_mode=StrategyMode.CONSERVATIVE,
    )
    assert p.profile is not None
    assert p.profile.dominant_archetype == "CONSERVATIVE"
