from __future__ import annotations

import time

import numpy as np
import pytest

from src.config import PollaRules
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator
from src.simulation.monte_carlo import MonteCarloEngine
from src.simulation.participants import (
    ParticipantSimulator,
    SimulatedParticipant,
    StrategyMode,
)
from src.simulation.tournament import (
    SimulationConfig,
    SimulationReport,
    TournamentSimulator,
)


@pytest.fixture
def sample_pred() -> MatchPrediction:
    model = DixonColes(max_goals=7)
    return model.predict_from_params(lambda_h=1.5, mu_a=1.0)


@pytest.fixture
def match_predictions(sample_pred: MatchPrediction) -> dict[str, MatchPrediction]:
    return {f"KOmatch-{i}": sample_pred for i in range(104)}


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(num_simulations=100, seed=42, track_progress=False)


@pytest.fixture
def tournament_sim(config: SimulationConfig) -> TournamentSimulator:
    return TournamentSimulator(config=config)


@pytest.fixture
def participant_sim() -> ParticipantSimulator:
    return ParticipantSimulator(seed=42)


@pytest.fixture
def ep_calculator() -> ExpectedScoreCalculator:
    return ExpectedScoreCalculator()


@pytest.fixture
def engine(
    tournament_sim: TournamentSimulator,
    participant_sim: ParticipantSimulator,
    ep_calculator: ExpectedScoreCalculator,
    config: SimulationConfig,
) -> MonteCarloEngine:
    return MonteCarloEngine(
        tournament_sim=tournament_sim,
        participant_sim=participant_sim,
        ep_calculator=ep_calculator,
        config=config,
    )


def test_monte_carlo_engine_initialization(
    tournament_sim: TournamentSimulator,
    participant_sim: ParticipantSimulator,
    ep_calculator: ExpectedScoreCalculator,
    config: SimulationConfig,
) -> None:
    engine = MonteCarloEngine(
        tournament_sim=tournament_sim,
        participant_sim=participant_sim,
        ep_calculator=ep_calculator,
        config=config,
    )
    assert engine is not None


def test_run_full_simulation_basic(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    strategies = {"MVP Strategy": my_strat}
    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies=strategies,
        opponent_profiles=None,
        n_simulations=10,
    )
    assert "MVP Strategy" in reports
    report = reports["MVP Strategy"]
    assert isinstance(report, SimulationReport)
    assert report.n_simulations == 10


def test_run_full_simulation_with_opponents(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    opponents = [
        SimulatedParticipant(
            name="Opp1", strategy_mode=StrategyMode.RANDOM
        ),
        SimulatedParticipant(
            name="Opp2", strategy_mode=StrategyMode.CONSERVATIVE
        ),
    ]

    strategies = {"MyStrat": my_strat}
    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies=strategies,
        opponent_profiles=opponents,
        n_simulations=10,
    )
    assert "MyStrat" in reports
    report = reports["MyStrat"]
    assert report.n_simulations == 10
    assert 0.0 <= report.win_probability <= 1.0


def test_run_what_if_favorites(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
) -> None:
    report = engine.run_what_if(
        scenario="favorites",
        match_predictions=match_predictions,
        n_simulations=10,
    )
    assert isinstance(report, SimulationReport)
    assert report.strategy_name == "favorites"
    assert report.n_simulations == 10


def test_run_what_if_contrarian(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
) -> None:
    report = engine.run_what_if(
        scenario="contrarian_50",
        match_predictions=match_predictions,
        n_simulations=10,
    )
    assert isinstance(report, SimulationReport)
    assert report.strategy_name == "contrarian_50"


def test_run_what_if_underdogs(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
) -> None:
    report = engine.run_what_if(
        scenario="underdogs",
        match_predictions=match_predictions,
        n_simulations=10,
    )
    assert isinstance(report, SimulationReport)


def test_simulation_report_all_metrics(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"Test": my_strat},
        n_simulations=20,
    )
    report = reports["Test"]

    assert report.strategy_name == "Test"
    assert report.mean_points >= 0
    assert report.std_points >= 0
    assert report.median_points >= 0
    assert report.min_points >= 0
    assert report.max_points >= report.min_points
    assert 0.0 <= report.win_probability <= 1.0
    assert 0.0 <= report.top3_probability <= 1.0
    assert 0.0 <= report.last_probability <= 1.0
    assert 1.0 <= report.expected_rank <= 15.0
    assert 0.0 <= report.risk_of_ruin <= 1.0
    assert len(report.rank_distribution) > 0
    assert len(report.points_percentiles) > 0


def test_rank_distribution_sums_to_one(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"Test": my_strat},
        n_simulations=50,
    )
    report = reports["Test"]
    total = sum(report.rank_distribution.values())
    assert abs(total - 1.0) < 1e-6


def test_seed_reproducibility(
    tournament_sim: TournamentSimulator,
    participant_sim: ParticipantSimulator,
    ep_calculator: ExpectedScoreCalculator,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    config1 = SimulationConfig(num_simulations=5, seed=42, track_progress=False)
    engine1 = MonteCarloEngine(
        tournament_sim=TournamentSimulator(config=config1),
        participant_sim=ParticipantSimulator(seed=42),
        ep_calculator=ExpectedScoreCalculator(),
        config=config1,
    )
    reports1 = engine1.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"Test": my_strat},
        n_simulations=5,
    )

    config2 = SimulationConfig(num_simulations=5, seed=42, track_progress=False)
    engine2 = MonteCarloEngine(
        tournament_sim=TournamentSimulator(config=config2),
        participant_sim=ParticipantSimulator(seed=42),
        ep_calculator=ExpectedScoreCalculator(),
        config=config2,
    )
    reports2 = engine2.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"Test": my_strat},
        n_simulations=5,
    )

    assert reports1["Test"].mean_points == reports2["Test"].mean_points


def test_multiple_strategies_comparison(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def strat_a(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    def strat_b(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        alt = (sample_pred.most_likely_score[1], sample_pred.most_likely_score[0])
        return alt

    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"A": strat_a, "B": strat_b},
        n_simulations=10,
    )
    assert "A" in reports
    assert "B" in reports


def test_performance_under_threshold(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    start = time.perf_counter()
    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"Test": my_strat},
        n_simulations=10,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30


def test_empty_opponents_handled(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"Solo": my_strat},
        opponent_profiles=[],
        n_simulations=5,
    )
    assert "Solo" in reports


def test_win_probability_consistency(
    engine: MonteCarloEngine,
    match_predictions: dict[str, MatchPrediction],
    sample_pred: MatchPrediction,
) -> None:
    def my_strat(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        return sample_pred.most_likely_score

    reports = engine.run_full_simulation(
        match_predictions=match_predictions,
        my_strategies={"MyStrat": my_strat},
        n_simulations=50,
    )
    report = reports["MyStrat"]
    assert 0.0 <= report.win_probability <= 1.0
    assert report.expected_rank >= 1.0
