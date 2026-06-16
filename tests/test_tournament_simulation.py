from __future__ import annotations

import time

import pytest

from src.config import PollaRules
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.simulation.tournament import (
    SimulationConfig,
    SimulationReport,
    TournamentSimulator,
    _calculate_points,
)


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(num_simulations=100, seed=42, track_progress=False)


@pytest.fixture
def sample_predictions() -> dict[str, MatchPrediction]:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.5, mu_a=1.0)
    result: dict[str, MatchPrediction] = {}
    for i in range(200):
        result[f"match-{i}"] = pred
    return result


@pytest.fixture
def simulator(config: SimulationConfig) -> TournamentSimulator:
    return TournamentSimulator(config=config)


@pytest.fixture
def simulator_with_preds(
    config: SimulationConfig, sample_predictions: dict[str, MatchPrediction]
) -> TournamentSimulator:
    sim = TournamentSimulator(config=config)
    sim.set_match_predictions(sample_predictions)
    return sim


def test_simulation_config_defaults() -> None:
    cfg = SimulationConfig()
    assert cfg.num_simulations == 10000
    assert cfg.max_goals == 7
    assert cfg.seed == 42
    assert cfg.n_groups == 12
    assert cfg.teams_per_group == 4
    assert cfg.n_third_place_qualify == 8


def test_simulator_initialization(config: SimulationConfig) -> None:
    sim = TournamentSimulator(config=config)
    assert sim.total_matches > 0
    assert len(sim.match_ids) == sim.total_matches


def test_simulator_total_matches(config: SimulationConfig) -> None:
    sim = TournamentSimulator(config=config)
    expected_group = config.n_groups * 6
    expected_ko = 16 + 8 + 4 + 2 + 1 + 1
    assert sim.total_matches == expected_group + expected_ko


def test_simulate_tournament_returns_result(simulator_with_preds: TournamentSimulator) -> None:
    result = simulator_with_preds.simulate_tournament()
    assert isinstance(result.match_results, list)
    assert len(result.match_results) > 0
    assert len(result.group_standings) == 12
    assert isinstance(result.knockout_results, list)


def test_simulate_tournament_match_count(simulator_with_preds: TournamentSimulator) -> None:
    result = simulator_with_preds.simulate_tournament()
    expected = simulator_with_preds.total_matches
    assert len(result.match_results) == expected


def test_simulate_tournament_scores_in_range(simulator_with_preds: TournamentSimulator) -> None:
    result = simulator_with_preds.simulate_tournament()
    for h, a in result.match_results:
        assert 0 <= h <= 7
        assert 0 <= a <= 7


def test_simulate_n_tournaments(simulator_with_preds: TournamentSimulator) -> None:
    results = simulator_with_preds.simulate_n_tournaments(10)
    assert len(results) == 10
    for r in results:
        assert len(r.match_results) > 0


def test_seed_reproducibility(config: SimulationConfig) -> None:
    pred_model = DixonColes(max_goals=7)
    pred = pred_model.predict_from_params(lambda_h=1.5, mu_a=1.0)
    preds: dict[str, MatchPrediction] = {}

    sim1 = TournamentSimulator(config=SimulationConfig(num_simulations=1, seed=42))
    for i in range(200):
        preds[f"match-{i}"] = pred
    sim1.set_match_predictions(preds)
    r1 = sim1.simulate_tournament()

    sim2 = TournamentSimulator(config=SimulationConfig(num_simulations=1, seed=42))
    sim2.set_match_predictions(preds)
    r2 = sim2.simulate_tournament()

    assert r1.match_results == r2.match_results


def test_win_probability_range() -> None:
    my_points = [10.0, 15.0, 12.0, 8.0, 14.0]
    opp_points = [[9.0, 13.0, 10.0, 7.0, 12.0]]
    prob = TournamentSimulator.calculate_win_probability(my_points, opp_points)
    assert 0.0 <= prob <= 1.0


def test_win_probability_matches_expected() -> None:
    my_points = [20.0, 20.0, 20.0, 20.0, 20.0]
    opp_points = [[10.0, 10.0, 10.0, 10.0, 10.0]]
    prob = TournamentSimulator.calculate_win_probability(my_points, opp_points)
    assert prob == 1.0

    my_points = [10.0, 10.0, 10.0, 10.0, 10.0]
    opp_points = [[20.0, 20.0, 20.0, 20.0, 20.0]]
    prob = TournamentSimulator.calculate_win_probability(my_points, opp_points)
    assert prob == 0.0


def test_evaluate_strategy_returns_report(
    simulator_with_preds: TournamentSimulator,
) -> None:
    tournament_results = simulator_with_preds.simulate_n_tournaments(20)

    def my_strategy(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        if pred is not None:
            return pred.most_likely_score
        return (1, 0)

    report = simulator_with_preds.evaluate_strategy(
        strategy_name="test_strategy",
        strategy_fn=my_strategy,
        tournament_results=tournament_results,
    )
    assert isinstance(report, SimulationReport)
    assert report.strategy_name == "test_strategy"
    assert report.n_simulations == 20


def test_evaluate_strategy_metrics_valid(
    simulator_with_preds: TournamentSimulator,
) -> None:
    tournament_results = simulator_with_preds.simulate_n_tournaments(20)

    def my_strategy(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        if pred is not None:
            return pred.most_likely_score
        return (1, 0)

    report = simulator_with_preds.evaluate_strategy(
        strategy_name="test_strategy",
        strategy_fn=my_strategy,
        tournament_results=tournament_results,
    )

    assert 0.0 <= report.win_probability <= 1.0
    assert 0.0 <= report.top3_probability <= 1.0
    assert 0.0 <= report.last_probability <= 1.0
    assert 1.0 <= report.expected_rank <= 15.0
    assert report.risk_of_ruin >= 0.0
    assert report.mean_points > 0
    assert report.std_points >= 0
    assert report.min_points <= report.median_points <= report.max_points


def test_rank_distribution_sums_to_one(
    simulator_with_preds: TournamentSimulator,
) -> None:
    tournament_results = simulator_with_preds.simulate_n_tournaments(20)

    def my_strategy(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        if pred is not None:
            return pred.most_likely_score
        return (1, 0)

    report = simulator_with_preds.evaluate_strategy(
        strategy_name="test_strategy",
        strategy_fn=my_strategy,
        tournament_results=tournament_results,
    )

    total_prob = sum(report.rank_distribution.values())
    assert abs(total_prob - 1.0) < 1e-6


def test_compare_strategies(
    simulator_with_preds: TournamentSimulator,
) -> None:
    def strat_a(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        if pred is not None:
            return pred.most_likely_score
        return (1, 0)

    def strat_b(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
        if pred is not None:
            alt = (pred.most_likely_score[1], pred.most_likely_score[0])
            return alt
        return (0, 1)

    reports = simulator_with_preds.compare_strategies(
        strategies={"Strategy A": strat_a, "Strategy B": strat_b},
        n_simulations=10,
    )

    assert "Strategy A" in reports
    assert "Strategy B" in reports
    assert reports["Strategy A"].strategy_name == "Strategy A"
    assert reports["Strategy B"].strategy_name == "Strategy B"


def test_performance_10_tournaments(
    simulator_with_preds: TournamentSimulator,
) -> None:
    start = time.perf_counter()
    results = simulator_with_preds.simulate_n_tournaments(10)
    elapsed = time.perf_counter() - start
    assert len(results) == 10
    assert elapsed < 10


def test_calculate_points_exact_score() -> None:
    rules = PollaRules(exact_score_pts=5)
    pts = _calculate_points((2, 1), (2, 1), rules)
    assert pts == 7.0


def test_calculate_points_result_only() -> None:
    rules = PollaRules(result_correct_pts=2)
    pts = _calculate_points((2, 1), (3, 0), rules)
    assert pts == 2.0


def test_calculate_points_wrong() -> None:
    rules = PollaRules()
    pts = _calculate_points((2, 1), (0, 3), rules)
    assert pts == 0.0


def test_calculate_points_draw_correct() -> None:
    rules = PollaRules(result_correct_pts=2)
    pts = _calculate_points((1, 1), (2, 2), rules)
    assert pts == 2.0


def test_set_group_teams(config: SimulationConfig) -> None:
    sim = TournamentSimulator(config=config)
    new_groups = [
        ["ARG", "IRN", "JPN", "SUI"],
        ["BRA", "SRB", "CMR", "KSA"],
        ["FRA", "DEN", "TUN", "PER"],
        ["ENG", "USA", "WAL", "QAT"],
        ["ESP", "CRC", "GER", "AUS"],
        ["POR", "URU", "KOR", "GHA"],
        ["NED", "SEN", "ECU", "CAN"],
        ["BEL", "CRO", "MAR", "NZL"],
        ["ITA", "COL", "EGY", "IRQ"],
        ["MEX", "POL", "NGA", "CHN"],
        ["CHI", "CIV", "ALG", "PAN"],
        ["SWE", "PER", "HON", "BFA"],
    ]
    sim.set_group_teams(new_groups)
    assert sim.total_matches > 0
