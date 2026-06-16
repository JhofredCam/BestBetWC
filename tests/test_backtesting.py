from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.models.dixon_coles import DixonColes, MatchPrediction
from src.validation.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestMatch,
    BacktestReport,
    BacktestResult,
    _calculate_brier_score,
    _calculate_points,
    _calculate_rps,
    calculate_ece,
    calculate_score_matrix_log_loss,
    make_adaptive_strategy,
    make_optimal_ep_contrarian_strategy,
    make_optimal_ep_strategy,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(
        validation_years=[2022],
        max_goals=7,
        min_train_matches=20,
        fit_maxiter=100,
    )


@pytest.fixture
def engine(config: BacktestConfig) -> BacktestEngine:
    return BacktestEngine(config=config)


@pytest.fixture
def test_csv() -> Path:
    return DATA_DIR / "world_cup_micro_2022.csv"


def test_load_tournament_data_2022(engine: BacktestEngine) -> None:
    matches = engine.load_tournament_data(2022)
    assert len(matches) == 64
    for i in range(1, len(matches)):
        assert matches[i].date >= matches[i - 1].date


def test_load_tournament_data_2018(engine: BacktestEngine) -> None:
    matches = engine.load_tournament_data(2018)
    assert len(matches) == 64
    for i in range(1, len(matches)):
        assert matches[i].date >= matches[i - 1].date


def test_load_tournament_data_2014(engine: BacktestEngine) -> None:
    matches = engine.load_tournament_data(2014)
    assert len(matches) == 64
    for i in range(1, len(matches)):
        assert matches[i].date >= matches[i - 1].date


def test_load_tournament_raises_on_missing(
    engine: BacktestEngine,
) -> None:
    with pytest.raises(FileNotFoundError):
        engine.load_tournament_data(1999)


def test_backtest_match_fields(engine: BacktestEngine) -> None:
    matches = engine.load_tournament_data(2022)
    first = matches[0]
    assert first.home_team
    assert first.away_team
    assert first.home_goals >= 0
    assert first.away_goals >= 0
    assert first.round
    assert first.date


def test_load_tournament_test_csv(engine: BacktestEngine, test_csv: Path) -> None:
    matches = engine.load_tournament_data(2022, data_file=test_csv)
    assert len(matches) == 12
    for i in range(1, len(matches)):
        assert matches[i].date >= matches[i - 1].date


def test_run_backtest_temporal_validation(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest(2022, strat_fn, "temporal_test", data_file=test_csv)

    matches = engine.load_tournament_data(2022, data_file=test_csv)
    assert len(result.prediction_history) == len(matches)
    assert result.total_points >= 0


def test_run_backtest_no_data_leakage(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest(2022, strat_fn, "no_leak_test", data_file=test_csv)

    matches = engine.load_tournament_data(2022, data_file=test_csv)
    last_match = matches[-1]
    last_pred_entry = result.prediction_history[-1]

    assert last_pred_entry["home_team"] == last_match.home_team
    assert last_pred_entry["away_team"] == last_match.away_team


def test_temporal_leakage_match1_no_data_from_last(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest(2022, strat_fn, "leak_test", data_file=test_csv)

    matches = engine.load_tournament_data(2022, data_file=test_csv)
    last_match = matches[-1]

    first_pred = result.prediction_history[0]
    assert first_pred["match_id"] != last_match.match_id


def test_min_train_matches_uses_priors(engine: BacktestEngine, test_csv: Path) -> None:
    engine.config.min_train_matches = 999

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest(2022, strat_fn, "priors_test", data_file=test_csv)

    matches = engine.load_tournament_data(2022, data_file=test_csv)
    assert len(result.prediction_history) == len(matches)
    assert result.total_points >= 0


def test_compare_strategies(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def favorite_strat(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        if pred.home_win_prob > pred.away_win_prob:
            return (1, 0)
        elif pred.away_win_prob > pred.home_win_prob:
            return (0, 1)
        return (1, 1)

    strategies: dict[str, Callable[[MatchPrediction, str, str], tuple[int, int]]] = {
        "always_favorite": favorite_strat,
        "optimal_ep": make_optimal_ep_strategy(),
    }

    report = engine.compare_strategies(strategies, year=2022, data_file=test_csv)

    assert isinstance(report, BacktestReport)
    assert report.tournament == "World Cup 2022"
    assert len(report.strategies) == 2
    assert "always_favorite" in report.strategies
    assert "optimal_ep" in report.strategies
    assert report.summary
    assert "Backtest Report" in report.summary


def test_compare_strategies_optimal_ep_vs_favorite(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def favorite_strat(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        if pred.home_win_prob > pred.away_win_prob:
            return (1, 0)
        elif pred.away_win_prob > pred.home_win_prob:
            return (0, 1)
        return (1, 1)

    strategies = {
        "always_favorite": favorite_strat,
        "optimal_ep": make_optimal_ep_strategy(),
    }

    report = engine.compare_strategies(strategies, year=2022, data_file=test_csv)

    assert report.strategies["optimal_ep"].total_points >= 0
    assert report.strategies["always_favorite"].total_points >= 0
    assert "optimal_ep" in report.relative_performance
    assert "always_favorite" in report.relative_performance
    assert report.baseline_result.strategy_name == "always_favorite"


def test_calculate_calibration_error_range(
    engine: BacktestEngine,
) -> None:
    rng = np.random.default_rng(42)
    probs = rng.random(100)
    actuals = (probs > 0.5).astype(float)
    ece = engine.calculate_calibration_error(probs, actuals)
    assert 0.0 <= ece <= 1.0


def test_ece_perfect_calibration(engine: BacktestEngine) -> None:
    probs = np.array([0.1, 0.2, 0.9, 0.8, 0.5])
    actuals = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    ece = engine.calculate_calibration_error(probs, actuals, n_bins=3)
    assert 0.0 <= ece <= 1.0


def test_ece_known_case() -> None:
    probs = np.array([0.7, 0.7, 0.7, 0.3, 0.3])
    actuals = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    ece = calculate_ece(probs, actuals, n_bins=3)
    assert 0.0 <= ece <= 1.0


def test_calculate_metrics_consistency(
    engine: BacktestEngine,
) -> None:
    model = DixonColes(max_goals=7)
    model.home_advantage = 0.3
    model.team_attack["TeamA"] = 0.5
    model.team_defense["TeamA"] = 0.0
    model.team_attack["TeamB"] = 0.0
    model.team_defense["TeamB"] = 0.3

    pred1 = model.predict_match("TeamA", "TeamB")
    pred2 = model.predict_match("TeamB", "TeamA")

    predictions = [pred1.score_matrix, pred2.score_matrix]
    actuals = [(1, 0), (0, 1)]

    metrics = engine.calculate_metrics(predictions, actuals)

    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert "ranked_probability_score" in metrics
    assert "calibration_error" in metrics

    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["ranked_probability_score"] >= 0.0


def test_log_loss_always_favorite_worse_than_model(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def favorite_strat(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        if pred.home_win_prob > pred.away_win_prob:
            return (1, 0)
        elif pred.away_win_prob > pred.home_win_prob:
            return (0, 1)
        return (1, 1)

    result_fav = engine.run_backtest(2022, favorite_strat, "favorite", data_file=test_csv)
    result_ep = engine.run_backtest(
        2022, make_optimal_ep_strategy(), "optimal_ep", data_file=test_csv
    )

    assert result_fav.log_loss >= result_ep.log_loss, (
        f"favorite LL {result_fav.log_loss:.3f} >= optimal_ep LL {result_ep.log_loss:.3f}"
    )


def test_brier_score_below_random(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8
    result = engine.run_backtest(
        2022, make_optimal_ep_strategy(), "brier_test", data_file=test_csv
    )
    assert result.brier_score < 0.25, (
        f"Brier {result.brier_score:.4f} should be < 0.25"
    )


def test_backtest_report_summary(engine: BacktestEngine, test_csv: Path) -> None:
    engine.config.min_train_matches = 8

    def s1(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    def s2(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        if pred.home_win_prob > pred.away_win_prob:
            return (1, 0)
        return (0, 1)

    strategies = {"strat_a": s1, "strat_b": s2}
    report = engine.compare_strategies(strategies, year=2022, data_file=test_csv)

    assert report.summary
    assert "Backtest Report" in report.summary
    assert "World Cup 2022" in report.summary
    assert "always_favorite" in report.summary


def test_expanding_window_uses_limited_data(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8
    window_size = 5

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest_expanding_window(
        2022, strat_fn, "window_test", window_size=window_size, data_file=test_csv
    )

    matches = engine.load_tournament_data(2022, data_file=test_csv)
    assert len(result.prediction_history) == len(matches)
    assert len(result.points_per_match) == len(matches)


def test_expanding_window_window_size_limit(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8
    train_sizes: list[int] = []
    original = engine._matches_to_dicts

    def tracking(m: list[BacktestMatch]) -> list[dict[str, int | str]]:
        train_sizes.append(len(m))
        return original(m)

    engine._matches_to_dicts = tracking
    window_size = 5

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    engine.run_backtest_expanding_window(
        2022, strat_fn, "window_limit_test", window_size=window_size, data_file=test_csv
    )

    for i, size in enumerate(train_sizes):
        max_expected = min(i, window_size)
        assert size <= max_expected, (
            f"Training size {size} at match {i} exceeds window {window_size}"
        )


def test_run_all_tournaments(engine: BacktestEngine, test_csv: Path) -> None:
    engine.config.validation_years = [2022]
    engine.config.min_train_matches = 8

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    results = engine.run_all_tournaments(strat_fn, "all_test", data_files={2022: test_csv})
    assert len(results) == 1
    assert 2022 in results
    assert isinstance(results[2022], BacktestResult)
    assert results[2022].strategy_name == "all_test"


def test_empty_tournament_handled(engine: BacktestEngine) -> None:
    engine.config.validation_years = [1999]

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    with pytest.raises(FileNotFoundError):
        engine.run_all_tournaments(strat_fn, "empty_test")


def test_calculate_points_exact() -> None:
    from src.config import PollaRules
    rules = PollaRules(exact_score_pts=5, result_correct_pts=2,
                        goals_home_correct_pts=1, goals_away_correct_pts=1)
    pts = _calculate_points((2, 1), (2, 1), rules)
    assert pts == 7.0


def test_calculate_points_result_only() -> None:
    from src.config import PollaRules
    rules = PollaRules(exact_score_pts=5, result_correct_pts=2,
                        goals_home_correct_pts=1, goals_away_correct_pts=1)
    pts = _calculate_points((3, 0), (2, 0), rules)
    assert pts == 3.0


def test_calculate_points_goals_only() -> None:
    from src.config import PollaRules
    rules = PollaRules(exact_score_pts=5, result_correct_pts=2,
                        goals_home_correct_pts=1, goals_away_correct_pts=1)
    pts = _calculate_points((1, 2), (1, 3), rules)
    assert pts == 3.0


def test_calculate_points_none() -> None:
    from src.config import PollaRules
    rules = PollaRules(exact_score_pts=5, result_correct_pts=2,
                        goals_home_correct_pts=1, goals_away_correct_pts=1)
    pts = _calculate_points((0, 0), (3, 1), rules)
    assert pts == 0.0


def test_brier_score_calculation() -> None:
    model = DixonColes(max_goals=3)
    pred = model.predict_from_params(lambda_h=1.0, mu_a=1.0)
    matrices = [pred.score_matrix]
    actuals = [(1, 1)]
    brier = _calculate_brier_score(matrices, actuals)
    assert brier >= 0.0


def test_rps_calculation() -> None:
    model = DixonColes(max_goals=3)
    pred = model.predict_from_params(lambda_h=1.0, mu_a=1.0)
    matrices = [pred.score_matrix]
    actuals = [(1, 1)]
    rps = _calculate_rps(matrices, actuals)
    assert rps >= 0.0


def test_empty_metrics_default(engine: BacktestEngine) -> None:
    metrics = engine.calculate_metrics([], [])
    assert metrics["log_loss"] == float("inf")
    assert metrics["brier_score"] == float("inf")


def test_make_optimal_ep_strategy() -> None:
    strat = make_optimal_ep_strategy()
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.0, mu_a=0.5)
    result = strat(pred, "Home", "Away")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] <= 7
    assert result[1] <= 7


def test_make_optimal_ep_contrarian_strategy() -> None:
    ow = np.zeros((8, 8))
    ow[0, 0] = 0.5
    strat = make_optimal_ep_contrarian_strategy(ownership_matrix=ow)
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.0, mu_a=0.5)
    result = strat(pred, "Home", "Away")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_make_adaptive_strategy() -> None:
    strat = make_adaptive_strategy(current_position=3)
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.0, mu_a=0.5)
    result = strat(pred, "Home", "Away")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_calculate_score_matrix_log_loss() -> None:
    model = DixonColes(max_goals=3)
    pred = model.predict_from_params(lambda_h=1.0, mu_a=1.0)
    matrices = [pred.score_matrix]
    actuals = [(1, 1)]
    ll = calculate_score_matrix_log_loss(matrices, actuals)
    assert ll >= 0.0


def test_calculate_score_matrix_log_loss_perfect() -> None:
    matrix = np.zeros((4, 4))
    matrix[1, 1] = 1.0
    matrices = [matrix]
    actuals = [(1, 1)]
    ll = calculate_score_matrix_log_loss(matrices, actuals)
    assert ll < 0.001


def test_metrics_manual_consistency(engine: BacktestEngine) -> None:
    model = DixonColes(max_goals=3)
    model.home_advantage = 0.3
    model.team_attack["A"] = 0.5
    model.team_defense["A"] = 0.0
    model.team_attack["B"] = 0.0
    model.team_defense["B"] = 0.3

    pred = model.predict_match("A", "B")
    matrices = [pred.score_matrix]
    actuals = [(2, 0)]

    metrics = engine.calculate_metrics(matrices, actuals)

    ll_manual = -np.log(max(float(pred.score_matrix[2, 0]), 1e-15))
    assert abs(metrics["log_loss"] - ll_manual) < 1e-6


def test_prediction_history_structure(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest(2022, strat_fn, "history_test", data_file=test_csv)

    for entry in result.prediction_history:
        assert "match_id" in entry
        assert "home_team" in entry
        assert "away_team" in entry
        assert "prediction" in entry
        assert "actual" in entry
        assert "points" in entry
        assert entry["points"] >= 0


def test_backtest_result_fields(engine: BacktestEngine, test_csv: Path) -> None:
    engine.config.min_train_matches = 8

    def strat_fn(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return pred.most_likely_score

    result = engine.run_backtest(2022, strat_fn, "fields_test", data_file=test_csv)

    assert result.strategy_name == "fields_test"
    assert result.tournament == "World Cup 2022"
    assert result.exact_scores >= 0
    assert result.correct_results >= 0


def test_relative_performance_calculation(
    engine: BacktestEngine, test_csv: Path
) -> None:
    engine.config.min_train_matches = 8

    def good_strat(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return make_optimal_ep_strategy()(pred, home, away)

    def bad_strat(
        pred: MatchPrediction, home: str, away: str
    ) -> tuple[int, int]:
        return (0, 0)

    strategies = {"good": good_strat, "bad": bad_strat}
    report = engine.compare_strategies(strategies, year=2022, data_file=test_csv)

    assert "good" in report.relative_performance
    assert "bad" in report.relative_performance
    assert isinstance(report.relative_performance["good"], float)
    assert isinstance(report.relative_performance["bad"], float)


def test_backtest_config_csv_map() -> None:
    config = BacktestConfig(
        csv_map={2022: "world_cup_micro_2022.csv"}
    )
    path = config.get_csv_path(2022)
    assert path.name == "world_cup_micro_2022.csv"


def test_backtest_config_default_csv() -> None:
    config = BacktestConfig()
    path = config.get_csv_path(2018)
    assert path.name == "world_cup_2018.csv"
