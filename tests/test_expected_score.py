import pytest

from src.config import PollaRules
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator


@pytest.fixture
def sample_prediction() -> MatchPrediction:
    model = DixonColes(max_goals=7)
    return model.predict_from_params(lambda_h=1.5, mu_a=1.0)


@pytest.fixture
def calculator() -> ExpectedScoreCalculator:
    return ExpectedScoreCalculator()


def test_dixon_coles_probabilities_sum_to_one(sample_prediction: MatchPrediction) -> None:
    total = sample_prediction.score_matrix.sum()
    assert abs(total - 1.0) < 1e-6


def test_dixon_coles_result_probabilities(sample_prediction: MatchPrediction) -> None:
    total = (
        sample_prediction.home_win_prob
        + sample_prediction.draw_prob
        + sample_prediction.away_win_prob
    )
    assert abs(total - 1.0) < 1e-6


def test_dixon_coles_home_advantage() -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.0, mu_a=1.0)
    assert pred.home_win_prob > pred.away_win_prob


def test_dixon_coles_equal_teams() -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.2, mu_a=1.2)
    assert abs(pred.home_win_prob - pred.away_win_prob) < 0.1


def test_expected_score_exact_match(
    calculator: ExpectedScoreCalculator, sample_prediction: MatchPrediction
) -> None:
    result = calculator.calculate_ep(sample_prediction, pred_home=1, pred_away=0)
    assert result.ep_total > 0
    assert result.prob_exact > 0
    assert result.prob_goals_home > 0


def test_expected_score_high_probability_higher_ep(
    calculator: ExpectedScoreCalculator,
) -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.5, mu_a=0.5)

    result_high = calculator.calculate_ep(pred, pred_home=2, pred_away=0)
    result_low = calculator.calculate_ep(pred, pred_home=4, pred_away=3)

    assert result_high.ep_total > result_low.ep_total


def test_expected_score_unique_bonus_increases_ep(
    calculator: ExpectedScoreCalculator, sample_prediction: MatchPrediction
) -> None:
    result_no_ownership = calculator.calculate_ep(
        sample_prediction, pred_home=2, pred_away=1, ownership_estimate=0.0
    )
    result_high_ownership = calculator.calculate_ep(
        sample_prediction, pred_home=2, pred_away=1, ownership_estimate=0.5
    )

    assert result_no_ownership.ep_unique > result_high_ownership.ep_unique


def test_find_optimal_prediction(
    calculator: ExpectedScoreCalculator, sample_prediction: MatchPrediction
) -> None:
    optimal = calculator.find_optimal_prediction(sample_prediction)
    assert optimal.ep_total > 0
    assert 0 <= optimal.home_goals <= 7
    assert 0 <= optimal.away_goals <= 7


def test_rank_all_predictions(
    calculator: ExpectedScoreCalculator, sample_prediction: MatchPrediction
) -> None:
    ranked = calculator.rank_all_predictions(sample_prediction)
    assert len(ranked) == 64
    for i in range(len(ranked) - 1):
        assert ranked[i].ep_total >= ranked[i + 1].ep_total


def test_polla_rules_config() -> None:
    rules = PollaRules(
        result_correct_pts=3,
        exact_score_pts=10,
        num_participants=20,
    )
    calculator = ExpectedScoreCalculator(rules)
    assert calculator.rules.result_correct_pts == 3
    assert calculator.rules.exact_score_pts == 10
    assert calculator.rules.num_participants == 20


def test_expected_score_components(
    calculator: ExpectedScoreCalculator, sample_prediction: MatchPrediction
) -> None:
    result = calculator.calculate_ep(sample_prediction, pred_home=1, pred_away=1)

    expected_total = (
        result.ep_exact
        + result.ep_result
        + result.ep_goals_home
        + result.ep_goals_away
        + result.ep_unique
    )
    assert abs(result.ep_total - expected_total) < 1e-6


def test_exact_score_replaces_result(
    calculator: ExpectedScoreCalculator,
) -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.5, mu_a=1.0)

    result = calculator.calculate_ep(pred, pred_home=1, pred_away=0)

    assert result.ep_exact == result.prob_exact * 5
    assert result.ep_result == result.prob_result * 2
