import numpy as np
import pytest

from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreResult
from src.optimization.strategy import StrategyMode, StrategySelector


@pytest.fixture
def selector() -> StrategySelector:
    return StrategySelector()


@pytest.fixture
def sample_prediction():
    model = DixonColes(max_goals=7)
    return model.predict_from_params(lambda_h=1.5, mu_a=1.0)


def test_strategy_mode_leading(selector: StrategySelector) -> None:
    mode = selector.determine_mode(current_position=1, total_participants=15)
    assert mode == StrategyMode.MINIMIZE_RISK


def test_strategy_mode_middle(selector: StrategySelector) -> None:
    mode = selector.determine_mode(current_position=3, total_participants=15)
    assert mode == StrategyMode.BALANCED


def test_strategy_mode_behind(selector: StrategySelector) -> None:
    mode = selector.determine_mode(current_position=8, total_participants=15)
    assert mode == StrategyMode.DIFFERENTIATION


def test_strategy_mode_trailing(selector: StrategySelector) -> None:
    mode = selector.determine_mode(current_position=13, total_participants=15)
    assert mode == StrategyMode.HIGH_RISK


def test_get_recommendation_returns_valid(
    selector: StrategySelector, sample_prediction
) -> None:
    rec = selector.get_recommendation(
        prediction=sample_prediction,
        current_position=1,
        total_participants=15,
    )

    assert rec.prediction.ep_total > 0
    assert rec.strategy_mode == StrategyMode.MINIMIZE_RISK
    assert 0 <= rec.risk_score <= 1
    assert rec.upside_potential > 0
    assert 0 <= rec.risk_of_ruin <= 1


def test_leading_strategy_lower_risk(
    selector: StrategySelector, sample_prediction
) -> None:
    rec_leading = selector.get_recommendation(
        prediction=sample_prediction,
        current_position=1,
        total_participants=15,
    )
    rec_trailing = selector.get_recommendation(
        prediction=sample_prediction,
        current_position=15,
        total_participants=15,
    )

    assert rec_leading.risk_score < rec_trailing.risk_score


def test_recommendation_includes_reasoning(
    selector: StrategySelector, sample_prediction
) -> None:
    rec = selector.get_recommendation(
        prediction=sample_prediction,
        current_position=5,
        total_participants=15,
    )

    assert len(rec.reasoning) > 0
    assert rec.strategy_mode == StrategyMode.BALANCED


def test_select_differentiation_fallback(
    selector: StrategySelector,
) -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.5, mu_a=1.0)
    max_goals = pred.score_matrix.shape[0]
    ownership_matrix = np.ones((max_goals, max_goals))

    all_preds = selector.ep_calculator.rank_all_predictions(pred, ownership_matrix)
    result = selector._select_differentiation(all_preds, pred, ownership_matrix)

    assert result is all_preds[0]


def test_select_high_risk_fallback(
    selector: StrategySelector,
) -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.5, mu_a=1.0)
    max_goals = pred.score_matrix.shape[0]
    ownership_matrix = np.ones((max_goals, max_goals))

    all_preds = selector.ep_calculator.rank_all_predictions(pred, ownership_matrix)
    result = selector._select_high_risk(all_preds, pred, ownership_matrix)

    assert result is all_preds[0]


def test_select_minimize_risk_fallback(
    selector: StrategySelector, sample_prediction
) -> None:
    preds = [
        ExpectedScoreResult(
            home_goals=i,
            away_goals=0,
            ep_total=float(i),
            ep_exact=0.1,
            ep_result=0.1,
            ep_goals_home=0.0,
            ep_goals_away=0.0,
            ep_unique=0.0,
            prob_exact=0.1,
            prob_result=0.1,
            prob_goals_home=0.0,
            prob_goals_away=0.0,
            ownership_estimate=0.0,
            contrarian_value=0.0,
        )
        for i in range(5)
    ]
    result = selector._select_minimize_risk(preds, sample_prediction)
    assert result is preds[0]
