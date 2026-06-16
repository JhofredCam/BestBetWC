import pytest

from src.config import (
    MODEL_CONFIG,
    POLLA_RULES,
    STRATEGY_CONFIG,
    ModelConfig,
    PollaRules,
    StrategyConfig,
)


class TestPollaRules:
    def test_default_result_correct_pts(self) -> None:
        rules = PollaRules()
        assert rules.result_correct_pts == 2

    def test_default_result_incorrect_pts(self) -> None:
        rules = PollaRules()
        assert rules.result_incorrect_pts == 0

    def test_default_exact_score_pts(self) -> None:
        rules = PollaRules()
        assert rules.exact_score_pts == 5

    def test_default_goals_home_correct_pts(self) -> None:
        rules = PollaRules()
        assert rules.goals_home_correct_pts == 1

    def test_default_goals_away_correct_pts(self) -> None:
        rules = PollaRules()
        assert rules.goals_away_correct_pts == 1

    def test_default_unique_prediction_bonus(self) -> None:
        rules = PollaRules()
        assert rules.unique_prediction_bonus == 2

    def test_default_round_bonus_16(self) -> None:
        rules = PollaRules()
        assert rules.round_bonus_16 == 10

    def test_default_round_bonus_8(self) -> None:
        rules = PollaRules()
        assert rules.round_bonus_8 == 8

    def test_default_round_bonus_4(self) -> None:
        rules = PollaRules()
        assert rules.round_bonus_4 == 4

    def test_default_round_bonus_semi(self) -> None:
        rules = PollaRules()
        assert rules.round_bonus_semi == 2

    def test_default_round_bonus_final(self) -> None:
        rules = PollaRules()
        assert rules.round_bonus_final == 5

    def test_default_num_participants(self) -> None:
        rules = PollaRules()
        assert rules.num_participants == 15

    def test_default_max_goals(self) -> None:
        rules = PollaRules()
        assert rules.max_goals == 7

    def test_custom_values(self) -> None:
        rules = PollaRules(
            result_correct_pts=3,
            exact_score_pts=6,
            num_participants=20,
            max_goals=5,
        )
        assert rules.result_correct_pts == 3
        assert rules.exact_score_pts == 6
        assert rules.num_participants == 20
        assert rules.max_goals == 5

    def test_singleton_instance(self) -> None:
        assert POLLA_RULES.result_correct_pts == 2
        assert POLLA_RULES.exact_score_pts == 5
        assert POLLA_RULES.num_participants == 15


class TestModelConfig:
    def test_default_dixon_coles_rho(self) -> None:
        config = ModelConfig()
        assert config.dixon_coles_rho == pytest.approx(-0.13)

    def test_default_calibration_method(self) -> None:
        config = ModelConfig()
        assert config.calibration_method == "isotonic"

    def test_default_validation_folds(self) -> None:
        config = ModelConfig()
        assert config.validation_folds == 5

    def test_default_ensemble_weights_filled(self) -> None:
        config = ModelConfig()
        assert config.ensemble_weights == {
            "dixon_coles": 0.4,
            "market": 0.4,
            "gradient_boost": 0.2,
        }

    def test_post_init_idempotent_on_none(self) -> None:
        config = ModelConfig()
        expected = {
            "dixon_coles": 0.4,
            "market": 0.4,
            "gradient_boost": 0.2,
        }
        assert config.ensemble_weights == expected
        config.__post_init__()
        assert config.ensemble_weights == expected

    def test_ensemble_weights_when_provided(self) -> None:
        weights = {"model_a": 0.5, "model_b": 0.5}
        config = ModelConfig(ensemble_weights=weights)
        assert config.ensemble_weights == weights

    def test_ensemble_weights_not_overwritten_when_provided(self) -> None:
        weights = {"model_a": 0.5, "model_b": 0.5}
        config = ModelConfig(ensemble_weights=weights)
        config.__post_init__()
        assert config.ensemble_weights == weights

    def test_singleton_instance(self) -> None:
        assert MODEL_CONFIG.dixon_coles_rho == pytest.approx(-0.13)
        assert MODEL_CONFIG.calibration_method == "isotonic"
        assert MODEL_CONFIG.validation_folds == 5


class TestStrategyConfig:
    def test_default_leading_threshold(self) -> None:
        config = StrategyConfig()
        assert config.leading_threshold == 1

    def test_default_middle_range(self) -> None:
        config = StrategyConfig()
        assert config.middle_range == (2, 5)

    def test_default_behind_range(self) -> None:
        config = StrategyConfig()
        assert config.behind_range == (6, 10)

    def test_default_trailing_range(self) -> None:
        config = StrategyConfig()
        assert config.trailing_range == (11, 15)

    def test_default_risk_aversion_leading(self) -> None:
        config = StrategyConfig()
        assert config.risk_aversion_leading == pytest.approx(0.8)

    def test_default_risk_aversion_trailing(self) -> None:
        config = StrategyConfig()
        assert config.risk_aversion_trailing == pytest.approx(0.2)

    def test_default_contrarian_weight(self) -> None:
        config = StrategyConfig()
        assert config.contrarian_weight == pytest.approx(0.3)

    def test_position_ranges_are_contiguous(self) -> None:
        config = StrategyConfig()
        assert config.leading_threshold == 1
        assert config.middle_range[1] + 1 == config.behind_range[0]
        assert config.behind_range[1] + 1 == config.trailing_range[0]

    def test_trailing_range_ends_at_participants(self) -> None:
        config = StrategyConfig()
        assert config.trailing_range[1] == POLLA_RULES.num_participants

    def test_risk_values_in_valid_range(self) -> None:
        config = StrategyConfig()
        assert 0.0 <= config.risk_aversion_leading <= 1.0
        assert 0.0 <= config.risk_aversion_trailing <= 1.0
        assert 0.0 <= config.contrarian_weight <= 1.0

    def test_singleton_instance(self) -> None:
        assert STRATEGY_CONFIG.leading_threshold == 1
        assert STRATEGY_CONFIG.middle_range == (2, 5)
        assert STRATEGY_CONFIG.trailing_range == (11, 15)
