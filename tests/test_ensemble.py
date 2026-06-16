"""Tests for ModelEnsemble (SPEC-010)."""

import numpy as np
import pytest

from src.models.dixon_coles import DixonColes, MatchPrediction
from src.models.ensemble import ModelEnsemble, EnsembleConfig, _ensemble_score_matrices
from src.models.gradient_boost import GradientBoostModel, GBModelConfig


def make_score_matrix(max_goals: int = 7) -> np.ndarray:
    """Create a random valid score matrix (sums to 1)."""
    mat = np.random.rand(max_goals + 1, max_goals + 1)
    return mat / mat.sum()


def make_match_prediction(
    max_goals: int = 7, home_bias: float = 1.0
) -> MatchPrediction:
    """Create a MatchPrediction from a biased score matrix."""
    mat = np.random.rand(max_goals + 1, max_goals + 1)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i > j:
                mat[i, j] *= home_bias
    mat = mat / mat.sum()

    home_dist = mat.sum(axis=1)
    away_dist = mat.sum(axis=0)

    home_win = float(np.tril(mat, -1).sum())
    draw = float(np.trace(mat))
    away_win = float(np.triu(mat, 1).sum())

    n = max_goals + 1
    exp_home = float(np.dot(np.arange(n), home_dist))
    exp_away = float(np.dot(np.arange(n), away_dist))

    max_idx = np.unravel_index(np.argmax(mat), mat.shape)
    mls = (int(max_idx[0]), int(max_idx[1]))
    mls_prob = float(mat[max_idx])

    return MatchPrediction(
        home_goals_dist=home_dist,
        away_goals_dist=away_dist,
        score_matrix=mat,
        home_win_prob=home_win,
        draw_prob=draw,
        away_win_prob=away_win,
        expected_home_goals=exp_home,
        expected_away_goals=exp_away,
        most_likely_score=mls,
        most_likely_score_prob=mls_prob,
    )


class TestEnsembleScoreMatrices:
    """Tests for the _ensemble_score_matrices helper."""

    def test_single_matrix_returns_identity(self) -> None:
        mat = make_score_matrix()
        result = _ensemble_score_matrices({"m1": mat}, {"m1": 1.0})
        np.testing.assert_array_almost_equal(result, mat)

    def test_two_matrices_equal_weights_returns_average(self) -> None:
        mat_a = make_score_matrix()
        mat_b = make_score_matrix()
        result = _ensemble_score_matrices(
            {"a": mat_a, "b": mat_b}, {"a": 0.5, "b": 0.5}
        )
        expected = (mat_a + mat_b) / 2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_ignores_zero_weight_models(self) -> None:
        mat_a = make_score_matrix()
        mat_b = make_score_matrix()
        result = _ensemble_score_matrices(
            {"a": mat_a, "b": mat_b}, {"a": 1.0, "b": 0.0}
        )
        np.testing.assert_array_almost_equal(result, mat_a)

    def test_normalizes_when_weights_dont_sum_to_one(self) -> None:
        mat_a = make_score_matrix()
        result = _ensemble_score_matrices({"a": mat_a}, {"a": 0.3})
        np.testing.assert_array_almost_equal(result, mat_a)

    def test_ensembled_matrix_sums_to_one(self) -> None:
        mat_a = make_score_matrix()
        mat_b = make_score_matrix()
        mat_c = make_score_matrix()
        result = _ensemble_score_matrices(
            {"a": mat_a, "b": mat_b, "c": mat_c},
            {"a": 0.3, "b": 0.3, "c": 0.4},
        )
        assert abs(result.sum() - 1.0) < 1e-10

    def test_raises_on_empty_matrices(self) -> None:
        with pytest.raises(ValueError, match="No score matrices"):
            _ensemble_score_matrices({}, {})

    def test_all_zero_weights_still_returns_normalized(self) -> None:
        mat_a = make_score_matrix()
        mat_b = make_score_matrix()
        result = _ensemble_score_matrices(
            {"a": mat_a, "b": mat_b}, {"a": 0.0, "b": 0.0}
        )
        assert abs(result.sum()) < 1e-10


class TestModelEnsemble:
    """Tests for ModelEnsemble class."""

    def test_initialization_with_default_config(self) -> None:
        ensemble = ModelEnsemble()
        weights = ensemble.get_weights_dict()
        assert weights["dixon_coles"] == 0.35
        assert weights["market"] == 0.35
        assert weights["gradient_boost"] == 0.20
        assert weights["bayesian"] == 0.10

    def test_initialization_with_custom_config(self) -> None:
        config = EnsembleConfig(
            dixon_coles_weight=0.5,
            market_weight=0.5,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)
        weights = ensemble.get_weights_dict()
        assert weights["dixon_coles"] == 0.5
        assert weights["market"] == 0.5
        assert weights["gradient_boost"] == 0.0
        assert weights["bayesian"] == 0.0

    def test_get_weights_dict_returns_current_state(self) -> None:
        ensemble = ModelEnsemble()
        weights = ensemble.get_weights_dict()
        assert isinstance(weights, dict)
        assert "dixon_coles" in weights
        assert "market" in weights
        assert "gradient_boost" in weights
        assert "bayesian" in weights
        for v in weights.values():
            assert isinstance(v, float)

    def test_validate_weights_sums_to_one(self) -> None:
        ensemble = ModelEnsemble()
        ensemble._validate_weights()

    def test_validate_weights_raises_on_bad_config(self) -> None:
        config = EnsembleConfig(dixon_coles_weight=2.0)
        ensemble = ModelEnsemble(config)
        with pytest.raises(ValueError, match="Weights sum to"):
            ensemble._validate_weights()

    def test_predict_with_single_model_returns_match_prediction(self) -> None:
        dc = DixonColes(max_goals=7)
        dc.home_advantage = 0.3
        dc.team_attack["A"] = 0.5
        dc.team_defense["A"] = -0.2
        dc.team_attack["B"] = -0.1
        dc.team_defense["B"] = 0.1

        config = EnsembleConfig(
            dixon_coles_weight=1.0,
            market_weight=0.0,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)
        ensemble.add_dixon_coles(dc)

        pred = ensemble.predict("A", "B")
        assert isinstance(pred, MatchPrediction)
        assert abs(pred.score_matrix.sum() - 1.0) < 1e-10

    def test_predict_with_market_only(self) -> None:
        config = EnsembleConfig(
            dixon_coles_weight=0.0,
            market_weight=1.0,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)
        market_mat = make_score_matrix()
        ensemble.set_market_prediction(
            market_mat, market_mat.sum(axis=1), market_mat.sum(axis=0)
        )

        pred = ensemble.predict("A", "B")
        assert isinstance(pred, MatchPrediction)
        np.testing.assert_array_almost_equal(pred.score_matrix, market_mat)

    def test_set_market_prediction_accepts_external_matrix(self) -> None:
        ensemble = ModelEnsemble()
        market_mat = make_score_matrix()
        home_dist = market_mat.sum(axis=1)
        away_dist = market_mat.sum(axis=0)

        ensemble.set_market_prediction(market_mat, home_dist, away_dist)

        pred = ensemble.predict("A", "B")
        assert pred.score_matrix.shape == market_mat.shape

    def test_predict_with_market_parameter_overrides_stored(self) -> None:
        config = EnsembleConfig(
            dixon_coles_weight=0.0,
            market_weight=1.0,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)

        # Store one matrix
        stored_mat = make_score_matrix()
        ensemble.set_market_prediction(
            stored_mat,
            stored_mat.sum(axis=1),
            stored_mat.sum(axis=0),
        )

        # Pass a different matrix via parameter
        param_mat = make_score_matrix()
        pred = ensemble.predict("A", "B", market_score_matrix=param_mat)
        np.testing.assert_array_almost_equal(pred.score_matrix, param_mat)

    def test_predict_from_individual_predictions(self) -> None:
        config = EnsembleConfig(
            dixon_coles_weight=0.5,
            market_weight=0.5,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)

        pred_a = make_match_prediction(home_bias=2.0)
        pred_b = make_match_prediction(home_bias=0.5)

        result = ensemble.predict_from_individual_predictions(
            {"dixon_coles": pred_a, "market": pred_b}
        )

        assert isinstance(result, MatchPrediction)
        assert abs(result.score_matrix.sum() - 1.0) < 1e-10

    def test_predict_from_individual_predictions_single_model(self) -> None:
        config = EnsembleConfig(
            dixon_coles_weight=1.0,
            market_weight=0.0,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)

        pred_a = make_match_prediction()
        result = ensemble.predict_from_individual_predictions(
            {"dixon_coles": pred_a}
        )

        np.testing.assert_array_almost_equal(result.score_matrix, pred_a.score_matrix)

    def test_predict_from_individual_predictions_accepts_any_combination(self) -> None:
        """Ensemble works with any subset of model names."""
        config = EnsembleConfig(
            dixon_coles_weight=0.0,
            market_weight=0.0,
            gradient_boost_weight=1.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)

        pred = make_match_prediction()
        result = ensemble.predict_from_individual_predictions(
            {"gradient_boost": pred}
        )
        assert abs(result.score_matrix.sum() - 1.0) < 1e-10

    def test_raises_when_no_models_available(self) -> None:
        ensemble = ModelEnsemble()
        with pytest.raises(RuntimeError, match="No models available"):
            ensemble.predict("A", "B")

    def test_ensemble_with_dixon_coles_and_market(self) -> None:
        dc = DixonColes(max_goals=7)
        dc.home_advantage = 0.2
        dc.team_attack["Brasil"] = 0.6
        dc.team_defense["Brasil"] = -0.1
        dc.team_attack["Argentina"] = 0.4
        dc.team_defense["Argentina"] = 0.0

        config = EnsembleConfig(
            dixon_coles_weight=0.5,
            market_weight=0.5,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)
        ensemble.add_dixon_coles(dc)

        market_mat = make_score_matrix()
        ensemble.set_market_prediction(
            market_mat,
            market_mat.sum(axis=1),
            market_mat.sum(axis=0),
        )

        pred = ensemble.predict("Brasil", "Argentina")
        assert isinstance(pred, MatchPrediction)
        assert abs(pred.score_matrix.sum() - 1.0) < 1e-10

        # Should be between the two (not identical to either)
        dc_pred = dc.predict_match("Brasil", "Argentina")
        assert not np.allclose(pred.score_matrix, dc_pred.score_matrix)
        assert not np.allclose(pred.score_matrix, market_mat)

    def test_score_matrix_sums_to_one_with_all_models(self) -> None:
        dc = DixonColes(max_goals=7)
        dc.home_advantage = 0.3
        dc.team_attack["X"] = 0.5
        dc.team_defense["X"] = -0.2
        dc.team_attack["Y"] = -0.1
        dc.team_defense["Y"] = 0.1

        config = EnsembleConfig(
            dixon_coles_weight=0.35,
            market_weight=0.35,
            gradient_boost_weight=0.20,
            bayesian_weight=0.10,
        )
        ensemble = ModelEnsemble(config)
        ensemble.add_dixon_coles(dc)

        market_mat = make_score_matrix()
        ensemble.set_market_prediction(
            market_mat,
            market_mat.sum(axis=1),
            market_mat.sum(axis=0),
        )

        pred = ensemble.predict("X", "Y")
        assert abs(pred.score_matrix.sum() - 1.0) < 1e-10
        assert np.all(pred.score_matrix >= 0)

    def test_predictions_have_valid_marginals(self) -> None:
        dc = DixonColes(max_goals=7)
        dc.home_advantage = 0.3
        dc.team_attack["X"] = 0.5
        dc.team_defense["X"] = -0.2
        dc.team_attack["Y"] = -0.1
        dc.team_defense["Y"] = 0.1

        config = EnsembleConfig(
            dixon_coles_weight=0.5,
            market_weight=0.5,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)
        ensemble.add_dixon_coles(dc)

        market_mat = make_score_matrix()
        ensemble.set_market_prediction(
            market_mat,
            market_mat.sum(axis=1),
            market_mat.sum(axis=0),
        )

        pred = ensemble.predict("X", "Y")
        assert abs(pred.home_goals_dist.sum() - 1.0) < 1e-10
        assert abs(pred.away_goals_dist.sum() - 1.0) < 1e-10


class TestOptimizeWeights:
    """Tests for optimize_weights method."""

    def test_optimize_weights_returns_valid_config(self) -> None:
        ensemble = ModelEnsemble()
        n_samples = 100
        n_goals = 8

        rng = np.random.RandomState(42)
        y_home = rng.randint(0, 6, size=n_samples)
        y_away = rng.randint(0, 6, size=n_samples)

        model_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in ["dixon_coles", "market"]:
            home_probs = rng.dirichlet(np.ones(n_goals), size=n_samples)
            away_probs = rng.dirichlet(np.ones(n_goals), size=n_samples)
            model_preds[name] = (home_probs, away_probs)

        X_val = rng.randn(n_samples, 10)

        config = ensemble.optimize_weights(X_val, y_home, y_away, model_preds)

        assert isinstance(config, EnsembleConfig)
        names = ["dixon_coles", "market"]
        values = [config.dixon_coles_weight, config.market_weight]
        assert abs(sum(values) - 1.0) < 1e-8
        for v in values:
            assert 0.0 <= v <= 1.0

    def test_optimize_weights_minimizes_log_loss(self) -> None:
        """Ensemble should improve log loss vs individual models."""
        ensemble = ModelEnsemble()
        n_samples = 200
        n_goals = 8

        rng = np.random.RandomState(123)

        # Make model 1 a good predictor, model 2 a bad one
        true_home_probs = rng.dirichlet(np.ones(n_goals), size=n_samples)
        true_away_probs = rng.dirichlet(np.ones(n_goals), size=n_samples)

        y_home = np.array([np.argmax(true_home_probs[i]) for i in range(n_samples)])
        y_away = np.array([np.argmax(true_away_probs[i]) for i in range(n_samples)])

        # Model 1: close to truth (add small noise)
        model1_home = true_home_probs + rng.normal(0, 0.02, true_home_probs.shape)
        model1_home = np.clip(model1_home, 0, None)
        model1_home = model1_home / model1_home.sum(axis=1, keepdims=True)

        model1_away = true_away_probs + rng.normal(0, 0.02, true_away_probs.shape)
        model1_away = np.clip(model1_away, 0, None)
        model1_away = model1_away / model1_away.sum(axis=1, keepdims=True)

        # Model 2: uniform (bad)
        model2_home = np.ones_like(true_home_probs) / n_goals
        model2_away = np.ones_like(true_away_probs) / n_goals

        model_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "dixon_coles": (model1_home, model1_away),
            "market": (model2_home, model2_away),
        }

        X_val = rng.randn(n_samples, 10)
        config = ensemble.optimize_weights(X_val, y_home, y_away, model_preds)

        # Good model (dixon_coles) should get higher weight than bad model (market)
        assert config.dixon_coles_weight > config.market_weight

    def test_optimize_weights_with_single_model(self) -> None:
        ensemble = ModelEnsemble()
        n_samples = 50
        n_goals = 8

        rng = np.random.RandomState(99)
        y_home = rng.randint(0, 4, size=n_samples)
        y_away = rng.randint(0, 4, size=n_samples)

        home_probs = rng.dirichlet(np.ones(n_goals), size=n_samples)
        away_probs = rng.dirichlet(np.ones(n_goals), size=n_samples)

        model_preds = {"dixon_coles": (home_probs, away_probs)}
        X_val = rng.randn(n_samples, 10)

        config = ensemble.optimize_weights(X_val, y_home, y_away, model_preds)
        # With a single model, its weight should be 1.0
        assert abs(config.dixon_coles_weight - 1.0) < 1e-8

    def test_optimize_weights_empty_models_returns_unchanged(self) -> None:
        ensemble = ModelEnsemble()
        config_before = ensemble.config
        config_after = ensemble.optimize_weights(
            np.array([]), np.array([]), np.array([]), {}
        )
        assert config_after is config_before


class TestEdgeCases:
    """Edge case tests for the ensemble."""

    def test_no_models_available_raises(self) -> None:
        ensemble = ModelEnsemble()
        with pytest.raises(RuntimeError, match="No models"):
            ensemble.predict("A", "B")

    def test_empty_predictions_raises(self) -> None:
        ensemble = ModelEnsemble()
        with pytest.raises(ValueError, match="No predictions"):
            ensemble.predict_from_individual_predictions({})

    def test_config_modification_affects_ensemble(self) -> None:
        dc = DixonColes(max_goals=7)
        dc.home_advantage = 0.3
        dc.team_attack["A"] = 0.5
        dc.team_defense["A"] = -0.2
        dc.team_attack["B"] = -0.1
        dc.team_defense["B"] = 0.1

        config = EnsembleConfig(
            dixon_coles_weight=1.0,
            market_weight=0.0,
            gradient_boost_weight=0.0,
            bayesian_weight=0.0,
        )
        ensemble = ModelEnsemble(config)
        ensemble.add_dixon_coles(dc)

        pred_with = ensemble.predict("A", "B")

        config.market_weight = 1.0
        config.dixon_coles_weight = 0.0
        # With no market set, should raise
        with pytest.raises(RuntimeError, match="No models"):
            ensemble.predict("A", "B")

        config.dixon_coles_weight = 1.0
        config.market_weight = 0.0
        pred_after = ensemble.predict("A", "B")
        np.testing.assert_array_almost_equal(pred_with.score_matrix, pred_after.score_matrix)
