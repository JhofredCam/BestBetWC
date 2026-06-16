"""Tests for GradientBoostModel (SPEC-009)."""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.features import MatchFeatureVector
from src.models.gradient_boost import (
    GBModelConfig,
    GradientBoostModel,
    ResidualGBModel,
    calibrate_model,
    temporal_train_test_split,
)


def _make_synthetic_features(n_samples: int, rng: np.random.Generator | None = None) -> tuple[list[MatchFeatureVector], np.ndarray, np.ndarray]:
    """Generate synthetic MatchFeatureVectors with predictable goal patterns.

    Uses elo_diff as primary signal. Returns (features, y_home, y_away).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    base_date = datetime(2018, 6, 14)
    features: list[MatchFeatureVector] = []
    y_home: list[int] = []
    y_away: list[int] = []

    for i in range(n_samples):
        elo_diff = rng.uniform(-2.0, 2.0)
        xg_diff = elo_diff + rng.normal(0.0, 0.1)

        hg = max(0, min(7, int(round(2.0 + 0.5 * elo_diff + rng.normal(0.0, 0.6)))))
        ag = max(0, min(7, int(round(1.5 - 0.3 * elo_diff + rng.normal(0.0, 0.6)))))

        fv = MatchFeatureVector(
            match_id=i + 1,
            timestamp=base_date + timedelta(days=i),
            market_home_prob=0.4 + 0.05 * elo_diff,
            market_draw_prob=0.3,
            market_away_prob=0.3 - 0.05 * elo_diff,
            elo_diff=elo_diff,
            xg_diff=xg_diff,
            xg_home=1.0 + 0.2 * elo_diff,
            xg_away=1.0 - 0.2 * elo_diff,
            home_performance_factor=1.0,
            away_performance_factor=1.0,
        )
        features.append(fv)
        y_home.append(hg)
        y_away.append(ag)

    return features, np.array(y_home), np.array(y_away)


def _features_to_array(features: list[MatchFeatureVector]) -> np.ndarray:
    return np.stack([f.to_array() for f in features], axis=0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_gb_model_config_defaults() -> None:
    config = GBModelConfig()
    assert config.n_estimators == 200
    assert config.max_depth == 5
    assert config.learning_rate == 0.05
    assert config.objective == "multi:softprob"
    assert config.early_stopping_rounds == 20


def test_gb_model_config_custom() -> None:
    config = GBModelConfig(n_estimators=50, max_depth=3, learning_rate=0.1)
    assert config.n_estimators == 50
    assert config.max_depth == 3


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------

def test_temporal_split_chronological() -> None:
    features, _, _ = _make_synthetic_features(20)
    train, test = temporal_train_test_split(features, train_ratio=0.8)
    assert len(train) == 16
    assert len(test) == 4
    # All train timestamps must be <= all test timestamps
    assert max(f.timestamp for f in train) <= min(f.timestamp for f in test)


def test_temporal_split_empty_returns_empty() -> None:
    train, test = temporal_train_test_split([], train_ratio=0.8)
    assert train == []
    assert test == []


def test_temporal_split_single_sample() -> None:
    features, _, _ = _make_synthetic_features(1)
    train, test = temporal_train_test_split(features, train_ratio=0.8)
    assert len(train) == 0
    assert len(test) == 1


# ---------------------------------------------------------------------------
# fit / predict_score_distribution
# ---------------------------------------------------------------------------

def test_fit_returns_training_history() -> None:
    features, y_home, y_away = _make_synthetic_features(60)
    X = _features_to_array(features)

    model = GradientBoostModel(GBModelConfig(n_estimators=20, max_depth=3))
    history = model.fit(X, y_home, y_away)

    assert "home" in history
    assert "away" in history


def test_fit_and_predict_distribution_sums_to_one() -> None:
    features, y_home, y_away = _make_synthetic_features(60)
    X = _features_to_array(features)

    model = GradientBoostModel(GBModelConfig(n_estimators=30, max_depth=4))
    model.fit(X, y_home, y_away)

    home_probs, away_probs = model.predict_score_distribution(X)
    assert home_probs.shape == (60, 8)
    assert away_probs.shape == (60, 8)

    for i in range(len(X)):
        assert abs(home_probs[i].sum() - 1.0) < 1e-5, f"Home row {i} sum = {home_probs[i].sum()}"
        assert abs(away_probs[i].sum() - 1.0) < 1e-5, f"Away row {i} sum = {away_probs[i].sum()}"


def test_fit_with_early_stopping() -> None:
    features, y_home, y_away = _make_synthetic_features(80)
    X = _features_to_array(features)

    X_train = X[:50]
    X_val = X[50:]
    y_home_train = y_home[:50]
    y_home_val = y_home[50:]
    y_away_train = y_away[:50]
    y_away_val = y_away[50:]

    model = GradientBoostModel(
        GBModelConfig(n_estimators=500, max_depth=4, early_stopping_rounds=10, learning_rate=0.1)
    )
    history = model.fit(
        X_train, y_home_train, y_away_train,
        eval_set=(X_val, y_home_val, y_away_val),
    )

    # Check that early stopping worked (n_estimators used < configured max)
    actual_trees = model.model_home.get_booster().num_boosted_rounds()
    assert actual_trees < 500, f"Early stopping should reduce trees, got {actual_trees}"

    assert "home" in history


# ---------------------------------------------------------------------------
# predict_match
# ---------------------------------------------------------------------------

def test_predict_match_returns_valid_match_prediction() -> None:
    features, y_home, y_away = _make_synthetic_features(60)
    X = _features_to_array(features)

    model = GradientBoostModel(GBModelConfig(n_estimators=30, max_depth=4))
    model.fit(X, y_home, y_away)

    pred = model.predict_match(features[0])
    assert pred.score_matrix.shape == (8, 8)
    assert abs(pred.score_matrix.sum() - 1.0) < 1e-5
    assert abs(pred.home_goals_dist.sum() - 1.0) < 1e-5
    assert abs(pred.away_goals_dist.sum() - 1.0) < 1e-5
    assert 0 <= pred.home_win_prob <= 1
    assert 0 <= pred.draw_prob <= 1
    assert 0 <= pred.away_win_prob <= 1
    # Home/away/draw should sum to 1
    assert abs(pred.home_win_prob + pred.draw_prob + pred.away_win_prob - 1.0) < 1e-5


def test_predict_match_with_custom_max_goals() -> None:
    features, y_home, y_away = _make_synthetic_features(60)
    X = _features_to_array(features)

    model = GradientBoostModel(GBModelConfig(n_estimators=30, max_depth=4), max_goals=7)
    model.fit(X, y_home, y_away)

    pred_5 = model.predict_match(features[0], max_goals=5)
    assert pred_5.score_matrix.shape == (6, 6)
    assert abs(pred_5.score_matrix.sum() - 1.0) < 1e-5

    pred_9 = model.predict_match(features[0], max_goals=9)
    assert pred_9.score_matrix.shape == (10, 10)
    assert abs(pred_9.score_matrix.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------------------

def test_feature_importance_two_features_coherent() -> None:
    """Model trained with only 2 meaningful features shows coherent importance."""
    rng = np.random.default_rng(42)
    n = 100
    elo_diff = rng.uniform(-2, 2, size=n)
    xg_signal = rng.uniform(-2, 2, size=n)
    noise_feat = rng.normal(0, 0.1, size=n)

    # Goals depend mainly on elo_diff, moderately on xg_signal
    hg = np.clip(np.round(2.0 + 0.6 * elo_diff + 0.2 * xg_signal + rng.normal(0, 0.4, size=n)), 0, 7).astype(int)
    ag = np.clip(np.round(1.5 - 0.4 * elo_diff + 0.1 * noise_feat + rng.normal(0, 0.4, size=n)), 0, 7).astype(int)

    # Build 3-feature matrix
    X = np.column_stack([elo_diff, xg_signal, noise_feat])

    model = GradientBoostModel(GBModelConfig(n_estimators=50, max_depth=3, learning_rate=0.1))
    model._feature_names = ["elo_diff", "xg_signal", "noise_feat"]
    model.fit(X, hg, ag)

    importance = model.get_feature_importance()
    assert "elo_diff" in importance
    assert "xg_signal" in importance
    assert "noise_feat" in importance

    # elo_diff should be most important for this synthetic data
    assert importance["elo_diff"] > importance["noise_feat"], (
        f"elo_diff importance {importance['elo_diff']:.3f} should exceed "
        f"noise_feat {importance['noise_feat']:.3f}"
    )


def test_feature_importance_raises_before_fit() -> None:
    model = GradientBoostModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.get_feature_importance()


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_predictions() -> None:
    features, y_home, y_away = _make_synthetic_features(60)
    X = _features_to_array(features)

    config = GBModelConfig(n_estimators=20, max_depth=3, learning_rate=0.1)
    model = GradientBoostModel(config)
    model.fit(X, y_home, y_away)

    home_before, away_before = model.predict_score_distribution(X[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = f"{tmpdir}/gb_model"
        model.save(base_path)

        model2 = GradientBoostModel()
        model2.load(base_path)

    home_after, away_after = model2.predict_score_distribution(X[:10])

    assert np.allclose(home_before, home_after, atol=1e-6)
    assert np.allclose(away_before, away_after, atol=1e-6)


def test_save_raises_before_fit() -> None:
    model = GradientBoostModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.save("/tmp/dummy")


# ---------------------------------------------------------------------------
# calibrate_model
# ---------------------------------------------------------------------------

def test_calibrate_model_runs_without_error() -> None:
    features, y_home, y_away = _make_synthetic_features(80)
    X = _features_to_array(features)

    model = GradientBoostModel(GBModelConfig(n_estimators=20, max_depth=3))
    X_train, X_cal = X[:40], X[40:]
    y_home_train, y_home_cal = y_home[:40], y_home[40:]
    y_away_train, y_away_cal = y_away[:40], y_away[40:]

    model.fit(X_train, y_home_train, y_away_train)
    calibrated = calibrate_model(model, X_cal, y_home_cal, y_away_cal)

    # Should still predict valid distributions
    home_probs, away_probs = calibrated.predict_score_distribution(X_cal)
    assert home_probs.shape == (40, 8)
    assert np.allclose(home_probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# predict with unfitted model
# ---------------------------------------------------------------------------

def test_predict_distribution_raises_before_fit() -> None:
    model = GradientBoostModel()
    X = np.random.rand(5, 52)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_score_distribution(X)


def test_predict_match_raises_before_fit() -> None:
    model = GradientBoostModel()
    features, _, _ = _make_synthetic_features(1)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_match(features[0])


# ---------------------------------------------------------------------------
# GradientBoostModel initialization
# ---------------------------------------------------------------------------

def test_model_initialization() -> None:
    model = GradientBoostModel()
    assert model.max_goals == 7
    assert model.model_home is None
    assert model.model_away is None
    assert isinstance(model.config, GBModelConfig)


def test_model_initialization_custom() -> None:
    config = GBModelConfig(n_estimators=100)
    model = GradientBoostModel(config=config, max_goals=5)
    assert model.max_goals == 5
    assert model.config.n_estimators == 100


# ---------------------------------------------------------------------------
# ResidualGBModel
# ---------------------------------------------------------------------------

def test_residual_gb_model_fit_and_predict() -> None:
    features, y_home, y_away = _make_synthetic_features(80)
    X = _features_to_array(features)
    n_classes = 8  # max_goals=7

    # Create dummy dixon predictions (uniform as baseline)
    dixon_home = np.full((80, n_classes), 1.0 / n_classes)
    dixon_away = np.full((80, n_classes), 1.0 / n_classes)

    model = ResidualGBModel(GBModelConfig(n_estimators=20, max_depth=3))
    history = model.fit(X, dixon_home, dixon_away, y_home, y_away)

    assert "home" in history
    assert "away" in history

    home_probs, away_probs = model.predict_score_distribution(X, dixon_home, dixon_away)
    assert home_probs.shape == (80, 8)
    assert away_probs.shape == (80, 8)
    assert np.allclose(home_probs.sum(axis=1), 1.0)
    assert np.allclose(away_probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fit_with_empty_data() -> None:
    model = GradientBoostModel(GBModelConfig(n_estimators=5))
    X = np.empty((0, 52))
    y = np.empty(0)
    with pytest.raises(Exception):
        model.fit(X, y, y)


def test_fit_with_high_goal_values_clipped() -> None:
    features, _, _ = _make_synthetic_features(40)
    X = _features_to_array(features)
    y_home_high = np.array([10, 15, 20, 8, 9] * 8)
    y_away_high = np.array([8, 12, 7, 9, 11] * 8)

    model = GradientBoostModel(GBModelConfig(n_estimators=10, max_depth=2))
    model.fit(X, y_home_high, y_away_high)

    home_probs, away_probs = model.predict_score_distribution(X)
    # max_goals=7, so class index 7 catches all clipped values
    assert home_probs.shape == (40, 8)


def test_predict_distribution_single_sample() -> None:
    features, y_home, y_away = _make_synthetic_features(30)
    X = _features_to_array(features)

    model = GradientBoostModel(GBModelConfig(n_estimators=15, max_depth=3))
    model.fit(X, y_home, y_away)

    X_single = features[0].to_array().reshape(1, -1)
    home_probs, away_probs = model.predict_score_distribution(X_single)

    assert home_probs.shape == (1, 8)
    assert away_probs.shape == (1, 8)


def test_temporal_split_preserves_order() -> None:
    """Temporal split must not shuffle or randomize."""
    features, _, _ = _make_synthetic_features(50)
    # Sort by timestamp explicitly
    features.sort(key=lambda f: f.timestamp)
    timestamps_before = [f.timestamp for f in features]

    train, test = temporal_train_test_split(features, train_ratio=0.7)
    assert len(train) == 35
    assert len(test) == 15

    # Train part comes first chronologically
    assert [f.timestamp for f in train] == timestamps_before[:35]
    assert [f.timestamp for f in test] == timestamps_before[35:]
