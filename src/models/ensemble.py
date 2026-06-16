"""Model Ensemble: weighted combination of predictions from multiple models.

Combines Dixon-Coles, betting market, XGBoost, and future Bayesian model
predictions into a single unified MatchPrediction using configurable weights.
"""

# ruff: noqa: N803, N806  # X/y naming is standard in ML code

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from src.models.dixon_coles import DixonColes, MatchPrediction
from src.models.gradient_boost import GradientBoostModel


@dataclass
class EnsembleConfig:
    dixon_coles_weight: float = 0.35
    market_weight: float = 0.35
    gradient_boost_weight: float = 0.20
    bayesian_weight: float = 0.10
    calibration_samples: int = 1000


def _ensemble_score_matrices(
    matrices: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted average of score matrices.

    Only includes models with weight > 0.
    """
    if not matrices:
        raise ValueError("No score matrices provided for ensembling")

    reference = next(iter(matrices.values()))
    result = np.zeros_like(reference, dtype=float)
    total_weight = 0.0

    for name, matrix in matrices.items():
        w = weights.get(name, 0.0)
        if w > 0:
            result += w * matrix
            total_weight += w

    if total_weight > 0:
        result /= total_weight

    return result


def _build_match_prediction(score_matrix: np.ndarray) -> MatchPrediction:
    """Build a MatchPrediction from a score matrix.

    Computes marginals, win/draw probabilities, expected goals, and most likely score.
    """
    home_goals_dist = score_matrix.sum(axis=1)
    away_goals_dist = score_matrix.sum(axis=0)

    home_win_prob = float(np.tril(score_matrix, -1).sum())
    draw_prob = float(np.trace(score_matrix))
    away_win_prob = float(np.triu(score_matrix, 1).sum())

    n_goals = score_matrix.shape[0]
    expected_home_goals = float(np.dot(np.arange(n_goals), home_goals_dist))
    expected_away_goals = float(np.dot(np.arange(n_goals), away_goals_dist))

    max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    most_likely_score = (int(max_idx[0]), int(max_idx[1]))
    most_likely_score_prob = float(score_matrix[max_idx])

    return MatchPrediction(
        home_goals_dist=home_goals_dist,
        away_goals_dist=away_goals_dist,
        score_matrix=score_matrix,
        home_win_prob=home_win_prob,
        draw_prob=draw_prob,
        away_win_prob=away_win_prob,
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        most_likely_score=most_likely_score,
        most_likely_score_prob=most_likely_score_prob,
    )


class ModelEnsemble:
    """Weighted ensemble of multiple prediction models.

    Combines Dixon-Coles, betting market, XGBoost, and Bayesian models
    using configurable weights to produce a unified MatchPrediction.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()

        self._dixon_coles: DixonColes | None = None
        self._gradient_boost: GradientBoostModel | None = None
        self._bayesian: Any = None

        self._market_score_matrix: np.ndarray | None = None
        self._market_home_dist: np.ndarray | None = None
        self._market_away_dist: np.ndarray | None = None

    def add_dixon_coles(self, model: DixonColes) -> None:
        self._dixon_coles = model

    def add_gradient_boost(self, model: GradientBoostModel) -> None:
        self._gradient_boost = model

    def add_bayesian(self, model: Any) -> None:
        self._bayesian = model

    def set_market_prediction(
        self,
        score_matrix: np.ndarray,
        home_dist: np.ndarray,
        away_dist: np.ndarray,
    ) -> None:
        self._market_score_matrix = score_matrix.copy()
        self._market_home_dist = home_dist.copy()
        self._market_away_dist = away_dist.copy()

    def _collect_score_matrices(
        self,
        team_home: str,
        team_away: str,
        features: Any | None = None,
        market_score_matrix: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Collect score matrices from all available models."""
        matrices: dict[str, np.ndarray] = {}

        if self.config.dixon_coles_weight > 0 and self._dixon_coles is not None:
            pred = self._dixon_coles.predict_match(team_home, team_away)
            matrices["dixon_coles"] = pred.score_matrix

        if self.config.market_weight > 0:
            market_mat = (
                market_score_matrix
                if market_score_matrix is not None
                else self._market_score_matrix
            )
            if market_mat is not None:
                matrices["market"] = market_mat

        if self.config.gradient_boost_weight > 0 and self._gradient_boost is not None:
            if features is not None:
                pred = self._gradient_boost.predict_match(features)
                matrices["gradient_boost"] = pred.score_matrix

        if self.config.bayesian_weight > 0 and self._bayesian is not None:
            raise NotImplementedError("Bayesian model integration not yet implemented")

        return matrices

    def predict(
        self,
        team_home: str,
        team_away: str,
        features: Any | None = None,
        market_score_matrix: np.ndarray | None = None,
    ) -> MatchPrediction:
        """Ensemble prediction from all available models.

        Algorithm:
        1. Obtain score_matrix from each available model
        2. Weight by EnsembleConfig weights
        3. Normalize the resulting matrix
        4. Compute marginal distributions
        5. Return MatchPrediction
        """
        matrices = self._collect_score_matrices(
            team_home, team_away, features, market_score_matrix
        )

        if not matrices:
            raise RuntimeError(
                "No models available for ensembling. Add at least one model "
                "or provide a market_score_matrix."
            )

        weights = self.get_weights_dict()
        ensembled = _ensemble_score_matrices(matrices, weights)
        return _build_match_prediction(ensembled)

    def predict_from_individual_predictions(
        self, predictions: dict[str, MatchPrediction]
    ) -> MatchPrediction:
        """Ensemble already-computed predictions.

        Args:
            predictions: {"dixon_coles": pred1, "market": pred2, ...}

        Returns:
            MatchPrediction with ensembled score_matrix.
        """
        matrices: dict[str, np.ndarray] = {}
        for name, pred in predictions.items():
            matrices[name] = pred.score_matrix

        if not matrices:
            raise ValueError("No predictions provided for ensembling")

        weights = self.get_weights_dict()

        effective_weights: dict[str, float] = {}
        for name in matrices:
            effective_weights[name] = weights.get(name, 0.0)

        ensembled = _ensemble_score_matrices(matrices, effective_weights)
        return _build_match_prediction(ensembled)

    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val_home: np.ndarray,
        y_val_away: np.ndarray,
        model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> EnsembleConfig:
        """Find optimal weights minimizing Log Loss on validation set.

        Uses scipy.optimize.minimize with constraints (weights >= 0, sum to 1).

        Args:
            X_val: Feature matrix (unused for weight optimization, reserved for future).
            y_val_home: Actual home goals.
            y_val_away: Actual away goals.
            model_predictions: Dict mapping model name to tuple of
                (home_probs, away_probs), each shape (n_samples, max_goals+1).

        Returns:
            Optimized EnsembleConfig with updated weights.
        """
        model_names = sorted(model_predictions.keys())
        n_models = len(model_names)

        if n_models == 0:
            return self.config

        n_samples = len(y_val_home)

        def _continuous_score_for_sample(
            home_probs: np.ndarray, away_probs: np.ndarray, idx: int
        ) -> np.ndarray:
            """Build score matrix for a single sample."""
            return np.outer(home_probs[idx], away_probs[idx])

        def neg_log_likelihood(weights: np.ndarray) -> float:
            ll = 0.0
            for i in range(n_samples):
                ensembled = np.zeros_like(
                    np.outer(
                        model_predictions[model_names[0]][0][i],
                        model_predictions[model_names[0]][1][i],
                    )
                )
                total_w = 0.0
                for j, name in enumerate(model_names):
                    w = weights[j]
                    if w > 0:
                        h_probs, a_probs = model_predictions[name]
                        score_mat = _continuous_score_for_sample(h_probs, a_probs, i)
                        ensembled += w * score_mat
                        total_w += w
                if total_w > 0:
                    ensembled /= total_w

                h = int(np.clip(y_val_home[i], 0, ensembled.shape[0] - 1))
                a = int(np.clip(y_val_away[i], 0, ensembled.shape[1] - 1))
                p = max(ensembled[h, a], 1e-10)
                ll += np.log(p)

            return -ll

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        x0 = np.ones(n_models) / n_models

        result = minimize(
            neg_log_likelihood,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimized = result.x
        weight_map: dict[str, float] = {}
        for i, name in enumerate(model_names):
            weight_map[name] = float(optimized[i])

        return EnsembleConfig(
            dixon_coles_weight=weight_map.get("dixon_coles", 0.0),
            market_weight=weight_map.get("market", 0.0),
            gradient_boost_weight=weight_map.get("gradient_boost", 0.0),
            bayesian_weight=weight_map.get("bayesian", 0.0),
            calibration_samples=self.config.calibration_samples,
        )

    def get_weights_dict(self) -> dict[str, float]:
        """Return current weights for logging/reporting."""
        return {
            "dixon_coles": self.config.dixon_coles_weight,
            "market": self.config.market_weight,
            "gradient_boost": self.config.gradient_boost_weight,
            "bayesian": self.config.bayesian_weight,
        }

    def _validate_weights(self) -> None:
        """Verify that weights sum to ~1.0."""
        total = sum(self.get_weights_dict().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights sum to {total}, expected 1.0")
