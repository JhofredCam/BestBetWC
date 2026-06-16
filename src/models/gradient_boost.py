"""Gradient Boosting Model for score distribution prediction.

Implements XGBoost-based models that predict goal distributions using
complete feature vectors (market + performance + context features).
"""

# ruff: noqa: N803, N806  # X/y naming is standard in ML code

from dataclasses import asdict, dataclass

import numpy as np
import xgboost as xgb

from src.features import FEATURE_COLUMNS, MatchFeatureVector
from src.models.dixon_coles import MatchPrediction


@dataclass
class GBModelConfig:
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int = 20
    random_state: int = 42


def temporal_train_test_split(
    matches: list[MatchFeatureVector],
    train_ratio: float = 0.8,
) -> tuple[list[MatchFeatureVector], list[MatchFeatureVector]]:
    """Split temporal, no aleatorio.

    Los primeros train_ratio% de partidos cronologicamente son train,
    el resto es test.
    """
    sorted_matches = sorted(matches, key=lambda m: m.timestamp)
    split_idx = int(len(sorted_matches) * train_ratio)
    return sorted_matches[:split_idx], sorted_matches[split_idx:]


def _make_score_matrix(
    home_probs: np.ndarray,
    away_probs: np.ndarray,
) -> np.ndarray:
    """Build score matrix from marginal distributions assuming independence."""
    matrix = np.outer(home_probs, away_probs)
    return np.asarray(matrix / matrix.sum())


def _build_match_prediction(
    home_probs: np.ndarray,
    away_probs: np.ndarray,
) -> MatchPrediction:
    """Build MatchPrediction from marginal goal distributions.

    Args:
        home_probs: P(home_goals=k) for k in [0, max_goals], shape (max_goals+1,)
        away_probs: P(away_goals=k) for k in [0, max_goals], shape (max_goals+1,)

    Returns:
        MatchPrediction with score_matrix computed via independence assumption.
    """
    score_matrix = _make_score_matrix(home_probs, away_probs)

    home_win_prob = float(np.tril(score_matrix, -1).sum())
    draw_prob = float(np.trace(score_matrix))
    away_win_prob = float(np.triu(score_matrix, 1).sum())

    n_home = len(home_probs)
    n_away = len(away_probs)
    expected_home_goals = float(np.dot(np.arange(n_home), home_probs))
    expected_away_goals = float(np.dot(np.arange(n_away), away_probs))

    max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    most_likely_score = (int(max_idx[0]), int(max_idx[1]))
    most_likely_score_prob = float(score_matrix[max_idx])

    return MatchPrediction(
        home_goals_dist=home_probs,
        away_goals_dist=away_probs,
        score_matrix=score_matrix,
        home_win_prob=home_win_prob,
        draw_prob=draw_prob,
        away_win_prob=away_win_prob,
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        most_likely_score=most_likely_score,
        most_likely_score_prob=most_likely_score_prob,
    )


class GradientBoostModel:
    """XGBoost wrapper that predicts home/away goal distributions.

    Trains two independent XGBoost classifiers:
    - model_home: predicts P(home_goals = k | features) for k in [0, max_goals]
    - model_away: predicts P(away_goals = k | features) for k in [0, max_goals]

    The joint score distribution is computed as outer product (independence).
    """

    def __init__(self, config: GBModelConfig | None = None, max_goals: int = 7) -> None:
        self.config = config or GBModelConfig()
        self.max_goals = max_goals
        self.model_home: xgb.XGBClassifier | None = None
        self.model_away: xgb.XGBClassifier | None = None
        self._home_history: dict[str, list[float]] = {}
        self._away_history: dict[str, list[float]] = {}
        self._feature_names: list[str] = list(FEATURE_COLUMNS)

    def _clip_goals(self, goals: np.ndarray) -> np.ndarray:
        return np.clip(goals, 0, self.max_goals).astype(int)

    def _build_xgb_params(self) -> dict:
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "objective": "multi:softprob",
            "eval_metric": self.config.eval_metric,
            "num_class": self.max_goals + 1,
            "random_state": self.config.random_state,
            "verbosity": 0,
        }

    def fit(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        sample_weights: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        """Train two independent XGBoost classifiers for home and away goal distributions.

        Args:
            X: Feature matrix, shape (n_matches, n_features).
            y_home: Home goals per match, shape (n_matches,).
            y_away: Away goals per match, shape (n_matches,).
            sample_weights: Optional sample weights, shape (n_matches,).
            eval_set: Optional validation set as (X_val, y_val_home, y_val_away).

        Returns:
            Training history dict with 'home' and 'away' keys, each containing
            a dict of metric -> list of values per iteration.
        """
        y_home_c = self._clip_goals(y_home)
        y_away_c = self._clip_goals(y_away)

        params = self._build_xgb_params()
        n_estimators = params.pop("n_estimators")
        eval_metric = params.pop("eval_metric")
        random_state = params.pop("random_state")

        use_early_stop = eval_set is not None

        # ---- Home model ----
        home_ctor_kw = dict(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            random_state=random_state,
            **params,
        )
        if use_early_stop:
            home_ctor_kw["early_stopping_rounds"] = self.config.early_stopping_rounds

        self.model_home = xgb.XGBClassifier(**home_ctor_kw)

        fit_kwargs_home: dict = {"sample_weight": sample_weights, "verbose": False}
        if eval_set is not None:
            X_val, y_val_home, _y_val_away = eval_set
            y_val_home_c = self._clip_goals(y_val_home)
            fit_kwargs_home["eval_set"] = [(X_val, y_val_home_c)]

        self.model_home.fit(X, y_home_c, **fit_kwargs_home)

        # ---- Away model ----
        away_ctor_kw = dict(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            random_state=random_state,
            **params,
        )
        if use_early_stop:
            away_ctor_kw["early_stopping_rounds"] = self.config.early_stopping_rounds

        self.model_away = xgb.XGBClassifier(**away_ctor_kw)

        fit_kwargs_away: dict = {"sample_weight": sample_weights, "verbose": False}
        if eval_set is not None:
            X_val, _y_val_home, y_val_away = eval_set
            y_val_away_c = self._clip_goals(y_val_away)
            fit_kwargs_away["eval_set"] = [(X_val, y_val_away_c)]

        self.model_away.fit(X, y_away_c, **fit_kwargs_away)

        # ---- History ----
        try:
            self._home_history = dict(self.model_home.evals_result().get("validation_0", {}))
        except Exception:
            self._home_history = {}
        try:
            self._away_history = dict(self.model_away.evals_result().get("validation_0", {}))
        except Exception:
            self._away_history = {}

        return {"home": self._home_history, "away": self._away_history}

    def predict_score_distribution(
        self, X: np.ndarray, max_goals: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict marginal goal distributions.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            max_goals: Override max_goals (pads/truncates output).

        Returns:
            Tuple of (home_probs, away_probs), each shape (n_samples, max_goals+1).
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mg = max_goals if max_goals is not None else self.max_goals

        home_raw = self.model_home.predict_proba(X)
        away_raw = self.model_away.predict_proba(X)

        home_probs = _resize_probs(home_raw, self.max_goals + 1, mg + 1)
        away_probs = _resize_probs(away_raw, self.max_goals + 1, mg + 1)

        return home_probs, away_probs

    def predict_match(
        self, features: MatchFeatureVector, max_goals: int | None = None
    ) -> MatchPrediction:
        """Predict match outcome from a single MatchFeatureVector.

        Args:
            features: Feature vector for the match.
            max_goals: Override max_goals dimension.

        Returns:
            MatchPrediction with full score matrix and derived stats.
        """
        X = features.to_array().reshape(1, -1)
        mg = max_goals if max_goals is not None else self.max_goals
        home_probs, away_probs = self.predict_score_distribution(X, max_goals=mg)
        return _build_match_prediction(home_probs[0], away_probs[0])

    def get_feature_importance(self) -> dict[str, float]:
        """Average feature importance across home and away models (gain metric).

        Returns:
            Dict mapping feature name -> importance score. Features with zero
            total importance are excluded.
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        home_imp: dict[str, float | list[float]] = self.model_home.get_booster().get_score(
            importance_type="gain"
        )
        away_imp: dict[str, float | list[float]] = self.model_away.get_booster().get_score(
            importance_type="gain"
        )

        merged: dict[str, float] = {}
        n_features = len(self._feature_names)

        for key, val in home_imp.items():
            idx = int(key.replace("f", ""))
            name = self._feature_names[idx] if idx < n_features else key
            val_float = float(val) if isinstance(val, (int, float)) else float(sum(val))
            merged[name] = merged.get(name, 0.0) + val_float

        for key, val in away_imp.items():
            idx = int(key.replace("f", ""))
            name = self._feature_names[idx] if idx < n_features else key
            val_float = float(val) if isinstance(val, (int, float)) else float(sum(val))
            merged[name] = merged.get(name, 0.0) + val_float

        if merged:
            total = sum(merged.values())
            if total > 0:
                for k in merged:
                    merged[k] /= total

        return merged

    def save(self, path: str) -> None:
        """Save models and config to disk.

        Produces three files: {path}_home.json, {path}_away.json, {path}_config.json.

        Args:
            path: Base path (without extension).
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model_home.save_model(f"{path}_home.json")
        self.model_away.save_model(f"{path}_away.json")

        import json

        config_data = asdict(self.config)
        config_data["max_goals"] = self.max_goals

        with open(f"{path}_config.json", "w") as f:
            json.dump(config_data, f)

    def load(self, path: str) -> None:
        """Load models and config from disk.

        Expects three files: {path}_home.json, {path}_away.json, {path}_config.json.

        Args:
            path: Base path (without extension).
        """
        import json

        with open(f"{path}_config.json") as f:
            config_data = json.load(f)

        self.max_goals = int(config_data.pop("max_goals"))
        self.config = GBModelConfig(**config_data)

        params = self._build_xgb_params()
        n_estimators = params.pop("n_estimators")
        eval_metric = params.pop("eval_metric")
        random_state = params.pop("random_state")

        self.model_home = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            random_state=random_state,
            **params,
        )
        self.model_home.load_model(f"{path}_home.json")

        self.model_away = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            random_state=random_state,
            **params,
        )
        self.model_away.load_model(f"{path}_away.json")


def _resize_probs(probs: np.ndarray, from_size: int, to_size: int) -> np.ndarray:
    """Pad or truncate probability arrays to desired number of classes.

    Args:
        probs: Probability array shape (n_samples, from_size).
        from_size: Current number of classes.
        to_size: Desired number of classes.

    Returns:
        Resized probability array shape (n_samples, to_size), normalized per sample.
    """
    if from_size == to_size:
        return probs.copy()
    elif from_size > to_size:
        truncated = probs[:, :to_size].copy()
    else:
        truncated = np.pad(probs, ((0, 0), (0, to_size - from_size)), mode="constant")
    row_sums = truncated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return np.asarray(truncated / row_sums)


def calibrate_model(
    model: GradientBoostModel,
    X_cal: np.ndarray,
    y_cal_home: np.ndarray,
    y_cal_away: np.ndarray,
) -> GradientBoostModel:
    """Post-calibrate probabilities using Platt scaling (sigmoid).

    Wraps each internal XGBClassifier with CalibratedClassifierCV (method='sigmoid').
    This improves probability calibration for better expected score estimates.

    Note: After calibration, save/load will only persist base models, not
    the calibration layer. Re-run calibrate_model after loading if needed.

    Args:
        model: Fitted GradientBoostModel.
        X_cal: Calibration features, shape (n_samples, n_features).
        y_cal_home: Calibration targets (home goals).
        y_cal_away: Calibration targets (away goals).

    Returns:
        The same model instance with calibrated classifiers.
    """
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]

    if model.model_home is None or model.model_away is None:
        raise RuntimeError("Model not fitted. Call fit() first.")

    y_home_c = model._clip_goals(y_cal_home)
    y_away_c = model._clip_goals(y_cal_away)

    cal_home: CalibratedClassifierCV = CalibratedClassifierCV(
        estimator=model.model_home, method="sigmoid", cv="prefit"
    )
    cal_home.fit(X_cal, y_home_c)
    model.model_home = cal_home

    cal_away: CalibratedClassifierCV = CalibratedClassifierCV(
        estimator=model.model_away, method="sigmoid", cv="prefit"
    )
    cal_away.fit(X_cal, y_away_c)
    model.model_away = cal_away

    return model


class ResidualGBModel:
    """XGBoost as a corrector of Dixon-Coles predictions.

    Trains on residuals between actual outcomes and Dixon-Coles probability
    predictions. The final prediction is:

        final = dixon_preds + xgb_residual (clipped and renormalized)

    This approach is less prone to overfitting than a standalone model
    because it only needs to learn the systematic errors of Dixon-Coles.
    """

    def __init__(self, config: GBModelConfig | None = None, max_goals: int = 7) -> None:
        self.config = config or GBModelConfig()
        self.max_goals = max_goals
        self.model_home: xgb.XGBClassifier | None = None
        self.model_away: xgb.XGBClassifier | None = None
        self._home_history: dict[str, list[float]] = {}
        self._away_history: dict[str, list[float]] = {}

    def _one_hot(self, goals: np.ndarray, n_classes: int) -> np.ndarray:
        encoded = np.zeros((len(goals), n_classes))
        for i, g in enumerate(np.clip(goals, 0, n_classes - 1).astype(int)):
            encoded[i, g] = 1.0
        return encoded

    def _clip_goals(self, goals: np.ndarray) -> np.ndarray:
        return np.clip(goals, 0, self.max_goals).astype(int)

    def _build_xgb_params(self) -> dict:
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "objective": "multi:softprob",
            "eval_metric": self.config.eval_metric,
            "num_class": self.max_goals + 1,
            "random_state": self.config.random_state,
            "verbosity": 0,
        }

    def fit(
        self,
        X: np.ndarray,
        dixon_preds_home: np.ndarray,
        dixon_preds_away: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
    ) -> dict[str, dict[str, list[float]]]:
        """Train residual correctors.

        Computes residuals: one_hot(y_true) - dixon_preds.
        XGBoost learns to predict residuals from features.
        Final prediction combines dixon + xgb_residual, clipped and renormalized.

        Args:
            X: Feature matrix, shape (n_matches, n_features).
            dixon_preds_home: Dixon-Coles home probabilities, shape (n_matches, max_goals+1).
            dixon_preds_away: Dixon-Coles away probabilities, shape (n_matches, max_goals+1).
            y_home: Actual home goals, shape (n_matches,).
            y_away: Actual away goals, shape (n_matches,).

        Returns:
            Training history dict.
        """
        n_classes = self.max_goals + 1
        y_home_c = self._clip_goals(y_home)
        y_away_c = self._clip_goals(y_away)

        # Compute residuals: one_hot(y) - dixon_preds
        resid_home = self._one_hot(y_home_c, n_classes) - dixon_preds_home
        resid_away = self._one_hot(y_away_c, n_classes) - dixon_preds_away

        # Convert residuals to class targets by finding max residual class
        # (which class benefits most from correction)
        resid_home_targets = np.argmax(resid_home, axis=1)
        resid_away_targets = np.argmax(resid_away, axis=1)

        params = self._build_xgb_params()
        n_estimators = params.pop("n_estimators")
        eval_metric = params.pop("eval_metric")
        random_state = params.pop("random_state")

        self.model_home = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            random_state=random_state,
            **params,
        )
        self.model_home.fit(X, resid_home_targets, verbose=False)

        self.model_away = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            random_state=random_state,
            **params,
        )
        self.model_away.fit(X, resid_away_targets, verbose=False)

        try:
            self._home_history = dict(
                self.model_home.evals_result().get("validation_0", {})
            )
        except Exception:
            self._home_history = {}
        try:
            self._away_history = dict(
                self.model_away.evals_result().get("validation_0", {})
            )
        except Exception:
            self._away_history = {}

        return {"home": self._home_history, "away": self._away_history}

    def predict_score_distribution(
        self,
        X: np.ndarray,
        dixon_preds_home: np.ndarray,
        dixon_preds_away: np.ndarray,
        max_goals: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict corrected score distributions.

        final = dixon_preds + xgb_residual (clipped and renormalized).

        Args:
            X: Feature matrix.
            dixon_preds_home: Dixon-Coles home predictions, shape (n_samples, max_goals+1).
            dixon_preds_away: Dixon-Coles away predictions, shape (n_samples, max_goals+1).

        Returns:
            Tuple of (home_probs, away_probs), corrected distributions.
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mg = max_goals if max_goals is not None else self.max_goals

        # XGBoost residual predictions (probabilities over correction classes)
        resid_home_pred = self.model_home.predict_proba(X)
        resid_away_pred = self.model_away.predict_proba(X)

        # Pad/truncate dixon and residual to max_goals
        resid_home_pred = _resize_probs(resid_home_pred, self.max_goals + 1, mg + 1)
        resid_away_pred = _resize_probs(resid_away_pred, self.max_goals + 1, mg + 1)
        dixon_home = _resize_probs(dixon_preds_home, self.max_goals + 1, mg + 1)
        dixon_away = _resize_probs(dixon_preds_away, self.max_goals + 1, mg + 1)

        # Blend: final = dixon + alpha * residual, renormalize
        alpha = 0.3  # weight for residual correction
        home_corrected = dixon_home + alpha * resid_home_pred
        away_corrected = dixon_away + alpha * resid_away_pred

        home_corrected = np.clip(home_corrected, 0.0, None)
        away_corrected = np.clip(away_corrected, 0.0, None)

        home_sum = home_corrected.sum(axis=1, keepdims=True)
        away_sum = away_corrected.sum(axis=1, keepdims=True)
        home_corrected = home_corrected / np.where(home_sum == 0, 1.0, home_sum)
        away_corrected = away_corrected / np.where(away_sum == 0, 1.0, away_sum)

        return home_corrected, away_corrected
