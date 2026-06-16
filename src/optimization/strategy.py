from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.config import STRATEGY_CONFIG, StrategyConfig
from src.models.dixon_coles import MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator, ExpectedScoreResult


class StrategyMode(Enum):
    MINIMIZE_RISK = "minimize_risk"
    BALANCED = "balanced"
    DIFFERENTIATION = "differentiation"
    HIGH_RISK = "high_risk"


@dataclass
class StrategyRecommendation:
    prediction: ExpectedScoreResult
    strategy_mode: StrategyMode
    reasoning: str
    risk_score: float
    upside_potential: float
    risk_of_ruin: float


class StrategySelector:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or STRATEGY_CONFIG
        self.ep_calculator = ExpectedScoreCalculator()

    def determine_mode(self, current_position: int, total_participants: int) -> StrategyMode:
        if current_position <= self.config.leading_threshold:
            return StrategyMode.MINIMIZE_RISK
        elif self.config.middle_range[0] <= current_position <= self.config.middle_range[1]:
            return StrategyMode.BALANCED
        elif self.config.behind_range[0] <= current_position <= self.config.behind_range[1]:
            return StrategyMode.DIFFERENTIATION
        else:
            return StrategyMode.HIGH_RISK

    def get_recommendation(
        self,
        prediction: MatchPrediction,
        current_position: int,
        total_participants: int,
        ownership_matrix: np.ndarray | None = None,
    ) -> StrategyRecommendation:
        mode = self.determine_mode(current_position, total_participants)

        max_goals = prediction.score_matrix.shape[0]
        if ownership_matrix is None:
            ownership_matrix = np.zeros((max_goals, max_goals))

        all_predictions = self.ep_calculator.rank_all_predictions(
            prediction, ownership_matrix
        )

        if mode == StrategyMode.MINIMIZE_RISK:
            selected = self._select_minimize_risk(all_predictions, prediction)
            reasoning = "Liderando: minimizando riesgo con predicción de alta probabilidad"
            risk_score = 0.2
            upside_potential = selected.ep_total
            risk_of_ruin = 1 - selected.prob_result - selected.prob_exact

        elif mode == StrategyMode.BALANCED:
            selected = self._select_balanced(all_predictions)
            reasoning = "Posición media: estrategia equilibrada entre seguridad y diferenciación"
            risk_score = 0.5
            upside_potential = selected.ep_total * 1.2
            risk_of_ruin = 1 - selected.prob_result - selected.prob_exact

        elif mode == StrategyMode.DIFFERENTIATION:
            selected = self._select_differentiation(all_predictions, prediction, ownership_matrix)
            reasoning = "Atrás en la tabla: buscando diferenciación con predicciones contrarian"
            risk_score = 0.7
            upside_potential = selected.ep_total * 1.5
            risk_of_ruin = 1 - selected.prob_result - selected.prob_exact

        else:
            selected = self._select_high_risk(all_predictions, prediction, ownership_matrix)
            reasoning = "Últimas posiciones: alto riesgo para maximizar potencial de remontada"
            risk_score = 0.9
            upside_potential = selected.ep_total * 2.0
            risk_of_ruin = 1 - selected.prob_result - selected.prob_exact

        return StrategyRecommendation(
            prediction=selected,
            strategy_mode=mode,
            reasoning=reasoning,
            risk_score=risk_score,
            upside_potential=upside_potential,
            risk_of_ruin=risk_of_ruin,
        )

    def _select_minimize_risk(
        self, predictions: list[ExpectedScoreResult], match_pred: MatchPrediction
    ) -> ExpectedScoreResult:
        high_prob = [p for p in predictions if p.prob_result + p.prob_exact > 0.3]
        if high_prob:
            return max(high_prob, key=lambda p: p.ep_total)
        return predictions[0]

    def _select_balanced(
        self, predictions: list[ExpectedScoreResult]
    ) -> ExpectedScoreResult:
        top_5 = predictions[:5]
        return max(top_5, key=lambda p: p.ep_total)

    def _select_differentiation(
        self,
        predictions: list[ExpectedScoreResult],
        match_pred: MatchPrediction,
        ownership_matrix: np.ndarray,
    ) -> ExpectedScoreResult:
        contrarian = [
            p for p in predictions if p.ownership_estimate < 0.2 and p.prob_exact > 0.05
        ]
        if contrarian:
            return max(contrarian, key=lambda p: p.contrarian_value)
        return predictions[0]

    def _select_high_risk(
        self,
        predictions: list[ExpectedScoreResult],
        match_pred: MatchPrediction,
        ownership_matrix: np.ndarray,
    ) -> ExpectedScoreResult:
        high_upside = [
            p for p in predictions if p.ownership_estimate < 0.1 and p.prob_exact > 0.02
        ]
        if high_upside:
            return max(high_upside, key=lambda p: p.contrarian_value * p.ep_total)
        return predictions[0]
