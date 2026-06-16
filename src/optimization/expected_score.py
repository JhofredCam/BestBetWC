from dataclasses import dataclass

import numpy as np

from src.config import POLLA_RULES, PollaRules
from src.models.dixon_coles import MatchPrediction


@dataclass
class ExpectedScoreResult:
    home_goals: int
    away_goals: int
    ep_total: float
    ep_exact: float
    ep_result: float
    ep_goals_home: float
    ep_goals_away: float
    ep_unique: float
    prob_exact: float
    prob_result: float
    prob_goals_home: float
    prob_goals_away: float
    ownership_estimate: float
    contrarian_value: float


class ExpectedScoreCalculator:
    def __init__(self, rules: PollaRules | None = None) -> None:
        self.rules = rules or POLLA_RULES

    def calculate_ep(
        self,
        prediction: MatchPrediction,
        pred_home: int,
        pred_away: int,
        ownership_estimate: float = 0.0,
    ) -> ExpectedScoreResult:
        score_matrix = prediction.score_matrix
        home_goals_dist = prediction.home_goals_dist
        away_goals_dist = prediction.away_goals_dist

        prob_exact = float(score_matrix[pred_home, pred_away])

        if pred_home > pred_away:
            prob_result = float(np.tril(score_matrix, -1).sum()) - prob_exact
        elif pred_home == pred_away:
            prob_result = float(np.trace(score_matrix)) - prob_exact
        else:
            prob_result = float(np.triu(score_matrix, 1).sum()) - prob_exact

        prob_goals_home = float(home_goals_dist[pred_home])
        prob_goals_away = float(away_goals_dist[pred_away])

        n_others = self.rules.num_participants - 1
        prob_unique = (1 - ownership_estimate) ** n_others

        ep_exact = prob_exact * self.rules.exact_score_pts
        ep_result = prob_result * self.rules.result_correct_pts
        ep_goals_home = prob_goals_home * self.rules.goals_home_correct_pts
        ep_goals_away = prob_goals_away * self.rules.goals_away_correct_pts
        ep_unique = prob_exact * prob_unique * self.rules.unique_prediction_bonus

        ep_total = ep_exact + ep_result + ep_goals_home + ep_goals_away + ep_unique

        contrarian_value = prob_exact * (1 - ownership_estimate)

        return ExpectedScoreResult(
            home_goals=pred_home,
            away_goals=pred_away,
            ep_total=ep_total,
            ep_exact=ep_exact,
            ep_result=ep_result,
            ep_goals_home=ep_goals_home,
            ep_goals_away=ep_goals_away,
            ep_unique=ep_unique,
            prob_exact=prob_exact,
            prob_result=prob_result,
            prob_goals_home=prob_goals_home,
            prob_goals_away=prob_goals_away,
            ownership_estimate=ownership_estimate,
            contrarian_value=contrarian_value,
        )

    def find_optimal_prediction(
        self,
        prediction: MatchPrediction,
        ownership_matrix: np.ndarray | None = None,
    ) -> ExpectedScoreResult:
        max_goals = prediction.score_matrix.shape[0]

        if ownership_matrix is None:
            ownership_matrix = np.zeros((max_goals, max_goals))

        best_ep = -1.0
        best_result: ExpectedScoreResult | None = None

        for i in range(max_goals):
            for j in range(max_goals):
                ownership = float(ownership_matrix[i, j]) if ownership_matrix is not None else 0.0
                result = self.calculate_ep(prediction, i, j, ownership)

                if result.ep_total > best_ep:
                    best_ep = result.ep_total
                    best_result = result

        assert best_result is not None
        return best_result

    def rank_all_predictions(
        self,
        prediction: MatchPrediction,
        ownership_matrix: np.ndarray | None = None,
    ) -> list[ExpectedScoreResult]:
        max_goals = prediction.score_matrix.shape[0]

        if ownership_matrix is None:
            ownership_matrix = np.zeros((max_goals, max_goals))

        results = []
        for i in range(max_goals):
            for j in range(max_goals):
                ownership = float(ownership_matrix[i, j])
                result = self.calculate_ep(prediction, i, j, ownership)
                results.append(result)

        results.sort(key=lambda r: r.ep_total, reverse=True)
        return results
