from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import poisson  # type: ignore[import-untyped]


@dataclass
class MatchPrediction:
    home_goals_dist: np.ndarray
    away_goals_dist: np.ndarray
    score_matrix: np.ndarray
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    most_likely_score: tuple[int, int]
    most_likely_score_prob: float


class DixonColes:
    def __init__(self, max_goals: int = 7) -> None:
        self.max_goals = max_goals
        self.home_advantage: float = 0.0
        self.rho: float = -0.13
        self.team_attack: dict[str, float] = {}
        self.team_defense: dict[str, float] = {}

    def _tau(self, x: int, y: int, lambda_h: float, mu_a: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1 - lambda_h * mu_a * rho
        elif x == 0 and y == 1:
            return 1 + lambda_h * rho
        elif x == 1 and y == 0:
            return 1 + mu_a * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def _score_probability(
        self, home_goals: int, away_goals: int, lambda_h: float, mu_a: float, rho: float
    ) -> float:
        p_home = float(poisson.pmf(home_goals, lambda_h))
        p_away = float(poisson.pmf(away_goals, mu_a))
        tau = self._tau(home_goals, away_goals, lambda_h, mu_a, rho)
        return p_home * p_away * tau

    def _expected_goals(self, team_home: str, team_away: str) -> tuple[float, float]:
        attack_home = self.team_attack.get(team_home, 1.0)
        defense_home = self.team_defense.get(team_home, 1.0)
        attack_away = self.team_attack.get(team_away, 1.0)
        defense_away = self.team_defense.get(team_away, 1.0)

        lambda_h = np.exp(self.home_advantage + attack_home - defense_away)
        mu_a = np.exp(attack_away - defense_home)

        return lambda_h, mu_a

    def predict_match(self, team_home: str, team_away: str) -> MatchPrediction:
        lambda_h, mu_a = self._expected_goals(team_home, team_away)

        score_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                score_matrix[i, j] = self._score_probability(i, j, lambda_h, mu_a, self.rho)

        score_matrix = score_matrix / score_matrix.sum()

        home_goals_dist = score_matrix.sum(axis=1)
        away_goals_dist = score_matrix.sum(axis=0)

        home_win_prob = np.tril(score_matrix, -1).sum()
        draw_prob = np.trace(score_matrix)
        away_win_prob = np.triu(score_matrix, 1).sum()

        max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        most_likely_score = (int(max_idx[0]), int(max_idx[1]))
        most_likely_score_prob = float(score_matrix[max_idx])

        return MatchPrediction(
            home_goals_dist=home_goals_dist,
            away_goals_dist=away_goals_dist,
            score_matrix=score_matrix,
            home_win_prob=float(home_win_prob),
            draw_prob=float(draw_prob),
            away_win_prob=float(away_win_prob),
            expected_home_goals=float(lambda_h),
            expected_away_goals=float(mu_a),
            most_likely_score=most_likely_score,
            most_likely_score_prob=most_likely_score_prob,
        )

    def fit(self, matches: list[dict[str, int | str]]) -> None:
        teams_set: set[str] = set()
        for m in matches:
            teams_set.add(str(m["home_team"]))
            teams_set.add(str(m["away_team"]))

        teams = sorted(teams_set)
        n_teams = len(teams)

        self.team_attack = {t: 0.0 for t in teams}
        self.team_defense = {t: 0.0 for t in teams}

        params = np.zeros(1 + 2 * n_teams)

        def neg_log_likelihood(params: np.ndarray) -> float:
            home_adv = params[0]
            attack = params[1 : n_teams + 1]
            defense = params[n_teams + 1 :]

            ll = 0.0
            for m in matches:
                home_idx = teams.index(str(m["home_team"]))
                away_idx = teams.index(str(m["away_team"]))
                home_goals = int(m["home_goals"])
                away_goals = int(m["away_goals"])

                lambda_h = np.exp(home_adv + attack[home_idx] - defense[away_idx])
                mu_a = np.exp(attack[away_idx] - defense[home_idx])

                lambda_h = max(lambda_h, 0.01)
                mu_a = max(mu_a, 0.01)

                p = self._score_probability(home_goals, away_goals, lambda_h, mu_a, self.rho)
                p = max(p, 1e-10)
                ll += np.log(p)

            return -ll

        result = minimize(neg_log_likelihood, params, method="L-BFGS-B")

        self.home_advantage = result.x[0]
        for i, team in enumerate(teams):
            self.team_attack[team] = result.x[1 + i]
            self.team_defense[team] = result.x[1 + n_teams + i]

    def predict_from_params(
        self, lambda_h: float, mu_a: float, rho: float | None = None
    ) -> MatchPrediction:
        if rho is None:
            rho = self.rho

        score_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                score_matrix[i, j] = self._score_probability(i, j, lambda_h, mu_a, rho)

        score_matrix = score_matrix / score_matrix.sum()

        home_goals_dist = score_matrix.sum(axis=1)
        away_goals_dist = score_matrix.sum(axis=0)

        home_win_prob = float(np.tril(score_matrix, -1).sum())
        draw_prob = float(np.trace(score_matrix))
        away_win_prob = float(np.triu(score_matrix, 1).sum())

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
            expected_home_goals=float(lambda_h),
            expected_away_goals=float(mu_a),
            most_likely_score=most_likely_score,
            most_likely_score_prob=most_likely_score_prob,
        )
