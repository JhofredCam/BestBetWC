"""Ownership estimation and contrarian value calculation.

Estimates the probability that each scoreline will be chosen by other
participants in the prediction pool, enabling contrarian strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.database.models import Match
    from src.game_theory.profiling import PlayerProfile, PlayerProfiler


@dataclass
class OwnershipEstimate:
    score_matrix: np.ndarray
    most_popular_score: tuple[int, int]
    ownership_of_most_popular: float
    entropy: float
    unique_opportunities: list[tuple[int, int]]


class OwnershipEstimator:
    def __init__(self, profiler: PlayerProfiler) -> None:
        self.profiler = profiler
        self._rng = np.random.default_rng(42)

    def estimate(
        self,
        match: Match,
        team_home: str,
        team_away: str,
        market_probs: np.ndarray | None = None,
        max_goals: int = 7,
    ) -> OwnershipEstimate:
        profiles = self.profiler.profile_all()
        exclude_ids: set[int] = set()

        n_participants = len(profiles)
        if n_participants == 0:
            empty = np.zeros((max_goals + 1, max_goals + 1))
            return OwnershipEstimate(
                score_matrix=empty,
                most_popular_score=(1, 0),
                ownership_of_most_popular=0.0,
                entropy=0.0,
                unique_opportunities=[],
            )

        score_matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

        for p_id, profile in profiles.items():
            if p_id in exclude_ids:
                continue
            home_g, away_g = self._simulate_participant_prediction(
                profile, team_home, team_away, market_probs, max_goals
            )
            home_g = min(home_g, max_goals)
            away_g = min(away_g, max_goals)
            score_matrix[home_g, away_g] += 1.0

        total = score_matrix.sum()
        if total > 0:
            score_matrix = score_matrix / total

        max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        most_popular_score = (int(max_idx[0]), int(max_idx[1]))
        ownership_of_most_popular = float(score_matrix[max_idx])

        entropy = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = score_matrix[i, j]
                if p > 0:
                    entropy -= p * np.log(p)
        entropy = float(entropy)

        unique_opportunities: list[tuple[int, int]] = []
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if score_matrix[i, j] < 0.10 and score_matrix[i, j] > 0:
                    unique_opportunities.append((i, j))

        return OwnershipEstimate(
            score_matrix=score_matrix,
            most_popular_score=most_popular_score,
            ownership_of_most_popular=ownership_of_most_popular,
            entropy=entropy,
            unique_opportunities=unique_opportunities,
        )

    def _simulate_participant_prediction(
        self,
        profile: PlayerProfile,
        home_team: str,
        away_team: str,
        market_probs: np.ndarray | None,
        max_goals: int,
    ) -> tuple[int, int]:
        home_bias = profile.home_bias
        fav_bias = profile.favorite_bias
        conservative_w = profile.conservative_score
        aggressive_w = profile.aggressive_score

        sigma = 1.5
        if conservative_w > 0.5:
            sigma = 1.0
        elif aggressive_w > 0.5:
            sigma = 2.5

        home_expected = 1.5 + home_bias * 0.8
        away_expected = 1.0 - home_bias * 0.4

        popular_home = profile.popular_team_score.get(home_team, 0.0)
        underdog_home = profile.underdog_team_score.get(home_team, 0.0)
        popular_away = profile.popular_team_score.get(away_team, 0.0)
        underdog_away = profile.underdog_team_score.get(away_team, 0.0)

        home_expected += popular_home * 0.5 - underdog_home * 0.3
        away_expected += popular_away * 0.5 - underdog_away * 0.3

        home_expected = max(home_expected, 0.1)
        away_expected = max(away_expected, 0.1)

        home_goals = int(np.round(self._rng.normal(home_expected, sigma)))
        away_goals = int(np.round(self._rng.normal(away_expected, sigma)))

        home_goals = max(0, min(home_goals, max_goals))
        away_goals = max(0, min(away_goals, max_goals))

        if fav_bias > 0.6:
            home_goals = max(home_goals, away_goals)

        if profile.draw_aversion > 0.7:
            if self._rng.random() < 0.3 and home_goals == away_goals:
                if home_bias > 0.5:
                    home_goals += 1
                else:
                    away_goals += 1

        return home_goals, away_goals

    def estimate_batch(
        self,
        matches: list[Match],
        max_goals: int = 7,
    ) -> dict[int, OwnershipEstimate]:
        results: dict[int, OwnershipEstimate] = {}
        for m in matches:
            team_home = m.home_team.name if m.home_team else "unknown"
            team_away = m.away_team.name if m.away_team else "unknown"
            results[m.id] = self.estimate(m, team_home, team_away, max_goals=max_goals)
        return results

    def get_contrarian_value(
        self,
        estimate: OwnershipEstimate,
        model_probs: np.ndarray,
    ) -> np.ndarray:
        ownership = estimate.score_matrix
        shape = model_probs.shape
        result = np.zeros(shape)
        min_g = min(shape[0], ownership.shape[0])
        min_h = min(shape[1], ownership.shape[1])
        for i in range(min_g):
            for j in range(min_h):
                result[i, j] = model_probs[i, j] * (1.0 - ownership[i, j])
        result = np.clip(result, 0.0, 1.0)
        return result
