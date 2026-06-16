"""Opponent modeling - predicts what each opponent will choose.

Extends PlayerProfiler with forward-looking prediction of opponent
scoreline choices for upcoming matches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.database.models import Match
    from src.game_theory.profiling import PlayerProfiler


class OpponentModel:
    def __init__(self, profiler: PlayerProfiler) -> None:
        self.profiler = profiler
        self._rng = np.random.default_rng(42)

    def predict_opponent_choice(
        self,
        participant_id: int,
        match: Match,
        team_home: str,
        team_away: str,
        max_goals: int = 7,
    ) -> tuple[int, int]:
        profile = self.profiler.profile_participant(participant_id)

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

        popular_home = profile.popular_team_score.get(team_home, 0.0)
        underdog_home = profile.underdog_team_score.get(team_home, 0.0)
        popular_away = profile.popular_team_score.get(team_away, 0.0)
        underdog_away = profile.underdog_team_score.get(team_away, 0.0)

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

    def predict_all_opponents(
        self,
        match: Match,
        team_home: str,
        team_away: str,
        exclude_participant: int | None = None,
        max_goals: int = 7,
    ) -> list[tuple[int, int]]:
        profiles = self.profiler.profile_all()
        results: list[tuple[int, int]] = []
        for p_id in profiles:
            if p_id == exclude_participant:
                continue
            choice = self.predict_opponent_choice(
                p_id, match, team_home, team_away, max_goals
            )
            results.append(choice)
        return results

    def build_ownership_matrix(
        self,
        match: Match,
        team_home: str,
        team_away: str,
        max_goals: int = 7,
    ) -> np.ndarray:
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        choices = self.predict_all_opponents(match, team_home, team_away)
        for home_g, away_g in choices:
            home_g = min(home_g, max_goals)
            away_g = min(away_g, max_goals)
            matrix[home_g, away_g] += 1.0
        total = matrix.sum()
        if total > 0:
            matrix = matrix / total
        return matrix
