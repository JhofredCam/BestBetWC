from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.models.dixon_coles import MatchPrediction


class StrategyMode(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    MARKET_FOLLOWER = "market_follower"
    RANDOM = "random"
    PERFECT = "perfect"


@dataclass
class SimulatedParticipant:
    name: str
    profile: Any | None = None
    strategy_mode: StrategyMode = StrategyMode.MARKET_FOLLOWER


class ParticipantSimulator:
    def __init__(self, seed: int | None = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate_predictions(
        self,
        participant: SimulatedParticipant,
        match_predictions: dict[str, MatchPrediction],
    ) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        for match_id, pred in match_predictions.items():
            if participant.strategy_mode == StrategyMode.CONSERVATIVE:
                result[match_id] = self._conservative_strategy(pred)
            elif participant.strategy_mode == StrategyMode.AGGRESSIVE:
                result[match_id] = self._aggressive_strategy(pred)
            elif participant.strategy_mode == StrategyMode.MARKET_FOLLOWER:
                market_probs = pred.score_matrix.copy()
                result[match_id] = self._market_follower_strategy(pred, market_probs)
            elif participant.strategy_mode == StrategyMode.RANDOM:
                result[match_id] = self._random_strategy(pred)
            elif participant.strategy_mode == StrategyMode.PERFECT:
                result[match_id] = pred.most_likely_score
            else:
                result[match_id] = pred.most_likely_score
        return result

    def simulate_all(
        self,
        participants: list[SimulatedParticipant],
        match_predictions: dict[str, MatchPrediction],
    ) -> dict[str, dict[str, tuple[int, int]]]:
        result: dict[str, dict[str, tuple[int, int]]] = {}
        for p in participants:
            result[p.name] = self.simulate_predictions(p, match_predictions)
        return result

    @staticmethod
    def _conservative_strategy(match_pred: MatchPrediction) -> tuple[int, int]:
        score_matrix = match_pred.score_matrix
        max_g = score_matrix.shape[0] - 1

        candidates: list[tuple[int, int]] = []
        max_prob = 0.0
        for h in range(min(3, max_g + 1)):
            for a in range(min(3, max_g + 1)):
                prob = float(score_matrix[h, a])
                if prob > max_prob:
                    max_prob = prob
                    candidates = [(h, a)]
                elif abs(prob - max_prob) < 1e-10:
                    candidates.append((h, a))

        if candidates:
            return candidates[0]

        return match_pred.most_likely_score

    @staticmethod
    def _aggressive_strategy(match_pred: MatchPrediction) -> tuple[int, int]:
        max_g = match_pred.score_matrix.shape[0] - 1

        if match_pred.home_win_prob > match_pred.away_win_prob:
            h_range = range(2, min(6, max_g + 1))
            a_range = range(0, min(3, max_g + 1))
        else:
            h_range = range(0, min(3, max_g + 1))
            a_range = range(2, min(6, max_g + 1))

        candidates: list[tuple[int, int]] = []
        max_prob = 0.0
        for h in h_range:
            for a in a_range:
                prob = float(match_pred.score_matrix[h, a])
                if prob > max_prob:
                    max_prob = prob
                    candidates = [(h, a)]
                elif abs(prob - max_prob) < 1e-10:
                    candidates.append((h, a))

        if candidates:
            return candidates[0]

        return match_pred.most_likely_score

    @staticmethod
    def _market_follower_strategy(
        match_pred: MatchPrediction,
        market_probs: np.ndarray,
    ) -> tuple[int, int]:
        flat = market_probs.flatten()
        flat = flat / flat.sum()
        rng = np.random.default_rng(42)
        idx = rng.choice(len(flat), p=flat)
        max_g = market_probs.shape[0] - 1
        h = idx // (max_g + 1)
        a = idx % (max_g + 1)
        return int(h), int(a)

    def _random_strategy(self, match_pred: MatchPrediction) -> tuple[int, int]:
        max_g = min(match_pred.score_matrix.shape[0] - 1, self._rng.integers(1, 5))
        h = int(self._rng.integers(0, max_g + 1))
        a = int(self._rng.integers(0, max_g + 1))
        return h, a

    @staticmethod
    def conservative_strategy(match_pred: MatchPrediction) -> tuple[int, int]:
        return ParticipantSimulator._conservative_strategy(match_pred)

    @staticmethod
    def aggressive_strategy(match_pred: MatchPrediction) -> tuple[int, int]:
        return ParticipantSimulator._aggressive_strategy(match_pred)

    @staticmethod
    def market_follower_strategy(
        match_pred: MatchPrediction,
        market_probs: np.ndarray | None = None,
    ) -> tuple[int, int]:
        probs = market_probs if market_probs is not None else match_pred.score_matrix
        return ParticipantSimulator._market_follower_strategy(match_pred, probs)

    @staticmethod
    def random_strategy(match_pred: MatchPrediction) -> tuple[int, int]:
        rng = np.random.default_rng(None)
        max_g = match_pred.score_matrix.shape[0] - 1
        h = int(rng.integers(0, max_g + 1))
        a = int(rng.integers(0, max_g + 1))
        return h, a
