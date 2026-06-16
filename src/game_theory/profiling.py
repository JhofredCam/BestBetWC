"""Player profiling with archetype detection and bias estimation.

Profiles participants based on historical predictions to estimate their
playing style (conservative, aggressive, market-follower, intuition) and
specific biases (favorite_bias, home_bias, draw_aversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from src.database.models import Participant, ParticipantPrediction, ParticipantProfile


class PlayerArchetype(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    MARKET_FOLLOWER = "market_follower"
    INTUITION = "intuition"
    HOMER = "homer"


@dataclass
class PlayerProfile:
    participant_id: int
    name: str
    conservative_score: float
    aggressive_score: float
    market_follower_score: float
    intuition_score: float
    favorite_bias: float = 0.0
    home_bias: float = 0.0
    draw_aversion: float = 0.0
    exact_score_freq: float = 0.0
    avg_goals_predicted: float = 0.0
    popular_team_score: dict[str, float] = field(default_factory=dict)
    underdog_team_score: dict[str, float] = field(default_factory=dict)
    total_predictions: int = 0
    result_accuracy: float = 0.0
    exact_accuracy: float = 0.0
    avg_points_per_match: float = 0.0
    predicted_matches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "conservative_score": self.conservative_score,
            "aggressive_score": self.aggressive_score,
            "market_follower_score": self.market_follower_score,
            "intuition_score": self.intuition_score,
            "favorite_bias": self.favorite_bias,
            "home_bias": self.home_bias,
            "draw_aversion": self.draw_aversion,
            "exact_score_freq": self.exact_score_freq,
            "avg_goals_predicted": self.avg_goals_predicted,
            "popular_team_score": self.popular_team_score,
            "underdog_team_score": self.underdog_team_score,
            "total_predictions": self.total_predictions,
            "result_accuracy": self.result_accuracy,
            "exact_accuracy": self.exact_accuracy,
            "avg_points_per_match": self.avg_points_per_match,
        }

    @property
    def dominant_archetype(self) -> PlayerArchetype:
        scores = {
            PlayerArchetype.CONSERVATIVE: self.conservative_score,
            PlayerArchetype.AGGRESSIVE: self.aggressive_score,
            PlayerArchetype.MARKET_FOLLOWER: self.market_follower_score,
            PlayerArchetype.INTUITION: self.intuition_score,
        }
        return max(scores, key=lambda k: scores[k])  # type: ignore[return-value]

    @classmethod
    def uniform(cls, participant_id: int, name: str) -> PlayerProfile:
        return cls(
            participant_id=participant_id,
            name=name,
            conservative_score=0.25,
            aggressive_score=0.25,
            market_follower_score=0.25,
            intuition_score=0.25,
        )


class PlayerProfiler:
    def __init__(self, session: Session) -> None:
        self.session = session

    def profile_all(self) -> dict[int, PlayerProfile]:
        participants = self.session.query(Participant).all()
        profiles: dict[int, PlayerProfile] = {}
        for p in participants:
            profiles[p.id] = self.profile_participant(p.id)
        return profiles

    def profile_participant(self, participant_id: int) -> PlayerProfile:
        participant = (
            self.session.query(Participant).filter_by(id=participant_id).first()
        )
        if participant is None:
            raise ValueError(f"Participant {participant_id} not found")

        predictions = (
            self.session.query(ParticipantPrediction)
            .filter_by(participant_id=participant_id)
            .all()
        )

        if not predictions:
            return PlayerProfile.uniform(participant_id, participant.name)

        archetype_scores = self.calculate_archetype_scores(predictions)
        biases = self.calculate_biases(predictions)
        popular_teams, underdog_teams = self.detect_team_preferences(predictions)

        exact_count = 0
        total_goals = 0.0
        for pred in predictions:
            total_goals += pred.home_goals + pred.away_goals
            if pred.match and pred.match.home_score is not None:
                if (
                    pred.match.home_score == pred.home_goals
                    and pred.match.away_score == pred.away_goals
                ):
                    exact_count += 1

        n = len(predictions)
        exact_accuracy = exact_count / n if n > 0 else 0.0
        exact_score_freq = exact_accuracy
        avg_goals = total_goals / n if n > 0 else 0.0

        return PlayerProfile(
            participant_id=participant_id,
            name=participant.name,
            conservative_score=archetype_scores[0],
            aggressive_score=archetype_scores[1],
            market_follower_score=archetype_scores[2],
            intuition_score=archetype_scores[3],
            favorite_bias=biases.get("favorite_bias", 0.0),
            home_bias=biases.get("home_bias", 0.0),
            draw_aversion=biases.get("draw_aversion", 0.0),
            exact_score_freq=exact_score_freq,
            avg_goals_predicted=avg_goals,
            popular_team_score=popular_teams,
            underdog_team_score=underdog_teams,
            total_predictions=n,
            result_accuracy=0.0,
            exact_accuracy=exact_accuracy,
            avg_points_per_match=0.0,
        )

    def calculate_archetype_scores(
        self,
        predictions: list[ParticipantPrediction],
        match_contexts: list[dict[str, Any]] | None = None,
    ) -> tuple[float, float, float, float]:
        if not predictions:
            return (0.25, 0.25, 0.25, 0.25)

        n = len(predictions)
        low_score_home = sum(1 for p in predictions if p.home_goals <= 2)
        high_score_home = sum(1 for p in predictions if p.home_goals >= 3)
        low_score_away = sum(1 for p in predictions if p.away_goals <= 2)
        high_score_away = sum(1 for p in predictions if p.away_goals >= 3)

        home_win_count = sum(1 for p in predictions if p.home_goals > p.away_goals)
        away_win_count = sum(1 for p in predictions if p.away_goals > p.home_goals)
        draw_count = sum(1 for p in predictions if p.home_goals == p.away_goals)

        conservative_score = (
            (low_score_home / n) * 0.4
            + (low_score_away / n) * 0.3
            + (home_win_count / n) * 0.3
        )

        aggressive_score = (
            (high_score_home / n) * 0.3
            + (high_score_away / n) * 0.3
            + (away_win_count / n) * 0.4
        )

        intuition_score = 0.25

        market_follower_score = (
            (home_win_count / n) * 0.5 + (draw_count / n) * 0.25 + 0.25
        ) * 0.5

        total = conservative_score + aggressive_score + intuition_score + market_follower_score
        if total > 0:
            conservative_score /= total
            aggressive_score /= total
            intuition_score /= total
            market_follower_score /= total
        else:
            return (0.25, 0.25, 0.25, 0.25)

        return (
            float(conservative_score),
            float(aggressive_score),
            float(market_follower_score),
            float(intuition_score),
        )

    def calculate_biases(
        self, predictions: list[ParticipantPrediction]
    ) -> dict[str, float]:
        if not predictions:
            return {"favorite_bias": 0.0, "home_bias": 0.0, "draw_aversion": 0.0}

        n = len(predictions)
        home_win_pred = sum(1 for p in predictions if p.home_goals > p.away_goals)
        draw_pred = sum(1 for p in predictions if p.home_goals == p.away_goals)

        favorite_bias = home_win_pred / n
        home_bias = home_win_pred / n
        draw_aversion = 1.0 - (draw_pred / n)

        return {
            "favorite_bias": min(favorite_bias, 1.0),
            "home_bias": min(home_bias, 1.0),
            "draw_aversion": min(draw_aversion, 1.0),
        }

    def detect_team_preferences(
        self, predictions: list[ParticipantPrediction]
    ) -> tuple[dict[str, float], dict[str, float]]:
        popular: dict[str, float] = {}
        underdog: dict[str, float] = {}

        for pred in predictions:
            if pred.match is None:
                continue
            home_name = pred.match.home_team.name if pred.match.home_team else "unknown"
            away_name = pred.match.away_team.name if pred.match.away_team else "unknown"

            if pred.home_goals > pred.away_goals:
                popular[home_name] = popular.get(home_name, 0.0) + 1.0
                underdog[away_name] = underdog.get(away_name, 0.0) + 1.0
            elif pred.away_goals > pred.home_goals:
                popular[away_name] = popular.get(away_name, 0.0) + 1.0
                underdog[home_name] = underdog.get(home_name, 0.0) + 1.0
            else:
                popular[home_name] = popular.get(home_name, 0.0) + 0.5
                popular[away_name] = popular.get(away_name, 0.0) + 0.5

        n = len(predictions)
        if n > 0:
            for key in popular:
                popular[key] /= n
            for key in underdog:
                underdog[key] /= n

        return popular, underdog

    def update_profiles(self) -> None:
        profiles = self.profile_all()
        for p_id, profile in profiles.items():
            existing = (
                self.session.query(ParticipantProfile)
                .filter_by(participant_id=p_id)
                .first()
            )
            if existing is not None:
                existing.conservative_score = profile.conservative_score
                existing.aggressive_score = profile.aggressive_score
                existing.market_follower = profile.market_follower_score
                existing.favorite_bias = profile.favorite_bias
                existing.home_bias = profile.home_bias
                existing.recency_bias = 0.0
                existing.updated_at = datetime.now(UTC)
            else:
                db_profile = ParticipantProfile(
                    participant_id=p_id,
                    conservative_score=profile.conservative_score,
                    aggressive_score=profile.aggressive_score,
                    market_follower=profile.market_follower_score,
                    favorite_bias=profile.favorite_bias,
                    recency_bias=0.0,
                    home_bias=profile.home_bias,
                    updated_at=datetime.now(UTC),
                )
                self.session.add(db_profile)
        self.session.commit()
