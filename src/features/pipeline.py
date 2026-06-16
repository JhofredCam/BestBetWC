"""Feature engineering pipeline that produces MatchFeatureVector for ML models."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from src.database.models import Match
from src.features.context import FEATURE_NAMES as CONTEXT_NAMES
from src.features.context import get_context_features
from src.features.market import FEATURE_NAMES as MARKET_NAMES
from src.features.market import get_market_features
from src.features.performance import FEATURE_NAMES as PERFORMANCE_NAMES
from src.features.performance import get_performance_features

FEATURE_COLUMNS: list[str] = MARKET_NAMES + PERFORMANCE_NAMES + CONTEXT_NAMES


def _safe_float(value: float | None, default: float = 0.0) -> float:
    return value if value is not None else default


@dataclass
class MatchFeatureVector:
    match_id: int
    timestamp: datetime

    market_home_prob: float = 0.333
    market_draw_prob: float = 0.333
    market_away_prob: float = 0.333
    market_btts_yes_prob: float = 0.0
    market_over_15_prob: float = 0.0
    market_over_25_prob: float = 0.0
    market_over_35_prob: float = 0.0
    market_margin: float = 0.0
    market_num_bookmakers: int = 0
    market_disagreement: float = 0.0
    odds_movement_home: float = 0.0
    odds_movement_draw: float = 0.0
    odds_movement_away: float = 0.0
    correct_score_entropy: float = 0.0

    elo_home: float = 0.0
    elo_away: float = 0.0
    elo_diff: float = 0.0
    fifa_rank_home: float = 0.0
    fifa_rank_away: float = 0.0
    fifa_rank_diff: float = 0.0
    form_pts_5_home: float = 0.0
    form_pts_5_away: float = 0.0
    form_pts_10_home: float = 0.0
    form_pts_10_away: float = 0.0
    goals_scored_5_home: float = 0.0
    goals_scored_5_away: float = 0.0
    goals_conceded_5_home: float = 0.0
    goals_conceded_5_away: float = 0.0
    xg_home: float = 0.0
    xg_away: float = 0.0
    xga_home: float = 0.0
    xga_away: float = 0.0
    xg_diff: float = 0.0
    xg_ratio_home: float = 0.0
    home_performance_factor: float = 1.0
    away_performance_factor: float = 1.0
    h2h_wins_home: int = 0
    h2h_draws: int = 0
    h2h_wins_away: int = 0
    h2h_avg_goals: float = 0.0
    possession_diff: float = 0.0
    shots_diff: float = 0.0
    shots_on_target_diff: float = 0.0

    match_importance: float = 0.5
    rest_days_home: float = 0.0
    rest_days_away: float = 0.0
    is_knockout: bool = False
    round_weight: float = 1.0
    must_win_home: float = 0.0
    must_win_away: float = 0.0
    already_qualified_home: bool = False
    already_qualified_away: bool = False

    def to_array(self) -> np.ndarray:
        values = [
            self.market_home_prob,
            self.market_draw_prob,
            self.market_away_prob,
            _safe_float(self.market_btts_yes_prob),
            _safe_float(self.market_over_15_prob),
            _safe_float(self.market_over_25_prob),
            _safe_float(self.market_over_35_prob),
            self.market_margin,
            float(self.market_num_bookmakers),
            self.market_disagreement,
            self.odds_movement_home,
            self.odds_movement_draw,
            self.odds_movement_away,
            self.correct_score_entropy,
            _safe_float(self.elo_home),
            _safe_float(self.elo_away),
            _safe_float(self.elo_diff),
            _safe_float(self.fifa_rank_home),
            _safe_float(self.fifa_rank_away),
            _safe_float(self.fifa_rank_diff),
            self.form_pts_5_home,
            self.form_pts_5_away,
            self.form_pts_10_home,
            self.form_pts_10_away,
            self.goals_scored_5_home,
            self.goals_scored_5_away,
            self.goals_conceded_5_home,
            self.goals_conceded_5_away,
            _safe_float(self.xg_home),
            _safe_float(self.xg_away),
            _safe_float(self.xga_home),
            _safe_float(self.xga_away),
            _safe_float(self.xg_diff),
            _safe_float(self.xg_ratio_home),
            self.home_performance_factor,
            self.away_performance_factor,
            float(self.h2h_wins_home),
            float(self.h2h_draws),
            float(self.h2h_wins_away),
            _safe_float(self.h2h_avg_goals),
            _safe_float(self.possession_diff),
            _safe_float(self.shots_diff),
            _safe_float(self.shots_on_target_diff),
            self.match_importance,
            self.rest_days_home,
            self.rest_days_away,
            float(self.is_knockout),
            self.round_weight,
            self.must_win_home,
            self.must_win_away,
            float(self.already_qualified_home),
            float(self.already_qualified_away),
        ]
        return np.array(values, dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return {
            "market_home_prob": self.market_home_prob,
            "market_draw_prob": self.market_draw_prob,
            "market_away_prob": self.market_away_prob,
            "market_btts_yes_prob": _safe_float(self.market_btts_yes_prob),
            "market_over_15_prob": _safe_float(self.market_over_15_prob),
            "market_over_25_prob": _safe_float(self.market_over_25_prob),
            "market_over_35_prob": _safe_float(self.market_over_35_prob),
            "market_margin": self.market_margin,
            "market_num_bookmakers": float(self.market_num_bookmakers),
            "market_disagreement": self.market_disagreement,
            "odds_movement_home": self.odds_movement_home,
            "odds_movement_draw": self.odds_movement_draw,
            "odds_movement_away": self.odds_movement_away,
            "correct_score_entropy": self.correct_score_entropy,
            "elo_home": _safe_float(self.elo_home),
            "elo_away": _safe_float(self.elo_away),
            "elo_diff": _safe_float(self.elo_diff),
            "fifa_rank_home": _safe_float(self.fifa_rank_home),
            "fifa_rank_away": _safe_float(self.fifa_rank_away),
            "fifa_rank_diff": _safe_float(self.fifa_rank_diff),
            "form_pts_5_home": self.form_pts_5_home,
            "form_pts_5_away": self.form_pts_5_away,
            "form_pts_10_home": self.form_pts_10_home,
            "form_pts_10_away": self.form_pts_10_away,
            "goals_scored_5_home": self.goals_scored_5_home,
            "goals_scored_5_away": self.goals_scored_5_away,
            "goals_conceded_5_home": self.goals_conceded_5_home,
            "goals_conceded_5_away": self.goals_conceded_5_away,
            "xg_home": _safe_float(self.xg_home),
            "xg_away": _safe_float(self.xg_away),
            "xga_home": _safe_float(self.xga_home),
            "xga_away": _safe_float(self.xga_away),
            "xg_diff": _safe_float(self.xg_diff),
            "xg_ratio_home": _safe_float(self.xg_ratio_home),
            "home_performance_factor": self.home_performance_factor,
            "away_performance_factor": self.away_performance_factor,
            "h2h_wins_home": float(self.h2h_wins_home),
            "h2h_draws": float(self.h2h_draws),
            "h2h_wins_away": float(self.h2h_wins_away),
            "h2h_avg_goals": _safe_float(self.h2h_avg_goals),
            "possession_diff": _safe_float(self.possession_diff),
            "shots_diff": _safe_float(self.shots_diff),
            "shots_on_target_diff": _safe_float(self.shots_on_target_diff),
            "match_importance": self.match_importance,
            "rest_days_home": self.rest_days_home,
            "rest_days_away": self.rest_days_away,
            "is_knockout": float(self.is_knockout),
            "round_weight": self.round_weight,
            "must_win_home": self.must_win_home,
            "must_win_away": self.must_win_away,
            "already_qualified_home": float(self.already_qualified_home),
            "already_qualified_away": float(self.already_qualified_away),
        }


class FeaturePipeline:
    def __init__(self, session: Session) -> None:
        self.session = session

    def build_features(self, match_id: int) -> MatchFeatureVector:
        match = self.session.query(Match).filter(Match.id == match_id).first()
        if match is None:
            raise ValueError(f"Match with id {match_id} not found")

        market = self._get_market_features(match_id)
        performance = self._get_performance_features(match)
        context = self._get_context_features(match)

        return MatchFeatureVector(
            match_id=match.id,
            timestamp=match.datetime,
            market_home_prob=market["market_home_prob"],
            market_draw_prob=market["market_draw_prob"],
            market_away_prob=market["market_away_prob"],
            market_btts_yes_prob=market["market_btts_yes_prob"],
            market_over_15_prob=market["market_over_15_prob"],
            market_over_25_prob=market["market_over_25_prob"],
            market_over_35_prob=market["market_over_35_prob"],
            market_margin=market["market_margin"],
            market_num_bookmakers=int(market["market_num_bookmakers"]),
            market_disagreement=market["market_disagreement"],
            odds_movement_home=market["odds_movement_home"],
            odds_movement_draw=market["odds_movement_draw"],
            odds_movement_away=market["odds_movement_away"],
            correct_score_entropy=market["correct_score_entropy"],
            elo_home=performance["elo_home"],
            elo_away=performance["elo_away"],
            elo_diff=performance["elo_diff"],
            fifa_rank_home=performance["fifa_rank_home"],
            fifa_rank_away=performance["fifa_rank_away"],
            fifa_rank_diff=performance["fifa_rank_diff"],
            form_pts_5_home=performance["form_pts_5_home"],
            form_pts_5_away=performance["form_pts_5_away"],
            form_pts_10_home=performance["form_pts_10_home"],
            form_pts_10_away=performance["form_pts_10_away"],
            goals_scored_5_home=performance["goals_scored_5_home"],
            goals_scored_5_away=performance["goals_scored_5_away"],
            goals_conceded_5_home=performance["goals_conceded_5_home"],
            goals_conceded_5_away=performance["goals_conceded_5_away"],
            xg_home=performance["xg_home"],
            xg_away=performance["xg_away"],
            xga_home=performance["xga_home"],
            xga_away=performance["xga_away"],
            xg_diff=performance["xg_diff"],
            xg_ratio_home=performance["xg_ratio_home"],
            home_performance_factor=performance["home_performance_factor"],
            away_performance_factor=performance["away_performance_factor"],
            h2h_wins_home=int(performance["h2h_wins_home"]),
            h2h_draws=int(performance["h2h_draws"]),
            h2h_wins_away=int(performance["h2h_wins_away"]),
            h2h_avg_goals=performance["h2h_avg_goals"],
            possession_diff=performance["possession_diff"],
            shots_diff=performance["shots_diff"],
            shots_on_target_diff=performance["shots_on_target_diff"],
            match_importance=context["match_importance"],
            rest_days_home=context["rest_days_home"],
            rest_days_away=context["rest_days_away"],
            is_knockout=bool(context["is_knockout"]),
            round_weight=context["round_weight"],
            must_win_home=context["must_win_home"],
            must_win_away=context["must_win_away"],
            already_qualified_home=bool(context["already_qualified_home"]),
            already_qualified_away=bool(context["already_qualified_away"]),
        )

    def build_features_batch(self, match_ids: list[int]) -> list[MatchFeatureVector]:
        return [self.build_features(mid) for mid in match_ids]

    def _get_market_features(self, match_id: int) -> dict[str, float]:
        return get_market_features(self.session, match_id)

    def _get_performance_features(self, match: Match) -> dict[str, float]:
        return get_performance_features(self.session, match)

    def _get_context_features(self, match: Match) -> dict[str, float]:
        return get_context_features(self.session, match)
