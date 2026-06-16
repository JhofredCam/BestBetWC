"""Context features: match importance, rest days, knockout status."""

from datetime import datetime

from sqlalchemy.orm import Session

from src.database.models import Match

FEATURE_NAMES = [
    "match_importance",
    "rest_days_home",
    "rest_days_away",
    "is_knockout",
    "round_weight",
    "must_win_home",
    "must_win_away",
    "already_qualified_home",
    "already_qualified_away",
]

KNOCKOUT_ROUNDS = {
    "round_of_16",
    "quarter",
    "quarter_final",
    "quarter-final",
    "semi",
    "semi_final",
    "semi-final",
    "final",
    "third_place",
}

ROUND_WEIGHTS: dict[str, float] = {
    "group": 1.0,
    "round_of_16": 2.0,
    "quarter": 3.0,
    "quarter_final": 3.0,
    "quarter-final": 3.0,
    "semi": 4.0,
    "semi_final": 4.0,
    "semi-final": 4.0,
    "third_place": 3.0,
    "final": 5.0,
}


def _calculate_rest_days(session: Session, team_id: int, before_date: datetime) -> int | None:
    last_match_query = (
        session.query(Match)
        .filter(
            (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
            Match.datetime < before_date,
        )
        .order_by(Match.datetime.desc())
        .first()
    )

    if last_match_query is None:
        return None

    delta = before_date - last_match_query.datetime
    return max(0, delta.days)


def _calculate_match_importance(match: Match, session: Session | None = None) -> float:
    round_lower = match.round.lower().replace(" ", "_")

    for ko_round in KNOCKOUT_ROUNDS:
        if ko_round in round_lower:
            return 1.0

    return 0.5


def _detect_knockout(match: Match) -> bool:
    round_lower = match.round.lower().replace(" ", "_")
    for ko_round in KNOCKOUT_ROUNDS:
        if ko_round in round_lower:
            return True
    return False


def _get_round_weight(match: Match) -> float:
    round_lower = match.round.lower().replace(" ", "_")
    for key, weight in ROUND_WEIGHTS.items():
        if key in round_lower:
            return weight
    return 1.0


def get_context_features(session: Session, match: Match) -> dict[str, float]:
    result: dict[str, float] = {
        "match_importance": _calculate_match_importance(match, session),
        "rest_days_home": 0.0,
        "rest_days_away": 0.0,
        "is_knockout": 0.0,
        "round_weight": _get_round_weight(match),
        "must_win_home": 0.0,
        "must_win_away": 0.0,
        "already_qualified_home": 0.0,
        "already_qualified_away": 0.0,
    }

    result["is_knockout"] = 1.0 if _detect_knockout(match) else 0.0

    rest_h = _calculate_rest_days(session, match.home_team_id, match.datetime)
    rest_a = _calculate_rest_days(session, match.away_team_id, match.datetime)

    result["rest_days_home"] = float(rest_h) if rest_h is not None else 0.0
    result["rest_days_away"] = float(rest_a) if rest_a is not None else 0.0

    return result
