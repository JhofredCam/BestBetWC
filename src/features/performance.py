"""Performance features from historical match statistics with temporal validation."""

from datetime import datetime

from sqlalchemy.orm import Session

from src.database.models import HeadToHead, Match, Team, TeamForm

FEATURE_NAMES = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "fifa_rank_home",
    "fifa_rank_away",
    "fifa_rank_diff",
    "form_pts_5_home",
    "form_pts_5_away",
    "form_pts_10_home",
    "form_pts_10_away",
    "goals_scored_5_home",
    "goals_scored_5_away",
    "goals_conceded_5_home",
    "goals_conceded_5_away",
    "xg_home",
    "xg_away",
    "xga_home",
    "xga_away",
    "xg_diff",
    "xg_ratio_home",
    "home_performance_factor",
    "away_performance_factor",
    "h2h_wins_home",
    "h2h_draws",
    "h2h_wins_away",
    "h2h_avg_goals",
    "possession_diff",
    "shots_diff",
    "shots_on_target_diff",
]


def _result_to_points(result: str) -> int:
    mapping = {"W": 3, "D": 1, "L": 0}
    return mapping.get(result.upper().strip(), 0)


def _calculate_form(
    session: Session, team_id: int, before_date: datetime, last_n: int,
) -> tuple[float, float, float]:
    entries = (
        session.query(TeamForm)
        .join(Match, TeamForm.match_id == Match.id)
        .filter(
            TeamForm.team_id == team_id,
            Match.datetime < before_date,
        )
        .order_by(Match.datetime.desc())
        .limit(last_n)
        .all()
    )

    if not entries:
        return 0.0, 0.0, 0.0

    total_pts = sum(_result_to_points(e.result) for e in entries)
    total_gf = sum(e.goals_scored for e in entries)
    total_ga = sum(e.goals_conceded for e in entries)

    return (
        float(total_pts),
        float(total_gf),
        float(total_ga),
    )


def _get_recent_form_stats(
    session: Session, team_id: int, before_date: datetime, last_n: int,
) -> tuple[float, float, float, float | None, float | None, float | None, float | None]:
    entries = (
        session.query(TeamForm)
        .join(Match, TeamForm.match_id == Match.id)
        .filter(
            TeamForm.team_id == team_id,
            Match.datetime < before_date,
        )
        .order_by(Match.datetime.desc())
        .limit(last_n)
        .all()
    )

    if not entries:
        return 0.0, 0.0, 0.0, None, None, None, None

    pts = sum(_result_to_points(e.result) for e in entries) / max(len(entries), 1)
    gf = sum(e.goals_scored for e in entries) / max(len(entries), 1)
    ga = sum(e.goals_conceded for e in entries) / max(len(entries), 1)

    xg_vals = [e.xg for e in entries if e.xg is not None]
    xga_vals = [e.xga for e in entries if e.xga is not None]
    possession_vals = [e.possession for e in entries if e.possession is not None]
    shots_vals = [e.shots for e in entries if e.shots is not None]

    avg_xg = float(sum(xg_vals) / len(xg_vals)) if xg_vals else None
    avg_xga = float(sum(xga_vals) / len(xga_vals)) if xga_vals else None
    avg_possession = float(sum(possession_vals) / len(possession_vals)) if possession_vals else None
    avg_shots = float(sum(shots_vals) / len(shots_vals)) if shots_vals else None

    return pts, gf, ga, avg_xg, avg_xga, avg_possession, avg_shots


def _calculate_h2h(
    session: Session, team_a_id: int, team_b_id: int,
) -> tuple[int, int, int, float | None]:
    h2h = (
        session.query(HeadToHead)
        .filter(
            (
                (HeadToHead.team_a_id == team_a_id) & (HeadToHead.team_b_id == team_b_id)
            )
            | (
                (HeadToHead.team_a_id == team_b_id) & (HeadToHead.team_b_id == team_a_id)
            )
        )
        .all()
    )

    wins_a = 0
    wins_b = 0
    draws = 0
    total_goals = 0

    for h in h2h:
        if h.team_a_id == team_a_id:
            if h.goals_a > h.goals_b:
                wins_a += 1
            elif h.goals_b > h.goals_a:
                wins_b += 1
            else:
                draws += 1
        else:
            if h.goals_b > h.goals_a:
                wins_a += 1
            elif h.goals_a > h.goals_b:
                wins_b += 1
            else:
                draws += 1
        total_goals += h.goals_a + h.goals_b

    num_matches = wins_a + wins_b + draws
    avg_goals = total_goals / num_matches if num_matches > 0 else None

    return wins_a, draws, wins_b, avg_goals


def get_performance_features(
    session: Session, match: Match,
) -> dict[str, float]:
    if not isinstance(match, Match):
        raise TypeError("match must be a Match instance")

    home_team = session.query(Team).filter(Team.id == match.home_team_id).first()
    away_team = session.query(Team).filter(Team.id == match.away_team_id).first()

    result: dict[str, float] = {
        "elo_home": float(home_team.elo_rating) if home_team and home_team.elo_rating else 0.0,
        "elo_away": float(away_team.elo_rating) if away_team and away_team.elo_rating else 0.0,
        "elo_diff": 0.0,
        "fifa_rank_home": float(home_team.fifa_rank) if home_team and home_team.fifa_rank else 0.0,
        "fifa_rank_away": float(away_team.fifa_rank) if away_team and away_team.fifa_rank else 0.0,
        "fifa_rank_diff": 0.0,
        "form_pts_5_home": 0.0,
        "form_pts_5_away": 0.0,
        "form_pts_10_home": 0.0,
        "form_pts_10_away": 0.0,
        "goals_scored_5_home": 0.0,
        "goals_scored_5_away": 0.0,
        "goals_conceded_5_home": 0.0,
        "goals_conceded_5_away": 0.0,
        "xg_home": 0.0,
        "xg_away": 0.0,
        "xga_home": 0.0,
        "xga_away": 0.0,
        "xg_diff": 0.0,
        "xg_ratio_home": 0.0,
        "home_performance_factor": 1.0,
        "away_performance_factor": 1.0,
        "h2h_wins_home": 0,
        "h2h_draws": 0,
        "h2h_wins_away": 0,
        "h2h_avg_goals": 0.0,
        "possession_diff": 0.0,
        "shots_diff": 0.0,
        "shots_on_target_diff": 0.0,
    }

    if home_team and away_team and home_team.elo_rating and away_team.elo_rating:
        result["elo_diff"] = home_team.elo_rating - away_team.elo_rating

    if home_team and away_team and home_team.fifa_rank and away_team.fifa_rank:
        result["fifa_rank_diff"] = float(away_team.fifa_rank - home_team.fifa_rank)

    if home_team and away_team:
        form_5_h = _calculate_form(session, home_team.id, match.datetime, 5)
        form_5_a = _calculate_form(session, away_team.id, match.datetime, 5)
        form_10_h = _calculate_form(session, home_team.id, match.datetime, 10)
        form_10_a = _calculate_form(session, away_team.id, match.datetime, 10)

        result["form_pts_5_home"] = form_5_h[0]
        result["form_pts_5_away"] = form_5_a[0]
        result["form_pts_10_home"] = form_10_h[0]
        result["form_pts_10_away"] = form_10_a[0]
        result["goals_scored_5_home"] = form_5_h[1]
        result["goals_scored_5_away"] = form_5_a[1]
        result["goals_conceded_5_home"] = form_5_h[2]
        result["goals_conceded_5_away"] = form_5_a[2]

        stats_h = _get_recent_form_stats(session, home_team.id, match.datetime, 5)
        stats_a = _get_recent_form_stats(session, away_team.id, match.datetime, 5)

        if stats_h[3] is not None:
            result["xg_home"] = stats_h[3]
        if stats_a[3] is not None:
            result["xg_away"] = stats_a[3]
        if stats_h[4] is not None:
            result["xga_home"] = stats_h[4]
        if stats_a[4] is not None:
            result["xga_away"] = stats_a[4]

        if stats_h[5] is not None and stats_a[5] is not None:
            result["possession_diff"] = stats_h[5] - stats_a[5]
        if stats_h[6] is not None and stats_a[6] is not None:
            result["shots_diff"] = stats_h[6] - stats_a[6]

        if result["xg_home"] is not None and result["xga_away"] is not None:
            denom = result["xg_home"] + result["xga_away"]
            result["xg_ratio_home"] = result["xg_home"] / denom if denom > 0 else 0.0
        if result["xg_home"] is not None and result["xg_away"] is not None:
            result["xg_diff"] = result["xg_home"] - result["xg_away"]

        h2h_wins_h, h2h_draws, h2h_wins_a, h2h_avg = _calculate_h2h(
            session, home_team.id, away_team.id,
        )
        result["h2h_wins_home"] = h2h_wins_h
        result["h2h_draws"] = h2h_draws
        result["h2h_wins_away"] = h2h_wins_a
        if h2h_avg is not None:
            result["h2h_avg_goals"] = h2h_avg

    return result
