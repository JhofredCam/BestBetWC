from datetime import datetime

import pytest
from sqlalchemy import text

from src.database.connection import Base, get_engine, get_session, initialize_database
from src.database.models import (
    CorrectScoreOdds,
    HeadToHead,
    Injury,
    Match,
    Odds,
    Participant,
    ParticipantPrediction,
    ParticipantProfile,
    Score,
    Standing,
    SystemPrediction,
    Team,
    TeamForm,
)


@pytest.fixture(autouse=True)
def setup_database() -> None:
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def test_engine_creation() -> None:
    engine = get_engine()
    assert engine is not None
    result = engine.execute(text("SELECT 1"))
    assert result.scalar() == 1


def test_session_creation() -> None:
    session = get_session()
    assert session is not None
    session.close()


def test_insert_team_and_match() -> None:
    session = get_session()
    try:
        team_home = Team(
            name="Argentina",
            fifa_code="ARG",
            confederation="CONMEBOL",
            elo_rating=1850.0,
            fifa_rank=1,
            group="A",
        )
        team_away = Team(
            name="Brasil",
            fifa_code="BRA",
            confederation="CONMEBOL",
            elo_rating=1840.0,
            fifa_rank=3,
            group="B",
        )
        session.add_all([team_home, team_away])
        session.flush()

        match = Match(
            home_team_id=team_home.id,
            away_team_id=team_away.id,
            datetime=datetime(2026, 6, 15, 20, 0),
            venue="Estadio Monumental",
            city="Buenos Aires",
            round="group",
            group="A",
            status="scheduled",
            home_score=None,
            away_score=None,
        )
        session.add(match)
        session.commit()

        queried_team = session.query(Team).filter_by(name="Argentina").first()
        assert queried_team is not None
        assert queried_team.fifa_code == "ARG"
        assert queried_team.confederation == "CONMEBOL"

        queried_match = session.query(Match).first()
        assert queried_match is not None
        assert queried_match.home_team.name == "Argentina"
        assert queried_match.away_team.name == "Brasil"
        assert queried_match.round == "group"
    finally:
        session.close()


def test_match_odds_relationship() -> None:
    session = get_session()
    try:
        team_home = Team(
            name="Alemania",
            fifa_code="GER",
            confederation="UEFA",
            elo_rating=1800.0,
            fifa_rank=5,
        )
        team_away = Team(
            name="Francia",
            fifa_code="FRA",
            confederation="UEFA",
            elo_rating=1820.0,
            fifa_rank=2,
        )
        session.add_all([team_home, team_away])
        session.flush()

        match = Match(
            home_team_id=team_home.id,
            away_team_id=team_away.id,
            datetime=datetime(2026, 6, 16, 20, 0),
            round="group",
            group="B",
            status="scheduled",
        )
        session.add(match)
        session.flush()

        odds1 = Odds(
            match_id=match.id,
            bookmaker="Pinnacle",
            timestamp=datetime(2026, 6, 14, 12, 0),
            home_odds=2.10,
            draw_odds=3.50,
            away_odds=3.20,
            over_25=1.80,
            btts_yes=1.70,
            is_closing=False,
        )
        odds2 = Odds(
            match_id=match.id,
            bookmaker="Bet365",
            timestamp=datetime(2026, 6, 14, 12, 0),
            home_odds=2.05,
            draw_odds=3.40,
            away_odds=3.30,
            is_closing=False,
        )
        session.add_all([odds1, odds2])
        session.commit()

        queried_match = session.query(Match).first()
        assert queried_match is not None
        assert len(queried_match.odds) == 2
        assert queried_match.odds[0].bookmaker == "Pinnacle"
        assert queried_match.odds[0].home_odds == 2.10
    finally:
        session.close()


def test_correct_score_odds() -> None:
    session = get_session()
    try:
        team_home = Team(
            name="España", fifa_code="ESP", confederation="UEFA", elo_rating=1780.0
        )
        team_away = Team(
            name="Inglaterra", fifa_code="ENG", confederation="UEFA", elo_rating=1790.0
        )
        session.add_all([team_home, team_away])
        session.flush()

        match = Match(
            home_team_id=team_home.id,
            away_team_id=team_away.id,
            datetime=datetime(2026, 6, 17, 18, 0),
            round="group",
            group="C",
            status="scheduled",
        )
        session.add(match)
        session.flush()

        cs_odds = CorrectScoreOdds(
            match_id=match.id,
            bookmaker="Pinnacle",
            timestamp=datetime(2026, 6, 14, 12, 0),
            home_goals=1,
            away_goals=0,
            odds=6.50,
        )
        session.add(cs_odds)
        session.commit()

        queried = session.query(CorrectScoreOdds).first()
        assert queried is not None
        assert queried.home_goals == 1
        assert queried.away_goals == 0
        assert queried.odds == 6.50
        assert queried.match.round == "group"
    finally:
        session.close()


def test_team_form() -> None:
    session = get_session()
    try:
        team = Team(
            name="Uruguay", fifa_code="URU", confederation="CONMEBOL", elo_rating=1700.0
        )
        session.add(team)
        session.flush()

        match = Match(
            home_team_id=team.id,
            away_team_id=team.id,
            datetime=datetime(2026, 5, 10, 20, 0),
            round="friendly",
            status="finished",
            home_score=2,
            away_score=1,
        )
        session.add(match)
        session.flush()

        form = TeamForm(
            team_id=team.id,
            match_id=match.id,
            goals_scored=2,
            goals_conceded=1,
            xg=1.8,
            xga=0.9,
            possession=55.0,
            shots=12,
            shots_on_target=5,
            result="W",
            is_home=True,
        )
        session.add(form)
        session.commit()

        queried_form = session.query(TeamForm).first()
        assert queried_form is not None
        assert queried_form.goals_scored == 2
        assert queried_form.result == "W"
        assert queried_form.team.name == "Uruguay"
        assert queried_form.match.home_score == 2
    finally:
        session.close()


def test_head_to_head() -> None:
    session = get_session()
    try:
        team_a = Team(
            name="Croacia", fifa_code="CRO", confederation="UEFA", elo_rating=1750.0
        )
        team_b = Team(
            name="Serbia", fifa_code="SRB", confederation="UEFA", elo_rating=1650.0
        )
        session.add_all([team_a, team_b])
        session.flush()

        h2h = HeadToHead(
            team_a_id=team_a.id,
            team_b_id=team_b.id,
            match_date=datetime(2022, 12, 1, 20, 0),
            goals_a=2,
            goals_b=1,
            competition="World Cup 2022",
        )
        session.add(h2h)
        session.commit()

        queried = session.query(HeadToHead).first()
        assert queried is not None
        assert queried.goals_a == 2
        assert queried.goals_b == 1
        assert queried.team_a.name == "Croacia"
        assert queried.team_b.name == "Serbia"
    finally:
        session.close()


def test_injury() -> None:
    session = get_session()
    try:
        team = Team(
            name="Portugal", fifa_code="POR", confederation="UEFA", elo_rating=1810.0
        )
        session.add(team)
        session.flush()

        injury = Injury(
            team_id=team.id,
            player_name="Cristiano Ronaldo",
            injury_type="Muscular",
            status="doubtful",
            expected_return=datetime(2026, 6, 20),
        )
        session.add(injury)
        session.commit()

        queried = session.query(Injury).first()
        assert queried is not None
        assert queried.player_name == "Cristiano Ronaldo"
        assert queried.status == "doubtful"
        assert queried.team.name == "Portugal"
    finally:
        session.close()


def test_system_prediction() -> None:
    session = get_session()
    try:
        team_home = Team(
            name="Japón", fifa_code="JPN", confederation="AFC", elo_rating=1550.0
        )
        team_away = Team(
            name="Senegal", fifa_code="SEN", confederation="CAF", elo_rating=1560.0
        )
        session.add_all([team_home, team_away])
        session.flush()

        match = Match(
            home_team_id=team_home.id,
            away_team_id=team_away.id,
            datetime=datetime(2026, 6, 18, 15, 0),
            round="group",
            group="D",
            status="scheduled",
        )
        session.add(match)
        session.flush()

        prediction = SystemPrediction(
            match_id=match.id,
            timestamp=datetime(2026, 6, 18, 10, 0),
            home_goals=1,
            away_goals=1,
            ep_score=2.35,
            ownership_estimate=0.15,
            contrarian_value=0.50,
            confidence=0.72,
            strategy_mode="balanced",
        )
        session.add(prediction)
        session.commit()

        queried = session.query(SystemPrediction).first()
        assert queried is not None
        assert queried.home_goals == 1
        assert queried.away_goals == 1
        assert queried.ep_score == 2.35
        assert queried.strategy_mode == "balanced"
        assert queried.match.home_team.name == "Japón"
    finally:
        session.close()


def test_participant_predictions_scores_relationship() -> None:
    session = get_session()
    try:
        team_home = Team(
            name="México", fifa_code="MEX", confederation="CONCACAF", elo_rating=1650.0
        )
        team_away = Team(
            name="Estados Unidos",
            fifa_code="USA",
            confederation="CONCACAF",
            elo_rating=1670.0,
        )
        session.add_all([team_home, team_away])
        session.flush()

        match = Match(
            home_team_id=team_home.id,
            away_team_id=team_away.id,
            datetime=datetime(2026, 6, 19, 20, 0),
            round="group",
            group="E",
            status="scheduled",
        )
        session.add(match)
        session.flush()

        participant = Participant(name="Juan Pérez", platform_id="juan_123")
        session.add(participant)
        session.flush()

        pred = ParticipantPrediction(
            match_id=match.id,
            participant_id=participant.id,
            home_goals=2,
            away_goals=0,
            timestamp=datetime(2026, 6, 19, 10, 0),
        )
        session.add(pred)
        session.flush()

        score = Score(
            match_id=match.id,
            participant_id=participant.id,
            result_pts=2,
            exact_pts=0,
            goals_home_pts=1,
            goals_away_pts=0,
            unique_pts=0,
            round_bonus_pts=0,
            total_pts=3,
        )
        session.add(score)
        session.commit()

        queried_participant = session.query(Participant).first()
        assert queried_participant is not None
        assert len(queried_participant.predictions) == 1
        assert queried_participant.predictions[0].home_goals == 2
        assert len(queried_participant.scores) == 1
        assert queried_participant.scores[0].total_pts == 3

        queried_match = session.query(Match).first()
        assert queried_match is not None
        assert len(queried_match.participant_predictions) == 1
        assert len(queried_match.scores) == 1
    finally:
        session.close()


def test_participant_profile() -> None:
    session = get_session()
    try:
        participant = Participant(name="María López", platform_id="maria_456")
        session.add(participant)
        session.flush()

        profile = ParticipantProfile(
            participant_id=participant.id,
            conservative_score=0.7,
            aggressive_score=0.3,
            market_follower=0.6,
            favorite_bias=0.5,
            recency_bias=0.4,
            home_bias=0.2,
            updated_at=datetime(2026, 6, 10, 12, 0),
        )
        session.add(profile)
        session.commit()

        queried = session.query(ParticipantProfile).first()
        assert queried is not None
        assert queried.conservative_score == 0.7
        assert queried.market_follower == 0.6
        assert queried.participant.name == "María López"
    finally:
        session.close()


def test_standing() -> None:
    session = get_session()
    try:
        participant = Participant(name="Carlos Gómez", platform_id="carlos_789")
        session.add(participant)
        session.flush()

        standing = Standing(
            participant_id=participant.id,
            round="group",
            total_points=45,
            position=3,
        )
        session.add(standing)
        session.commit()

        queried = session.query(Standing).first()
        assert queried is not None
        assert queried.total_points == 45
        assert queried.position == 3
        assert queried.round == "group"
        assert queried.participant.name == "Carlos Gómez"
    finally:
        session.close()


def test_initialize_database_creates_tables() -> None:
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.drop_all(engine)

    initialize_database("sqlite:///:memory:")

    table_names = Base.metadata.tables.keys()
    expected_tables = {
        "teams",
        "matches",
        "odds",
        "correct_score_odds",
        "team_form",
        "head_to_head",
        "injuries",
        "system_predictions",
        "participants",
        "participant_predictions",
        "participant_profiles",
        "scores",
        "standings",
    }
    assert set(table_names) == expected_tables


def test_cascade_team_relationships() -> None:
    session = get_session()
    try:
        team = Team(
            name="Chile", fifa_code="CHI", confederation="CONMEBOL", elo_rating=1600.0
        )
        session.add(team)
        session.flush()

        form = TeamForm(
            team_id=team.id,
            match_id=1,
            goals_scored=3,
            goals_conceded=0,
            xg=2.5,
            xga=0.5,
            possession=60.0,
            shots=15,
            shots_on_target=8,
            result="W",
            is_home=True,
        )

        match = Match(
            home_team_id=team.id,
            away_team_id=team.id,
            datetime=datetime(2026, 6, 20, 18, 0),
            round="group",
            status="scheduled",
        )
        session.add_all([match, form])
        session.flush()

        form.match_id = match.id
        session.commit()

        queried_team = session.query(Team).first()
        assert queried_team is not None
        assert len(queried_team.form_entries) == 1
        assert queried_team.form_entries[0].goals_scored == 3
    finally:
        session.close()
