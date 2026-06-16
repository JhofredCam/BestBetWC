from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    fifa_code: Mapped[str] = mapped_column(String(3), unique=True)
    confederation: Mapped[str]
    elo_rating: Mapped[float | None]
    fifa_rank: Mapped[int | None]
    group: Mapped[str | None]

    home_matches: Mapped[list["Match"]] = relationship(
        back_populates="home_team", foreign_keys="Match.home_team_id"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        back_populates="away_team", foreign_keys="Match.away_team_id"
    )
    form_entries: Mapped[list["TeamForm"]] = relationship(back_populates="team")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    datetime: Mapped[datetime]
    venue: Mapped[str | None]
    city: Mapped[str | None]
    round: Mapped[str]
    group: Mapped[str | None]
    status: Mapped[str]
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]

    home_team: Mapped["Team"] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    odds: Mapped[list["Odds"]] = relationship(back_populates="match")
    correct_score_odds: Mapped[list["CorrectScoreOdds"]] = relationship(back_populates="match")
    team_form: Mapped[list["TeamForm"]] = relationship(back_populates="match")
    system_predictions: Mapped[list["SystemPrediction"]] = relationship(back_populates="match")
    participant_predictions: Mapped[list["ParticipantPrediction"]] = relationship(
        back_populates="match"
    )
    scores: Mapped[list["Score"]] = relationship(back_populates="match")


class Odds(Base):
    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    bookmaker: Mapped[str]
    timestamp: Mapped[datetime]
    home_odds: Mapped[float]
    draw_odds: Mapped[float]
    away_odds: Mapped[float]
    over_15: Mapped[float | None]
    over_25: Mapped[float | None]
    over_35: Mapped[float | None]
    btts_yes: Mapped[float | None]
    btts_no: Mapped[float | None]
    is_closing: Mapped[bool]

    match: Mapped["Match"] = relationship(back_populates="odds")


class CorrectScoreOdds(Base):
    __tablename__ = "correct_score_odds"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    bookmaker: Mapped[str]
    timestamp: Mapped[datetime]
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    odds: Mapped[float]

    match: Mapped["Match"] = relationship(back_populates="correct_score_odds")


class TeamForm(Base):
    __tablename__ = "team_form"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    goals_scored: Mapped[int]
    goals_conceded: Mapped[int]
    xg: Mapped[float | None]
    xga: Mapped[float | None]
    possession: Mapped[float | None]
    shots: Mapped[int | None]
    shots_on_target: Mapped[int | None]
    result: Mapped[str]
    is_home: Mapped[bool]

    team: Mapped["Team"] = relationship(back_populates="form_entries")
    match: Mapped["Match"] = relationship(back_populates="team_form")


class HeadToHead(Base):
    __tablename__ = "head_to_head"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_a_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    team_b_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    match_date: Mapped[datetime]
    goals_a: Mapped[int]
    goals_b: Mapped[int]
    competition: Mapped[str | None]

    team_a: Mapped["Team"] = relationship(foreign_keys=[team_a_id])
    team_b: Mapped["Team"] = relationship(foreign_keys=[team_b_id])


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    player_name: Mapped[str]
    injury_type: Mapped[str]
    status: Mapped[str]
    expected_return: Mapped[datetime | None]

    team: Mapped["Team"] = relationship()


class SystemPrediction(Base):
    __tablename__ = "system_predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    timestamp: Mapped[datetime]
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    ep_score: Mapped[float]
    ownership_estimate: Mapped[float | None]
    contrarian_value: Mapped[float | None]
    confidence: Mapped[float]
    strategy_mode: Mapped[str | None]

    match: Mapped["Match"] = relationship(back_populates="system_predictions")


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    platform_id: Mapped[str | None]

    predictions: Mapped[list["ParticipantPrediction"]] = relationship(back_populates="participant")
    profiles: Mapped[list["ParticipantProfile"]] = relationship(back_populates="participant")
    scores: Mapped[list["Score"]] = relationship(back_populates="participant")
    standings: Mapped[list["Standing"]] = relationship(back_populates="participant")


class ParticipantPrediction(Base):
    __tablename__ = "participant_predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    timestamp: Mapped[datetime]

    match: Mapped["Match"] = relationship(back_populates="participant_predictions")
    participant: Mapped["Participant"] = relationship(back_populates="predictions")


class ParticipantProfile(Base):
    __tablename__ = "participant_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    conservative_score: Mapped[float]
    aggressive_score: Mapped[float]
    market_follower: Mapped[float]
    favorite_bias: Mapped[float]
    recency_bias: Mapped[float]
    home_bias: Mapped[float]
    updated_at: Mapped[datetime]

    participant: Mapped["Participant"] = relationship(back_populates="profiles")


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    result_pts: Mapped[int]
    exact_pts: Mapped[int]
    goals_home_pts: Mapped[int]
    goals_away_pts: Mapped[int]
    unique_pts: Mapped[int]
    round_bonus_pts: Mapped[int]
    total_pts: Mapped[int]

    match: Mapped["Match"] = relationship(back_populates="scores")
    participant: Mapped["Participant"] = relationship(back_populates="scores")


class Standing(Base):
    __tablename__ = "standings"

    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    round: Mapped[str]
    total_points: Mapped[int]
    position: Mapped[int]

    participant: Mapped["Participant"] = relationship(back_populates="standings")
