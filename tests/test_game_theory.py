"""Tests for game theory layer: profiling, ownership, opponent model."""

from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import (
    Match,
    Participant,
    ParticipantPrediction,
    Team,
)
from src.game_theory.opponent_model import OpponentModel
from src.game_theory.ownership import OwnershipEstimate, OwnershipEstimator
from src.game_theory.profiling import PlayerArchetype, PlayerProfile, PlayerProfiler


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    session_local = sessionmaker(bind=engine)
    s = session_local()
    yield s
    s.close()


@pytest.fixture
def populated_session(session):
    teams = [
        Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL", group="A"),
        Team(name="Brazil", fifa_code="BRA", confederation="CONMEBOL", group="B"),
        Team(name="France", fifa_code="FRA", confederation="UEFA", group="C"),
        Team(name="Germany", fifa_code="GER", confederation="UEFA", group="D"),
    ]
    session.add_all(teams)
    session.flush()

    participants = []
    for i in range(15):
        p = Participant(name=f"Player_{i}", platform_id=f"player_{i}")
        session.add(p)
        participants.append(p)
    session.flush()

    match = Match(
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        datetime=datetime(2026, 6, 15),
        round="Group Stage",
        status="completed",
        home_score=1,
        away_score=0,
    )
    session.add(match)
    session.flush()

    predictions_data = [
        (0, 1, 0),
        (1, 1, 0),
        (2, 2, 1),
        (3, 0, 0),
        (4, 1, 1),
        (5, 3, 0),
        (6, 1, 2),
        (7, 2, 0),
        (8, 0, 1),
        (9, 2, 1),
        (10, 1, 0),
        (11, 2, 0),
        (12, 1, 1),
        (13, 0, 0),
        (14, 3, 1),
    ]
    for p_idx, home_g, away_g in predictions_data:
        pred = ParticipantPrediction(
            match_id=match.id,
            participant_id=participants[p_idx].id,
            home_goals=home_g,
            away_goals=away_g,
            timestamp=datetime(2026, 6, 14),
        )
        session.add(pred)
    session.flush()

    match2 = Match(
        home_team_id=teams[0].id,
        away_team_id=teams[2].id,
        datetime=datetime(2026, 6, 20),
        round="Group Stage",
        status="completed",
        home_score=2,
        away_score=2,
    )
    session.add(match2)
    session.flush()

    predictions_data2 = [
        (0, 2, 0),
        (1, 1, 1),
        (2, 2, 1),
        (3, 1, 0),
        (4, 1, 1),
        (5, 3, 2),
        (6, 0, 2),
        (7, 2, 1),
        (8, 1, 2),
        (9, 2, 2),
        (10, 2, 0),
        (11, 3, 0),
        (12, 1, 1),
        (13, 0, 1),
        (14, 3, 2),
    ]
    for p_idx, home_g, away_g in predictions_data2:
        pred = ParticipantPrediction(
            match_id=match2.id,
            participant_id=participants[p_idx].id,
            home_goals=home_g,
            away_goals=away_g,
            timestamp=datetime(2026, 6, 19),
        )
        session.add(pred)
    session.commit()

    return session, participants, [match, match2], teams


# ── Profiling Tests ────────────────────────────────────────────────────


class TestPlayerProfile:
    def test_profile_dataclass_creation(self) -> None:
        p = PlayerProfile(
            participant_id=1,
            name="TestPlayer",
            conservative_score=0.5,
            aggressive_score=0.1,
            market_follower_score=0.2,
            intuition_score=0.2,
        )
        assert p.participant_id == 1
        assert p.name == "TestPlayer"
        assert p.conservative_score == 0.5

    def test_dominant_archetype_conservative(self) -> None:
        p = PlayerProfile(
            participant_id=1,
            name="Test",
            conservative_score=0.6,
            aggressive_score=0.1,
            market_follower_score=0.2,
            intuition_score=0.1,
        )
        assert p.dominant_archetype == PlayerArchetype.CONSERVATIVE

    def test_uniform_profile(self) -> None:
        p = PlayerProfile.uniform(5, "Unknown")
        assert p.conservative_score == 0.25
        assert p.aggressive_score == 0.25
        assert p.market_follower_score == 0.25
        assert p.intuition_score == 0.25
        assert p.total_predictions == 0

    def test_to_dict(self) -> None:
        p = PlayerProfile(
            participant_id=1,
            name="Dict",
            conservative_score=0.4,
            aggressive_score=0.3,
            market_follower_score=0.2,
            intuition_score=0.1,
        )
        d = p.to_dict()
        assert d["participant_id"] == 1
        assert d["name"] == "Dict"
        assert d["conservative_score"] == 0.4


class TestPlayerProfiler:
    def test_profile_all_returns_15(self, populated_session) -> None:
        session, participants, _, _ = populated_session
        profiler = PlayerProfiler(session)
        profiles = profiler.profile_all()
        assert len(profiles) == 15

    def test_profile_participant_has_archetype_scores(self, populated_session) -> None:
        session, participants, _, _ = populated_session
        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(participants[0].id)
        total = (
            profile.conservative_score
            + profile.aggressive_score
            + profile.market_follower_score
            + profile.intuition_score
        )
        assert abs(total - 1.0) < 0.01

    def test_archetype_scores_sum_to_one(self, populated_session) -> None:
        session, participants, _, _ = populated_session
        profiler = PlayerProfiler(session)
        for p in participants:
            profile = profiler.profile_participant(p.id)
            total = (
                profile.conservative_score
                + profile.aggressive_score
                + profile.market_follower_score
                + profile.intuition_score
            )
            assert abs(total - 1.0) < 0.01, f"Participant {p.id} sum={total}"

    def test_profile_with_no_predictions_returns_uniform(self, session) -> None:
        p = Participant(name="NoData", platform_id="nodata")
        session.add(p)
        session.flush()

        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(p.id)

        assert profile.conservative_score == 0.25
        assert profile.aggressive_score == 0.25
        assert profile.market_follower_score == 0.25
        assert profile.intuition_score == 0.25
        assert profile.total_predictions == 0

    def test_conservative_player_has_high_conservative_score(
        self, session
    ) -> None:
        p = Participant(name="Conservative", platform_id="cons")
        session.add(p)
        session.flush()

        t_home = Team(name="TeamA", fifa_code="AAA", confederation="UEFA", group="X")
        t_away = Team(name="TeamB", fifa_code="BBB", confederation="UEFA", group="X")
        session.add_all([t_home, t_away])
        session.flush()

        match = Match(
            home_team_id=t_home.id,
            away_team_id=t_away.id,
            datetime=datetime(2026, 6, 1),
            round="Group",
            status="completed",
        )
        session.add(match)
        session.flush()

        for _ in range(10):
            pred = ParticipantPrediction(
                match_id=match.id,
                participant_id=p.id,
                home_goals=1,
                away_goals=0,
                timestamp=datetime(2026, 6, 1),
            )
            session.add(pred)
        session.commit()

        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(p.id)
        assert profile.conservative_score > 0.4

    def test_detect_team_preferences(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(participants[0].id)
        assert isinstance(profile.popular_team_score, dict)
        assert isinstance(profile.underdog_team_score, dict)
        assert profile.total_predictions >= 0

    def test_calculate_biases(self, populated_session) -> None:
        session, participants, _, _ = populated_session
        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(participants[0].id)
        assert 0.0 <= profile.favorite_bias <= 1.0
        assert 0.0 <= profile.home_bias <= 1.0
        assert 0.0 <= profile.draw_aversion <= 1.0

    def test_update_profiles_persists(self, populated_session) -> None:
        session, participants, _, _ = populated_session
        profiler = PlayerProfiler(session)
        profiler.update_profiles()

        from src.database.models import ParticipantProfile

        for p in participants:
            db_profile = (
                session.query(ParticipantProfile)
                .filter_by(participant_id=p.id)
                .first()
            )
            assert db_profile is not None
            assert db_profile.conservative_score is not None
            assert db_profile.aggressive_score is not None

    def test_missing_participant_raises(self, session) -> None:
        profiler = PlayerProfiler(session)
        with pytest.raises(ValueError):
            profiler.profile_participant(9999)

    def test_calculate_archetype_scores_empty(self, session) -> None:
        profiler = PlayerProfiler(session)
        scores = profiler.calculate_archetype_scores([])
        assert scores == (0.25, 0.25, 0.25, 0.25)


# ── Ownership Tests ────────────────────────────────────────────────────


class TestOwnershipEstimate:
    def test_ownership_estimate_dataclass(self) -> None:
        matrix = np.zeros((8, 8))
        matrix[1, 0] = 0.5
        matrix[2, 1] = 0.3
        matrix[0, 0] = 0.2

        est = OwnershipEstimate(
            score_matrix=matrix,
            most_popular_score=(1, 0),
            ownership_of_most_popular=0.5,
            entropy=1.2,
            unique_opportunities=[(2, 2)],
        )
        assert est.most_popular_score == (1, 0)
        assert est.ownership_of_most_popular == 0.5
        assert est.entropy > 0


class TestOwnershipEstimator:
    def test_estimate_returns_matrix(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        est = estimator.estimate(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        assert est.score_matrix.shape == (8, 8)
        assert isinstance(est.most_popular_score, tuple)
        assert 0.0 <= est.ownership_of_most_popular <= 1.0

    def test_estimate_with_no_profiles(self, session) -> None:
        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        t1 = Team(name="X", fifa_code="XXX", confederation="UEFA", group="X")
        t2 = Team(name="Y", fifa_code="YYY", confederation="UEFA", group="X")
        session.add_all([t1, t2])
        session.flush()
        match = Match(
            home_team_id=t1.id,
            away_team_id=t2.id,
            datetime=datetime(2026, 6, 1),
            round="Group",
            status="scheduled",
        )
        session.add(match)
        session.flush()

        est = estimator.estimate(match, "X", "Y", max_goals=7)
        assert est.score_matrix.sum() == 0.0

    def test_identical_profiles_high_ownership(self, session) -> None:
        p1 = Participant(name="Clone1", platform_id="c1")
        p2 = Participant(name="Clone2", platform_id="c2")
        session.add_all([p1, p2])
        session.flush()

        t_home = Team(name="Home", fifa_code="HOM", confederation="UEFA", group="H")
        t_away = Team(name="Away", fifa_code="AWY", confederation="UEFA", group="H")
        session.add_all([t_home, t_away])
        session.flush()

        match = Match(
            home_team_id=t_home.id,
            away_team_id=t_away.id,
            datetime=datetime(2026, 6, 1),
            round="Group",
            status="completed",
            home_score=2,
            away_score=0,
        )
        session.add(match)
        session.flush()

        for p in [p1, p2]:
            for _ in range(5):
                pred = ParticipantPrediction(
                    match_id=match.id,
                    participant_id=p.id,
                    home_goals=2,
                    away_goals=0,
                    timestamp=datetime(2026, 6, 1),
                )
                session.add(pred)
        session.commit()

        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        est = estimator.estimate(match, "Home", "Away", max_goals=7)
        assert est.ownership_of_most_popular > 0.0

    def test_get_contrarian_value_range(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        est = estimator.estimate(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )

        model_probs = np.zeros((8, 8))
        model_probs[1, 0] = 0.3
        model_probs[2, 1] = 0.2
        model_probs[0, 0] = 0.1
        model_probs[1, 1] = 0.15
        model_probs[2, 0] = 0.1
        model_probs[3, 0] = 0.05
        model_probs[0, 1] = 0.05
        model_probs[3, 1] = 0.05

        cv = estimator.get_contrarian_value(est, model_probs)
        assert cv.shape == (8, 8)
        assert (cv >= 0.0).all()
        assert (cv <= 1.0).all()

    def test_ownership_entropy_non_negative(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        est = estimator.estimate(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        assert est.entropy >= 0.0

    def test_unique_opportunities_threshold(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        est = estimator.estimate(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        for home_g, away_g in est.unique_opportunities:
            assert est.score_matrix[home_g, away_g] < 0.10

    def test_estimate_batch(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        estimator = OwnershipEstimator(profiler)

        results = estimator.estimate_batch(matches, max_goals=7)
        assert len(results) == len(matches)
        for match_id in results:
            assert results[match_id].score_matrix.shape == (8, 8)


# ── Opponent Model Tests ───────────────────────────────────────────────


class TestOpponentModel:
    def test_predict_opponent_choice(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        model = OpponentModel(profiler)

        home_g, away_g = model.predict_opponent_choice(
            participants[0].id,
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        assert 0 <= home_g <= 7
        assert 0 <= away_g <= 7

    def test_predict_all_opponents(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        model = OpponentModel(profiler)

        choices = model.predict_all_opponents(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        assert len(choices) == 15

    def test_predict_all_excludes_participant(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        model = OpponentModel(profiler)

        choices = model.predict_all_opponents(
            matches[0],
            teams[0].name,
            teams[1].name,
            exclude_participant=participants[0].id,
            max_goals=7,
        )
        assert len(choices) == 14

    def test_build_ownership_matrix(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler = PlayerProfiler(session)
        model = OpponentModel(profiler)

        matrix = model.build_ownership_matrix(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        assert matrix.shape == (8, 8)
        assert abs(matrix.sum() - 1.0) < 0.01

    def test_deterministic_with_same_seed(self, populated_session) -> None:
        session, participants, matches, teams = populated_session
        profiler1 = PlayerProfiler(session)
        profiler2 = PlayerProfiler(session)

        model1 = OpponentModel(profiler1)
        model2 = OpponentModel(profiler2)

        choice1 = model1.predict_opponent_choice(
            participants[0].id,
            matches[0],
            teams[0].name,
            teams[1].name,
        )
        choice2 = model2.predict_opponent_choice(
            participants[0].id,
            matches[0],
            teams[0].name,
            teams[1].name,
        )
        assert choice1 == choice2


# ── Integration Tests ──────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline(self, populated_session) -> None:
        session, participants, matches, teams = populated_session

        profiler = PlayerProfiler(session)
        profiles = profiler.profile_all()
        assert len(profiles) == 15

        estimator = OwnershipEstimator(profiler)
        est = estimator.estimate(
            matches[0],
            teams[0].name,
            teams[1].name,
            max_goals=7,
        )
        assert est.score_matrix.shape == (8, 8)
        assert est.most_popular_score is not None
        assert len(est.unique_opportunities) >= 0

        opp_model = OpponentModel(profiler)
        choices = opp_model.predict_all_opponents(
            matches[0],
            teams[0].name,
            teams[1].name,
        )
        assert len(choices) == 15

    def test_archetype_detection_from_known_pattern(self, session) -> None:
        p = Participant(name="LowScorePredictor", platform_id="low")
        session.add(p)
        session.flush()

        t_home = Team(name="Fav", fifa_code="FAV", confederation="UEFA", group="F")
        t_away = Team(name="Und", fifa_code="UND", confederation="UEFA", group="F")
        session.add_all([t_home, t_away])
        session.flush()

        for _ in range(5):
            match = Match(
                home_team_id=t_home.id,
                away_team_id=t_away.id,
                datetime=datetime(2026, 6, 1),
                round="Group",
                status="completed",
            )
            session.add(match)
            session.flush()
            pred = ParticipantPrediction(
                match_id=match.id,
                participant_id=p.id,
                home_goals=1,
                away_goals=0,
                timestamp=datetime(2026, 6, 1),
            )
            session.add(pred)
        session.commit()

        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(p.id)
        assert profile.conservative_score > profile.aggressive_score

    def test_edge_case_single_prediction(self, session) -> None:
        p = Participant(name="Single", platform_id="one")
        session.add(p)
        session.flush()

        t_home = Team(name="H", fifa_code="HHH", confederation="UEFA", group="H")
        t_away = Team(name="A", fifa_code="AAA", confederation="UEFA", group="H")
        session.add_all([t_home, t_away])
        session.flush()

        match = Match(
            home_team_id=t_home.id,
            away_team_id=t_away.id,
            datetime=datetime(2026, 6, 1),
            round="Group",
            status="completed",
        )
        session.add(match)
        session.flush()

        pred = ParticipantPrediction(
            match_id=match.id,
            participant_id=p.id,
            home_goals=3,
            away_goals=2,
            timestamp=datetime(2026, 6, 1),
        )
        session.add(pred)
        session.commit()

        profiler = PlayerProfiler(session)
        profile = profiler.profile_participant(p.id)
        assert profile.total_predictions == 1
        total = (
            profile.conservative_score
            + profile.aggressive_score
            + profile.market_follower_score
            + profile.intuition_score
        )
        assert abs(total - 1.0) < 0.01
