"""Tests for SPEC-008: FBref scraper + feature pipeline."""

import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.database.connection import Base, get_engine, get_session
from src.database.models import (
    CorrectScoreOdds,
    HeadToHead,
    Match,
    Odds,
    Team,
    TeamForm,
)
from src.etl.fbref import FBrefMatchStats, FBrefScraper
from src.features.context import get_context_features
from src.features.market import _shannon_entropy, get_market_features
from src.features.performance import get_performance_features
from src.features.pipeline import FEATURE_COLUMNS, FeaturePipeline, MatchFeatureVector

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "fbref"


@pytest.fixture(autouse=True)
def setup_database() -> Generator[None, None, None]:
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def session() -> Generator[Session, None, None]:
    s = get_session()
    try:
        yield s
    finally:
        s.close()


# ═══════════════════════════════════════════════════════════════════════
# FBref scraper tests
# ═══════════════════════════════════════════════════════════════════════


def test_fbref_match_stats_dataclass() -> None:
    stats = FBrefMatchStats(
        match_url="/matches/test",
        home_team="Argentina",
        away_team="Brazil",
        home_xg=1.8,
        away_xg=0.9,
        home_xga=0.9,
        away_xga=1.8,
        home_possession=55.0,
        away_possession=45.0,
        home_shots=12,
        away_shots=8,
        home_shots_on_target=5,
        away_shots_on_target=3,
        home_passes=450,
        away_passes=380,
    )
    assert stats.home_team == "Argentina"
    assert stats.away_xg == 0.9
    assert stats.home_xga == 0.9
    assert stats.away_xga == 1.8
    assert stats.home_shots == 12


def test_fbref_scraper_parse_match_table_from_fixture() -> None:
    html_path = FIXTURES_DIR / "world_cup_2022.html"
    if not html_path.exists():
        pytest.skip("FBref fixture HTML not found")

    html = html_path.read_text(encoding="utf-8")
    scraper = FBrefScraper()
    matches = scraper._parse_match_table(html)

    assert len(matches) == 4
    assert matches[0].home_team == "Qatar"
    assert matches[0].away_team == "Ecuador"
    assert matches[0].home_xg == 0.1
    assert matches[0].away_xg == 1.8
    assert matches[0].home_score == 0
    assert matches[0].away_score == 2

    assert matches[1].home_team == "England"
    assert matches[1].home_xg == 3.5
    assert matches[1].away_xg == 1.1


def test_fbref_scraper_parse_match_detail_from_fixture() -> None:
    html_path = FIXTURES_DIR / "match_qatar_ecuador.html"
    if not html_path.exists():
        pytest.skip("FBref match detail fixture HTML not found")

    html = html_path.read_text(encoding="utf-8")
    scraper = FBrefScraper()
    result = scraper._parse_match_detail(html, "/matches/abc123/Qatar-Ecuador")

    assert result is not None
    assert result.home_team == "Qatar"
    assert result.away_team == "Ecuador"
    assert result.home_xg == pytest.approx(0.10, rel=0.2)
    assert result.away_xg == pytest.approx(1.80, rel=0.2)
    assert result.home_shots == 3
    assert result.away_shots == 3
    assert result.home_shots_on_target == 1
    assert result.away_shots_on_target == 3
    assert result.home_possession == 65.0
    assert result.away_possession == 35.0


def test_fbref_scraper_parse_empty_html() -> None:
    scraper = FBrefScraper()
    results = scraper._parse_match_table("<html><body>No tables here</body></html>")
    assert results == []


def test_fbref_scraper_cache_write_and_read() -> None:
    cache_dir = Path(tempfile.mkdtemp(prefix="fbref_test_"))
    try:
        scraper = FBrefScraper(cache_dir=cache_dir)
        html_content = "<html><body>Test</body></html>"
        scraper._write_cache("test_key", html_content)
        cached = scraper._read_cache("test_key")
        assert cached == html_content
    finally:
        for f in cache_dir.iterdir():
            f.unlink()
        cache_dir.rmdir()


def test_fbref_scraper_cache_miss() -> None:
    cache_dir = Path(tempfile.mkdtemp(prefix="fbref_test_"))
    try:
        scraper = FBrefScraper(cache_dir=cache_dir)
        assert scraper._read_cache("nonexistent_key") is None
    finally:
        for f in cache_dir.iterdir():
            f.unlink()
        cache_dir.rmdir()


def test_fbref_scraper_cache_key_deterministic() -> None:
    scraper = FBrefScraper()
    key1 = scraper._cache_key("https://fbref.com/en/matches/test")
    key2 = scraper._cache_key("https://fbref.com/en/matches/test")
    assert key1 == key2
    assert len(key1) == 16


def test_fbref_xg_values_are_positive() -> None:
    html_path = FIXTURES_DIR / "world_cup_2022.html"
    if not html_path.exists():
        pytest.skip("FBref fixture HTML not found")

    html = html_path.read_text(encoding="utf-8")
    scraper = FBrefScraper()
    matches = scraper._parse_match_table(html)

    for m in matches:
        assert m.home_xg >= 0.0
        assert m.away_xg >= 0.0


@pytest.mark.asyncio
async def test_fbref_scraper_rate_limit() -> None:
    scraper = FBrefScraper()
    import time
    start = time.time()
    await scraper._rate_limit_wait()
    await scraper._rate_limit_wait()
    elapsed = time.time() - start
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_fbref_scraper_close() -> None:
    scraper = FBrefScraper()
    await scraper.close()
    assert scraper._client is None


# ═══════════════════════════════════════════════════════════════════════
# Market feature tests
# ═══════════════════════════════════════════════════════════════════════


def test_shannon_entropy_uniform() -> None:
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = _shannon_entropy(probs)
    assert entropy == pytest.approx(1.386, rel=0.01)


def test_shannon_entropy_certain() -> None:
    probs = np.array([1.0, 0.0])
    entropy = _shannon_entropy(probs)
    assert entropy == pytest.approx(0.0, abs=0.001)


def test_shannon_entropy_empty() -> None:
    assert _shannon_entropy(np.array([])) == 0.0


def test_shannon_entropy_all_zeros() -> None:
    assert _shannon_entropy(np.array([0.0, 0.0])) == 0.0


def test_get_market_features_empty(session: Session) -> None:
    result = get_market_features(session, 999)
    assert result["market_home_prob"] == pytest.approx(0.333)
    assert result["market_draw_prob"] == pytest.approx(0.333)
    assert result["market_away_prob"] == pytest.approx(0.333)
    assert result["market_num_bookmakers"] == 0


def test_get_market_features_basic(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    o1 = Odds(
        match_id=m.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.10, draw_odds=3.50, away_odds=3.20,
        is_closing=True,
    )
    o2 = Odds(
        match_id=m.id, bookmaker="Bet365",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.05, draw_odds=3.40, away_odds=3.30,
        is_closing=True,
    )
    session.add_all([o1, o2])
    session.commit()

    result = get_market_features(session, m.id)

    assert result["market_num_bookmakers"] == 2
    home_sum = result["market_home_prob"] + result["market_draw_prob"] + result["market_away_prob"]
    assert home_sum == pytest.approx(1.0, abs=0.02)
    assert result["market_home_prob"] > 0.4
    assert result["market_disagreement"] >= 0.0
    assert result["market_margin"] > 0.0


def test_get_market_features_disagreement_single_bookmaker(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    o = Odds(
        match_id=m.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.10, draw_odds=3.50, away_odds=3.20,
        is_closing=True,
    )
    session.add(o)
    session.commit()

    result = get_market_features(session, m.id)
    assert result["market_disagreement"] == pytest.approx(0.0)


def test_get_market_features_with_totals_and_btts(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    o = Odds(
        match_id=m.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.10, draw_odds=3.50, away_odds=3.20,
        over_15=0.71, over_25=0.55, over_35=0.42,
        btts_yes=0.56, btts_no=0.44,
        is_closing=True,
    )
    session.add(o)
    session.commit()

    result = get_market_features(session, m.id)
    assert result["market_btts_yes_prob"] == pytest.approx(0.56)
    assert result["market_over_15_prob"] == pytest.approx(0.71)
    assert result["market_over_25_prob"] == pytest.approx(0.55)
    assert result["market_over_35_prob"] == pytest.approx(0.42)


def test_get_market_features_odds_movement(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    early = Odds(
        match_id=m.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
        home_odds=2.50, draw_odds=3.40, away_odds=2.80,
        is_closing=False,
    )
    late = Odds(
        match_id=m.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.10, draw_odds=3.50, away_odds=3.20,
        is_closing=True,
    )
    session.add_all([early, late])
    session.commit()

    result = get_market_features(session, m.id)
    assert result["odds_movement_home"] > 0.0


def test_get_market_features_correct_score_entropy(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    for hg, ag, odds in [
        (1, 0, 6.50), (0, 0, 9.00), (0, 1, 8.00),
        (2, 0, 12.00), (2, 1, 10.00),
    ]:
        cs = CorrectScoreOdds(
            match_id=m.id, bookmaker="Pinnacle",
            timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
            home_goals=hg, away_goals=ag, odds=odds,
        )
        session.add(cs)
    session.commit()

    result = get_market_features(session, m.id)
    assert result["correct_score_entropy"] > 0.0


# ═══════════════════════════════════════════════════════════════════════
# Performance feature tests
# ═══════════════════════════════════════════════════════════════════════


def test_performance_features_elo_and_rank(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    result = get_performance_features(session, m)

    assert result["elo_home"] == 2100.0
    assert result["elo_away"] == 1750.0
    assert result["elo_diff"] == 350.0
    assert result["fifa_rank_home"] == 1.0
    assert result["fifa_rank_away"] == 44.0
    assert result["fifa_rank_diff"] == 43.0


def test_performance_features_form(session: Session) -> None:
    t_h = Team(name="Brazil", fifa_code="BRA", confederation="CONMEBOL", group="A")
    t_a = Team(name="Serbia", fifa_code="SRB", confederation="UEFA", group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m_target = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 20, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m_target)

    m1 = Match(
        home_team_id=t_h.id, away_team_id=999,
        datetime=datetime(2026, 6, 10, 18, 0),
        round="group", group="A", status="finished",
        home_score=2, away_score=0,
    )
    m2 = Match(
        home_team_id=999, away_team_id=t_h.id,
        datetime=datetime(2026, 6, 15, 18, 0),
        round="group", group="A", status="finished",
        home_score=1, away_score=1,
    )
    session.add_all([m1, m2])
    session.commit()

    tf1 = TeamForm(
        team_id=t_h.id, match_id=m1.id,
        goals_scored=2, goals_conceded=0,
        xg=2.1, xga=0.4, possession=0.55, shots=14, shots_on_target=6,
        result="W", is_home=True,
    )
    tf2 = TeamForm(
        team_id=t_h.id, match_id=m2.id,
        goals_scored=1, goals_conceded=1,
        xg=1.2, xga=0.9, possession=0.48, shots=10, shots_on_target=4,
        result="D", is_home=False,
    )
    session.add_all([tf1, tf2])
    session.commit()

    result = get_performance_features(session, m_target)
    assert result["form_pts_5_home"] == 4.0
    assert result["goals_scored_5_home"] == 3.0
    assert result["goals_conceded_5_home"] == 1.0


def test_performance_features_temporal_no_future_leak(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA", group="A")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA", group="A")
    session.add_all([t_h, t_a])
    session.flush()

    target = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(target)

    future_match = Match(
        home_team_id=t_h.id, away_team_id=999,
        datetime=datetime(2026, 6, 20, 18, 0),
        round="group", group="A", status="scheduled",
        home_score=None, away_score=None,
    )
    session.add(future_match)
    session.commit()

    tf_future = TeamForm(
        team_id=t_h.id, match_id=future_match.id,
        goals_scored=10, goals_conceded=0, result="W", is_home=True,
    )
    session.add(tf_future)
    session.commit()

    result = get_performance_features(session, target)
    assert result["form_pts_5_home"] == 0.0
    assert result["goals_scored_5_home"] == 0.0


def test_performance_features_h2h(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL", group="A")
    t_a = Team(name="Brazil", fifa_code="BRA", confederation="CONMEBOL", group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    h2h_entries = [
        HeadToHead(team_a_id=t_h.id, team_b_id=t_a.id,
                   match_date=datetime(2021, 7, 10), goals_a=1, goals_b=0),
        HeadToHead(team_a_id=t_h.id, team_b_id=t_a.id,
                   match_date=datetime(2019, 7, 2), goals_a=2, goals_b=1),
        HeadToHead(team_a_id=t_a.id, team_b_id=t_h.id,
                   match_date=datetime(2020, 11, 15), goals_a=1, goals_b=1),
    ]
    session.add_all(h2h_entries)
    session.commit()

    result = get_performance_features(session, m)
    assert result["h2h_wins_home"] == 2
    assert result["h2h_draws"] == 1
    assert result["h2h_wins_away"] == 0
    assert result["h2h_avg_goals"] == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════
# Context feature tests
# ═══════════════════════════════════════════════════════════════════════


def test_context_features_group_stage(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA", group="A")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA", group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="Group A - 1", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    result = get_context_features(session, m)
    assert result["match_importance"] == 0.5
    assert result["is_knockout"] == 0.0
    assert result["round_weight"] == 1.0


def test_context_features_knockout(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 7, 5, 20, 0),
        round="round_of_16", group=None, status="scheduled",
    )
    session.add(m)
    session.commit()

    result = get_context_features(session, m)
    assert result["match_importance"] == 1.0
    assert result["is_knockout"] == 1.0
    assert result["round_weight"] == 2.0


def test_context_features_final(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 7, 19, 20, 0),
        round="final", group=None, status="scheduled",
    )
    session.add(m)
    session.commit()

    result = get_context_features(session, m)
    assert result["match_importance"] == 1.0
    assert result["is_knockout"] == 1.0
    assert result["round_weight"] == 5.0


def test_context_features_rest_days(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA", group="A")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA", group="A")
    session.add_all([t_h, t_a])
    session.flush()

    prev_match = Match(
        home_team_id=t_h.id, away_team_id=999,
        datetime=datetime(2026, 6, 12, 20, 0),
        round="group", group="A", status="finished",
    )
    session.add(prev_match)

    target = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(target)
    session.commit()

    result = get_context_features(session, target)
    assert result["rest_days_home"] == 3.0
    assert result["rest_days_away"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Pipeline integration tests
# ═══════════════════════════════════════════════════════════════════════


def test_match_feature_vector_defaults() -> None:
    fv = MatchFeatureVector(match_id=1, timestamp=datetime.now(UTC))
    assert fv.market_home_prob == pytest.approx(0.333)
    assert fv.elo_diff == 0.0
    assert fv.match_importance == 0.5
    assert fv.is_knockout is False


def test_match_feature_vector_to_array(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    pipeline = FeaturePipeline(session)
    fv = pipeline.build_features(m.id)

    arr = fv.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert len(arr) == len(FEATURE_COLUMNS)
    assert arr.dtype == np.float64


def test_match_feature_vector_to_dict(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    pipeline = FeaturePipeline(session)
    fv = pipeline.build_features(m.id)

    d = fv.to_dict()
    assert isinstance(d, dict)
    assert "elo_diff" in d
    assert d["elo_diff"] == 350.0
    assert d["fifa_rank_diff"] == 43.0


def test_match_feature_vector_serialization(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    pipeline = FeaturePipeline(session)
    fv = pipeline.build_features(m.id)

    d = fv.to_dict()
    assert all(isinstance(k, str) for k in d)
    assert all(isinstance(v, (int, float)) for v in d.values())

    arr = fv.to_array()
    assert not np.any(np.isnan(arr))
    assert np.all(np.isfinite(arr))


def test_feature_pipeline_build_features_sets_match_id(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    pipeline = FeaturePipeline(session)
    fv = pipeline.build_features(m.id)

    assert fv.match_id == m.id
    assert isinstance(fv.timestamp, datetime)


def test_feature_pipeline_missing_match_raises(session: Session) -> None:
    pipeline = FeaturePipeline(session)
    with pytest.raises(ValueError, match="not found"):
        pipeline.build_features(99999)


def test_feature_pipeline_batch(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m1 = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    m2 = Match(
        home_team_id=t_a.id, away_team_id=t_h.id,
        datetime=datetime(2026, 6, 20, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add_all([m1, m2])
    session.commit()

    pipeline = FeaturePipeline(session)
    results = pipeline.build_features_batch([m1.id, m2.id])

    assert len(results) == 2
    assert results[0].match_id == m1.id
    assert results[1].match_id == m2.id


def test_feature_pipeline_with_full_odds_data(session: Session) -> None:
    t_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL",
               elo_rating=2100.0, fifa_rank=1, group="A")
    t_a = Team(name="Ecuador", fifa_code="ECU", confederation="CONMEBOL",
               elo_rating=1750.0, fifa_rank=44, group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    o = Odds(
        match_id=m.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=1.40, draw_odds=5.00, away_odds=8.00,
        over_15=0.85, over_25=0.60, over_35=0.35,
        btts_yes=0.55, btts_no=0.45,
        is_closing=True,
    )
    session.add(o)
    session.commit()

    pipeline = FeaturePipeline(session)
    fv = pipeline.build_features(m.id)

    assert fv.market_home_prob > 0.5
    assert fv.market_home_prob > fv.market_draw_prob
    assert fv.market_home_prob > fv.market_away_prob


def test_feature_pipeline_no_odds(session: Session) -> None:
    t_h = Team(name="Home", fifa_code="HOM", confederation="UEFA", group="A")
    t_a = Team(name="Away", fifa_code="AWY", confederation="UEFA", group="A")
    session.add_all([t_h, t_a])
    session.flush()

    m = Match(
        home_team_id=t_h.id, away_team_id=t_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(m)
    session.commit()

    pipeline = FeaturePipeline(session)
    fv = pipeline.build_features(m.id)

    assert fv.market_home_prob == pytest.approx(0.333)
    assert fv.market_num_bookmakers == 0


def test_feature_vector_to_array_consistent_order() -> None:
    fv1 = MatchFeatureVector(match_id=1, timestamp=datetime.now(UTC))
    fv2 = MatchFeatureVector(match_id=2, timestamp=datetime.now(UTC),
                             market_home_prob=0.5, market_draw_prob=0.25, market_away_prob=0.25)

    arr1 = fv1.to_array()
    arr2 = fv2.to_array()

    assert len(arr1) == len(arr2)
    assert arr1[0] == pytest.approx(0.333)
    assert arr2[0] == pytest.approx(0.5)


def test_feature_column_names_match_array_length() -> None:
    fv = MatchFeatureVector(match_id=1, timestamp=datetime.now(UTC))
    arr = fv.to_array()
    d = fv.to_dict()
    assert len(arr) == len(FEATURE_COLUMNS)
    assert len(d) == len(FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        assert col in d
