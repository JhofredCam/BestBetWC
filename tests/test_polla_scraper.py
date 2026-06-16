"""Tests for pollamundial.org scraper module."""

import csv
import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from src.database.connection import Base, get_engine, get_session
from src.database.models import Participant, ParticipantPrediction
from src.etl.polla_scraper import (
    PollaMatch,
    PollaParticipant,
    PollaPrediction,
    PollaScraper,
    PollaScraperFallback,
    PollaStandings,
    get_participant_history,
    save_predictions,
    sync_participants,
)


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


def _load_fixture(name: str) -> str:
    path = Path(__file__).parent / "fixtures" / "polla" / name
    return path.read_text(encoding="utf-8")


# ── PollaParticipant dataclass ───────────────────────────────────────────


def test_polla_participant_creation() -> None:
    p = PollaParticipant(
        platform_id="user1",
        name="TestUser",
        total_points=50,
        position=3,
        exact_scores=2,
        correct_results=5,
    )
    assert p.platform_id == "user1"
    assert p.name == "TestUser"
    assert p.total_points == 50
    assert p.position == 3
    assert p.exact_scores == 2
    assert p.correct_results == 5


def test_polla_prediction_creation() -> None:
    ts = datetime(2026, 6, 15, 10, 0)
    p = PollaPrediction(
        platform_match_id="m001",
        participant_platform_id="user1",
        home_goals=2,
        away_goals=1,
        timestamp=ts,
    )
    assert p.platform_match_id == "m001"
    assert p.home_goals == 2
    assert p.away_goals == 1
    assert p.timestamp == ts


def test_polla_match_creation() -> None:
    ts = datetime(2026, 6, 15, 20, 0)
    m = PollaMatch(
        platform_id="m001",
        home_team="Argentina",
        away_team="Brasil",
        datetime=ts,
        round="Grupo A",
        home_score=2,
        away_score=0,
    )
    assert m.platform_id == "m001"
    assert m.home_team == "Argentina"
    assert m.home_score == 2
    assert m.away_score == 0


def test_polla_match_defaults() -> None:
    ts = datetime(2026, 6, 15, 20, 0)
    m = PollaMatch(
        platform_id="m001",
        home_team="Argentina",
        away_team="Brasil",
        datetime=ts,
        round="Grupo A",
    )
    assert m.home_score is None
    assert m.away_score is None


def test_polla_standings_creation() -> None:
    s = PollaStandings(
        participant_id="user1",
        position=1,
        total_points=85,
        round_points=12,
    )
    assert s.participant_id == "user1"
    assert s.position == 1
    assert s.total_points == 85
    assert s.round_points == 12


# ── Scraping Participants ────────────────────────────────────────────────


def test_scrape_participants_from_fixture() -> None:
    html = _load_fixture("participants.html")
    scraper = PollaScraper(headless=True)
    participants = scraper._parse_participants(html)
    assert len(participants) == 15
    assert participants[0].name == "CarlosMaster"
    assert participants[0].total_points == 85
    assert participants[0].position == 1
    assert participants[0].exact_scores == 5
    assert participants[0].correct_results == 12
    last = participants[-1]
    assert last.name == "Pichanga"
    assert last.total_points == 50
    assert last.position == 15


@pytest.mark.asyncio
async def test_scrape_participants_public_method() -> None:
    html = _load_fixture("participants.html")
    scraper = PollaScraper(headless=True)
    result = await scraper.scrape_participants(html=html)
    assert len(result) == 15
    assert result[0].name == "CarlosMaster"


# ── Scraping Predictions ─────────────────────────────────────────────────


def test_scrape_predictions_from_fixture() -> None:
    html = _load_fixture("predictions_match.html")
    scraper = PollaScraper(headless=True)
    predictions = scraper._parse_predictions(html, "m001")
    assert len(predictions) == 15
    assert predictions[0].participant_platform_id == "CarlosMaster"
    assert predictions[0].home_goals == 2
    assert predictions[0].away_goals == 1
    assert predictions[0].platform_match_id == "m001"
    assert predictions[0].timestamp == datetime(2026, 6, 14, 10, 0)


@pytest.mark.asyncio
async def test_scrape_predictions_public_method() -> None:
    html = _load_fixture("predictions_match.html")
    scraper = PollaScraper(headless=True)
    result = await scraper.scrape_predictions("m001", html=html)
    assert len(result) == 15
    assert result[3].away_goals == 1


# ── Scraping All Predictions ─────────────────────────────────────────────


def test_scrape_all_predictions_from_fixture() -> None:
    html = _load_fixture("all_predictions.html")
    scraper = PollaScraper(headless=True)
    result = scraper._parse_all_predictions(html)
    assert len(result) == 2
    assert "m001" in result
    assert "m002" in result
    assert len(result["m001"]) == 2
    assert len(result["m002"]) == 2
    assert result["m001"][0].participant_platform_id == "CarlosMaster"
    assert result["m002"][0].home_goals == 1
    assert result["m002"][0].away_goals == 2


@pytest.mark.asyncio
async def test_scrape_all_predictions_public_method() -> None:
    html = _load_fixture("all_predictions.html")
    scraper = PollaScraper(headless=True)
    result = await scraper.scrape_all_predictions(html=html)
    assert len(result) == 2


# ── Scraping Matches ─────────────────────────────────────────────────────


def test_scrape_matches_from_fixture() -> None:
    html = _load_fixture("matches.html")
    scraper = PollaScraper(headless=True)
    matches = scraper._parse_matches(html)
    assert len(matches) == 4
    assert matches[0].platform_id == "m001"
    assert matches[0].home_team == "Argentina"
    assert matches[0].away_team == "Brasil"
    assert matches[0].round == "Grupo A"
    assert matches[0].home_score is None
    assert matches[0].away_score is None
    assert matches[2].platform_id == "m003"
    assert matches[2].home_score == 2
    assert matches[2].away_score == 0


@pytest.mark.asyncio
async def test_scrape_matches_public_method() -> None:
    html = _load_fixture("matches.html")
    scraper = PollaScraper(headless=True)
    result = await scraper.scrape_matches(html=html)
    assert len(result) == 4


# ── Scraping Standings ───────────────────────────────────────────────────


def test_scrape_standings_from_fixture() -> None:
    html = _load_fixture("standings.html")
    scraper = PollaScraper(headless=True)
    standings = scraper._parse_standings(html)
    assert len(standings) == 15
    assert standings[0].participant_id == "CarlosMaster"
    assert standings[0].position == 1
    assert standings[0].total_points == 85
    assert standings[0].round_points == 12
    assert standings[-1].position == 15
    assert standings[-1].total_points == 50


@pytest.mark.asyncio
async def test_scrape_standings_public_method() -> None:
    html = _load_fixture("standings.html")
    scraper = PollaScraper(headless=True)
    result = await scraper.scrape_standings(html=html)
    assert len(result) == 15


# ── Historical Results ───────────────────────────────────────────────────


def test_scrape_historical_results_filters_finished() -> None:
    html = _load_fixture("matches.html")
    scraper = PollaScraper(headless=True)
    matches = scraper._parse_matches(html)
    results = [
        {
            "platform_match_id": m.platform_id,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "datetime": m.datetime.isoformat(),
            "round": m.round,
            "home_score": m.home_score,
            "away_score": m.away_score,
        }
        for m in matches
        if m.home_score is not None and m.away_score is not None
    ]
    assert len(results) == 1
    assert results[0]["platform_match_id"] == "m003"
    assert results[0]["home_score"] == 2
    assert results[0]["away_score"] == 0


@pytest.mark.asyncio
async def test_scrape_historical_results_public_method() -> None:
    html = _load_fixture("matches.html")
    scraper = PollaScraper(headless=True)
    result = await scraper.scrape_historical_results(html=html)
    assert len(result) == 1
    assert result[0]["home_score"] == 2


# ── Rate Limiting ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_enforces_delay() -> None:
    import time as time_mod

    scraper = PollaScraper(headless=True)
    scraper._rate_limit_seconds = 0.1

    start = time_mod.monotonic()
    await scraper._rate_limit()
    await scraper._rate_limit()
    elapsed = time_mod.monotonic() - start
    assert elapsed >= 0.1


# ── Scraper close ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_when_no_browser() -> None:
    scraper = PollaScraper(headless=True)
    await scraper.close()
    assert scraper._browser is None


# ── Login stubbed ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_login_not_implemented() -> None:
    scraper = PollaScraper(headless=True)
    with pytest.raises(NotImplementedError):
        await scraper.login("user", "pass")


# ── CSV Fallback ─────────────────────────────────────────────────────────


def test_load_from_csv_parses_predictions() -> None:
    csv_content = (
        "participant_name,match_id,home_goals,away_goals,timestamp\n"
        "CarlosMaster,m001,2,1,2026-06-14 10:00\n"
        "FutbolFan,m001,1,0,2026-06-14 09:30\n"
        "ElPronosticador,m002,3,2,2026-06-15 11:00\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_content)
        tmp_path = Path(f.name)

    try:
        fallback = PollaScraperFallback()
        results = fallback.load_from_csv(tmp_path)
        assert len(results) == 3
        assert results[0].participant_platform_id == "CarlosMaster"
        assert results[0].home_goals == 2
        assert results[0].away_goals == 1
        assert results[0].timestamp == datetime(2026, 6, 14, 10, 0)
        assert results[2].platform_match_id == "m002"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_from_csv_skips_invalid_rows() -> None:
    csv_content = (
        "participant_name,match_id,home_goals,away_goals,timestamp\n"
        "CarlosMaster,m001,2,1,2026-06-14 10:00\n"
        "BadUser,m001,BAD,1,2026-06-14 10:00\n"
        "FutbolFan,m001,1,0,2026-06-14 09:30\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_content)
        tmp_path = Path(f.name)

    try:
        fallback = PollaScraperFallback()
        results = fallback.load_from_csv(tmp_path)
        assert len(results) == 2
    finally:
        tmp_path.unlink(missing_ok=True)


def test_export_template_creates_csv() -> None:
    match = PollaMatch(
        platform_id="m001",
        home_team="Argentina",
        away_team="Brasil",
        datetime=datetime(2026, 6, 15, 20, 0),
        round="Grupo A",
    )
    fallback = PollaScraperFallback()
    template_path = fallback.export_template([match])
    try:
        assert template_path.exists()
        content = template_path.read_text(encoding="utf-8")
        assert "participant_name" in content
        assert "match_id" in content
        assert "home_goals" in content
        assert "away_goals" in content
        assert "timestamp" in content
    finally:
        template_path.unlink(missing_ok=True)


# ── Persistence: sync_participants ───────────────────────────────────────


def test_sync_participants_creates_new(session: Session) -> None:
    participants = [
        PollaParticipant(
            platform_id="user1", name="Carlos", total_points=85,
            position=1, exact_scores=5, correct_results=12,
        ),
        PollaParticipant(
            platform_id="user2", name="FutbolFan", total_points=79,
            position=2, exact_scores=4, correct_results=11,
        ),
    ]
    mapping = sync_participants(session, participants)
    assert len(mapping) == 2
    assert "user1" in mapping
    assert "user2" in mapping
    assert isinstance(mapping["user1"], int)

    db_parts = session.query(Participant).all()
    assert len(db_parts) == 2
    assert db_parts[0].name == "Carlos"
    assert db_parts[0].platform_id == "user1"


def test_sync_participants_updates_existing(session: Session) -> None:
    existing = Participant(name="OldName", platform_id="user1")
    session.add(existing)
    session.commit()

    participants = [
        PollaParticipant(
            platform_id="user1", name="NewName", total_points=90,
            position=1, exact_scores=6, correct_results=13,
        ),
    ]
    mapping = sync_participants(session, participants)
    assert mapping["user1"] == existing.id

    db_part = session.query(Participant).filter_by(platform_id="user1").first()
    assert db_part is not None
    assert db_part.name == "NewName"


# ── Persistence: save_predictions ────────────────────────────────────────


def test_save_predictions_inserts(session: Session) -> None:
    participant = Participant(name="Carlos", platform_id="user1")
    session.add(participant)
    session.commit()

    pred = PollaPrediction(
        platform_match_id="m001",
        participant_platform_id="user1",
        home_goals=2,
        away_goals=1,
        timestamp=datetime(2026, 6, 14, 10, 0),
    )
    count = save_predictions(
        session, [pred],
        participant_map={"user1": participant.id},
        match_map={"m001": 1},
    )
    assert count == 1

    db_pred = session.query(ParticipantPrediction).first()
    assert db_pred is not None
    assert db_pred.home_goals == 2
    assert db_pred.away_goals == 1
    assert db_pred.match_id == 1
    assert db_pred.participant_id == participant.id


def test_save_predictions_updates_existing(session: Session) -> None:
    participant = Participant(name="Carlos", platform_id="user1")
    session.add(participant)
    session.commit()

    existing_pred = ParticipantPrediction(
        match_id=1,
        participant_id=participant.id,
        home_goals=1,
        away_goals=0,
        timestamp=datetime(2026, 6, 14, 8, 0),
    )
    session.add(existing_pred)
    session.commit()

    new_data = PollaPrediction(
        platform_match_id="m001",
        participant_platform_id="user1",
        home_goals=3,
        away_goals=2,
        timestamp=datetime(2026, 6, 14, 12, 0),
    )
    count = save_predictions(
        session, [new_data],
        participant_map={"user1": participant.id},
        match_map={"m001": 1},
    )
    assert count == 1

    db_pred = session.query(ParticipantPrediction).first()
    assert db_pred is not None
    assert db_pred.home_goals == 3
    assert db_pred.away_goals == 2


def test_save_predictions_skips_unknown_participant(session: Session) -> None:
    pred = PollaPrediction(
        platform_match_id="m001",
        participant_platform_id="unknown",
        home_goals=2,
        away_goals=1,
        timestamp=datetime(2026, 6, 14, 10, 0),
    )
    count = save_predictions(
        session, [pred],
        participant_map={"user1": 999},
        match_map={"m001": 1},
    )
    assert count == 0


# ── Persistence: get_participant_history ─────────────────────────────────


def test_get_participant_history(session: Session) -> None:
    participant = Participant(name="Carlos", platform_id="user1")
    session.add(participant)
    session.commit()

    pred1 = ParticipantPrediction(
        match_id=1, participant_id=participant.id,
        home_goals=2, away_goals=1,
        timestamp=datetime(2026, 6, 14, 10, 0),
    )
    pred2 = ParticipantPrediction(
        match_id=2, participant_id=participant.id,
        home_goals=0, away_goals=0,
        timestamp=datetime(2026, 6, 14, 10, 0),
    )
    session.add_all([pred1, pred2])
    session.commit()

    history = get_participant_history(session, participant.id)
    assert len(history) == 2
    assert history[0].home_goals == 2
    assert history[1].home_goals == 0


# ── HTML parsing edge cases ──────────────────────────────────────────────


def test_parse_participants_empty_html() -> None:
    scraper = PollaScraper(headless=True)
    results = scraper._parse_participants("<html><body></body></html>")
    assert results == []


def test_parse_predictions_empty_html() -> None:
    scraper = PollaScraper(headless=True)
    results = scraper._parse_predictions("<html><body></body></html>", "m001")
    assert results == []


def test_parse_matches_empty_html() -> None:
    scraper = PollaScraper(headless=True)
    results = scraper._parse_matches("<html><body></body></html>")
    assert results == []


def test_parse_standings_empty_html() -> None:
    scraper = PollaScraper(headless=True)
    results = scraper._parse_standings("<html><body></body></html>")
    assert results == []


def test_parse_all_predictions_empty_html() -> None:
    scraper = PollaScraper(headless=True)
    results = scraper._parse_all_predictions("<html><body></body></html>")
    assert results == {}


# ── Playwright integration test (mocked) ─────────────────────────────────


@pytest.mark.asyncio
async def test_scrape_participants_with_mocked_playwright() -> None:
    html = _load_fixture("participants.html")
    scraper = PollaScraper(headless=True)
    with patch.object(scraper, "_scrape_page", AsyncMock(return_value=html)):
        result = await scraper.scrape_participants()
        assert len(result) == 15


@pytest.mark.asyncio
async def test_scrape_predictions_with_mocked_playwright() -> None:
    html = _load_fixture("predictions_match.html")
    scraper = PollaScraper(headless=True)
    with patch.object(scraper, "_scrape_page", AsyncMock(return_value=html)):
        result = await scraper.scrape_predictions("m001")
        assert len(result) == 15


@pytest.mark.asyncio
async def test_scrape_standings_with_mocked_playwright() -> None:
    html = _load_fixture("standings.html")
    scraper = PollaScraper(headless=True)
    with patch.object(scraper, "_scrape_page", AsyncMock(return_value=html)):
        result = await scraper.scrape_standings()
        assert len(result) == 15


@pytest.mark.asyncio
async def test_scrape_matches_with_mocked_playwright() -> None:
    html = _load_fixture("matches.html")
    scraper = PollaScraper(headless=True)
    with patch.object(scraper, "_scrape_page", AsyncMock(return_value=html)):
        result = await scraper.scrape_matches()
        assert len(result) == 4


@pytest.mark.asyncio
async def test_scrape_all_predictions_with_mocked_playwright() -> None:
    html = _load_fixture("all_predictions.html")
    scraper = PollaScraper(headless=True)
    with patch.object(scraper, "_scrape_page", AsyncMock(return_value=html)):
        result = await scraper.scrape_all_predictions()
        assert len(result) == 2


@pytest.mark.asyncio
async def test_scrape_historical_results_with_mocked_playwright() -> None:
    html = _load_fixture("matches.html")
    scraper = PollaScraper(headless=True)
    with patch.object(scraper, "_scrape_page", AsyncMock(return_value=html)):
        result = await scraper.scrape_historical_results()
        assert len(result) == 1
