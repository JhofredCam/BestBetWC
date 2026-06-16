"""Tests for API-Football ETL module."""

import json
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from src.database.connection import Base, get_engine, get_session
from src.database.models import Injury, Match, Team, TeamForm
from src.etl.api_football import (
    APIFootballClient,
    MatchData,
    MatchStats,
    MatchStatus,
    PlayerInjury,
    TeamData,
    _match_api_to_db,
    _team_api_to_db,
    extract_world_cup_data,
    save_injuries,
    save_match_stats,
    save_matches,
    save_teams,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "api_football"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def setup_database() -> Generator[None, None, None]:
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    _team_api_to_db.clear()
    _match_api_to_db.clear()
    yield


@pytest.fixture
def session() -> Generator[Session, None, None]:
    s = get_session()
    try:
        yield s
    finally:
        s.close()


# ── Team parsing ─────────────────────────────────────────────────────


def test_parse_teams_from_fixture() -> None:
    data = _load_fixture("teams")
    teams = APIFootballClient._parse_teams(data)
    assert len(teams) == 32
    assert teams[0].name == "Argentina"
    assert teams[0].code == "ARG"
    assert teams[0].api_id == 1
    assert teams[0].venue_name == "Estadio Monumental"
    assert teams[0].venue_city == "Buenos Aires"


def test_parse_teams_list_format() -> None:
    data = [
        {"id": 99, "name": "Test Team", "code": "TST", "country": "Testland"},
    ]
    teams = APIFootballClient._parse_teams(data)
    assert len(teams) == 1
    assert teams[0].name == "Test Team"


def test_parse_teams_empty() -> None:
    assert APIFootballClient._parse_teams({}) == []
    assert APIFootballClient._parse_teams([]) == []


# ── Fixture parsing ──────────────────────────────────────────────────


def test_parse_fixtures_from_fixture() -> None:
    data = _load_fixture("fixtures")
    matches = APIFootballClient._parse_fixtures(data)
    assert len(matches) == 12
    assert matches[0].api_id == 1001
    assert matches[0].home_team_api_id == 1
    assert matches[0].away_team_api_id == 6
    assert matches[0].status == MatchStatus.FINISHED
    assert matches[0].home_score == 2
    assert matches[0].away_score == 1
    assert matches[0].round == "Group Stage - 1"


def test_parse_fixtures_scheduled_status() -> None:
    data = _load_fixture("fixtures")
    matches = APIFootballClient._parse_fixtures(data)
    future = [m for m in matches if m.status == MatchStatus.SCHEDULED]
    assert len(future) > 0
    assert future[0].home_score is None
    assert future[0].away_score is None


def test_parse_fixtures_empty() -> None:
    assert APIFootballClient._parse_fixtures({}) == []
    assert APIFootballClient._parse_fixtures([]) == []


# ── Statistics parsing ───────────────────────────────────────────────


def test_parse_statistics() -> None:
    data = _load_fixture("match_stats")
    stats = APIFootballClient._parse_statistics(data, 1001)
    assert len(stats) == 2
    assert stats[0].match_api_id == 1001
    assert stats[0].team_api_id == 1
    assert stats[0].possession == pytest.approx(0.58)
    assert stats[0].shots_total == 14
    assert stats[0].shots_on_target == 6
    assert stats[0].corners == 7
    assert stats[0].fouls == 12
    assert stats[0].yellow_cards == 2
    assert stats[0].passes_total == 523
    assert stats[0].passes_accurate == 456
    assert stats[1].team_api_id == 6
    assert stats[1].possession == pytest.approx(0.42)


def test_parse_statistics_empty() -> None:
    assert APIFootballClient._parse_statistics({}, 1) == []
    assert APIFootballClient._parse_statistics([], 1) == []


# ── Injury parsing ───────────────────────────────────────────────────


def test_parse_injuries() -> None:
    data = _load_fixture("injuries")
    injuries = APIFootballClient._parse_injuries(data, 1)
    assert len(injuries) == 2
    assert injuries[0].player_name == "Lionel Messi"
    assert injuries[0].team_api_id == 1
    assert injuries[0].injury_type == "Muscular"
    assert injuries[0].status == "Doubtful"
    assert injuries[0].expected_return == datetime(2026, 6, 15, tzinfo=UTC)
    assert injuries[1].player_name == "Angel Di Maria"
    assert injuries[1].status == "Out"


def test_parse_injuries_empty() -> None:
    assert APIFootballClient._parse_injuries({}, 1) == []
    assert APIFootballClient._parse_injuries([], 1) == []


# ── Client tests with mocked HTTP ────────────────────────────────────


@pytest.mark.asyncio
async def test_get_world_cup_teams() -> None:
    mock_data = _load_fixture("teams")
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        teams = await client.get_world_cup_teams()
    assert len(teams) == 32
    assert isinstance(teams[0], TeamData)


@pytest.mark.asyncio
async def test_get_world_cup_fixtures() -> None:
    mock_data = _load_fixture("fixtures")
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        matches = await client.get_world_cup_fixtures()
    assert len(matches) == 12
    assert matches[0].round == "Group Stage - 1"


@pytest.mark.asyncio
async def test_get_match_statistics() -> None:
    mock_data = _load_fixture("match_stats")
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        stats = await client.get_match_statistics(1001)
    assert len(stats) == 2
    assert stats[0].possession == pytest.approx(0.58)


@pytest.mark.asyncio
async def test_get_team_fixtures_last_10() -> None:
    mock_data = _load_fixture("team_fixtures")
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        matches = await client.get_team_fixtures(1, last=10)
    assert len(matches) == 10
    assert matches[0].home_team_api_id == 6
    assert matches[0].away_team_api_id == 1


@pytest.mark.asyncio
async def test_get_team_fixtures_fewer_than_last() -> None:
    mock_data = {"response": [
        {"fixture": {"id": 5001, "date": "2026-06-01T20:00:00Z", "status": {"short": "FT"}},
         "league": {"round": "Friendly"}, "teams": {"home": {"id": 1}, "away": {"id": 2}},
         "goals": {"home": 1, "away": 0}, "score": {}},
    ]}
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        matches = await client.get_team_fixtures(1, last=10)
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_get_head_to_head() -> None:
    mock_data = _load_fixture("head_to_head")
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        matches = await client.get_head_to_head(1, 2, last=5)
    assert len(matches) == 5
    assert matches[0].home_team_api_id == 2  # Brazil at home
    assert matches[0].away_team_api_id == 1  # Argentina away


@pytest.mark.asyncio
async def test_get_team_injuries() -> None:
    mock_data = _load_fixture("injuries")
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        injuries = await client.get_team_injuries(1)
    assert len(injuries) == 2
    assert isinstance(injuries[0], PlayerInjury)


@pytest.mark.asyncio
async def test_get_team_squad() -> None:
    mock_data = {"response": [
        {"id": 101, "name": "L. Messi", "position": "Forward", "number": 10},
        {"id": 102, "name": "E. Martinez", "position": "Goalkeeper", "number": 23},
    ]}
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)):
        squad = await client.get_team_squad(1)
    assert len(squad) == 2
    assert squad[0]["name"] == "L. Messi"


# ── Cache tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_avoids_duplicate_requests() -> None:
    mock_data = {"response": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = APIFootballClient("dummy", cache_dir=tmp_dir)
        call_count = 0

        async def counting_request(_self: APIFootballClient, endpoint: str, params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return mock_data

        with patch.object(APIFootballClient, "_http_request", counting_request):
            await client.get_world_cup_teams()
            assert call_count == 1

            await client.get_world_cup_teams()
            assert call_count == 1
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


@pytest.mark.asyncio
async def test_cache_file_created() -> None:
    mock_data = {"response": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = APIFootballClient("dummy", cache_dir=tmp_dir)

        async def fake_request(_self: APIFootballClient, endpoint: str, params: dict) -> dict:
            return mock_data

        with patch.object(APIFootballClient, "_http_request", fake_request):
            await client.get_world_cup_teams()

        cache_files = list(tmp_dir.glob("*.json"))
        assert len(cache_files) == 1
        with open(cache_files[0], encoding="utf-8") as f:
            cached = json.load(f)
        assert cached["data"] == mock_data
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


@pytest.mark.asyncio
async def test_cache_corrupted_file_handled() -> None:
    mock_data = {"response": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = APIFootballClient("dummy", cache_dir=tmp_dir)

        async def fake_request(_self: APIFootballClient, endpoint: str, params: dict) -> dict:
            return mock_data

        with patch.object(APIFootballClient, "_http_request", fake_request):
            await client.get_world_cup_teams()

        cache_files = list(tmp_dir.glob("*.json"))
        assert len(cache_files) == 1
        cache_files[0].write_text("not json", encoding="utf-8")

        call_count = 0
        async def counting_request(_self: APIFootballClient, endpoint: str, params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return mock_data

        with patch.object(APIFootballClient, "_http_request", counting_request):
            await client.get_world_cup_teams()
            assert call_count == 1
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


# ── Configurable league/season ───────────────────────────────────────


@pytest.mark.asyncio
async def test_configurable_league_id() -> None:
    client = APIFootballClient("dummy", league_id=2, season=2022)
    mock_data = {"response": []}
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_data)) as mock_req:
        await client.get_world_cup_teams()
        call_args = mock_req.call_args[0]
        assert call_args[0] == "/teams"
        assert "league" in call_args[1]
        assert call_args[1]["league"] == "2"
        assert call_args[1]["season"] == "2022"


# ── Persistence: save_teams ──────────────────────────────────────────


def test_save_teams_insert(session: Session) -> None:
    teams = [
        TeamData(api_id=1, name="Argentina", code="ARG", country="Argentina"),
        TeamData(api_id=2, name="Brazil", code="BRA", country="Brazil"),
    ]
    team_map = save_teams(session, teams)
    assert len(team_map) == 2
    assert team_map[1] > 0
    assert team_map[2] > 0
    assert session.query(Team).count() == 2

    stored = session.query(Team).filter_by(fifa_code="ARG").first()
    assert stored is not None
    assert stored.name == "Argentina"


def test_save_teams_update_existing(session: Session) -> None:
    existing = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL")
    session.add(existing)
    session.commit()

    teams = [TeamData(api_id=1, name="Argentina", code="ARG", country="Argentina")]
    team_map = save_teams(session, teams)
    assert len(team_map) == 1
    assert session.query(Team).count() == 1
    assert team_map[1] == existing.id


def test_save_teams_empty(session: Session) -> None:
    team_map = save_teams(session, [])
    assert team_map == {}


# ── Persistence: save_matches ────────────────────────────────────────


def test_save_matches_insert(session: Session) -> None:
    team_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL")
    team_a = Team(name="Brazil", fifa_code="BRA", confederation="CONMEBOL")
    session.add_all([team_h, team_a])
    session.flush()
    team_map = {1: team_h.id, 2: team_a.id}
    session.commit()

    matches = [
        MatchData(
            api_id=1001,
            home_team_api_id=1,
            away_team_api_id=2,
            date=datetime(2026, 6, 11, 20, 0, tzinfo=UTC),
            venue="MetLife Stadium",
            round="Group Stage - 1",
            status=MatchStatus.SCHEDULED,
        ),
    ]
    db_ids = save_matches(session, matches, team_map)
    assert len(db_ids) == 1
    assert db_ids[0] > 0

    stored = session.query(Match).first()
    assert stored is not None
    assert stored.home_team_id == team_h.id
    assert stored.away_team_id == team_a.id
    assert stored.round == "Group Stage - 1"
    assert stored.status == "scheduled"


def test_save_matches_update_existing(session: Session) -> None:
    team_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL")
    team_a = Team(name="Brazil", fifa_code="BRA", confederation="CONMEBOL")
    session.add_all([team_h, team_a])
    session.flush()
    team_map = {1: team_h.id, 2: team_a.id}

    existing_match = Match(
        home_team_id=team_h.id,
        away_team_id=team_a.id,
        datetime=datetime(2026, 6, 11, 20, 0, tzinfo=UTC),
        round="Group Stage - 1",
        group="A",
        status="scheduled",
    )
    session.add(existing_match)
    session.commit()

    matches = [
        MatchData(
            api_id=1001,
            home_team_api_id=1,
            away_team_api_id=2,
            date=datetime(2026, 6, 11, 20, 0, tzinfo=UTC),
            venue="MetLife Stadium",
            round="Group Stage - 1",
            status=MatchStatus.FINISHED,
            home_score=2,
            away_score=1,
        ),
    ]
    db_ids = save_matches(session, matches, team_map)
    assert len(db_ids) == 1
    assert db_ids[0] == existing_match.id
    assert session.query(Match).count() == 1

    updated = session.query(Match).first()
    assert updated is not None
    assert updated.status == "finished"
    assert updated.home_score == 2
    assert updated.away_score == 1


def test_save_matches_missing_team_skipped(session: Session) -> None:
    matches = [
        MatchData(
            api_id=1,
            home_team_api_id=999,
            away_team_api_id=888,
            date=datetime(2026, 6, 11, 20, 0, tzinfo=UTC),
            venue=None,
            round="Group Stage",
            status=MatchStatus.SCHEDULED,
        ),
    ]
    db_ids = save_matches(session, matches, {})
    assert len(db_ids) == 0


# ── Persistence: save_match_stats ────────────────────────────────────


def test_save_match_stats(session: Session) -> None:
    team_h = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL")
    team_a = Team(name="England", fifa_code="ENG", confederation="UEFA")
    session.add_all([team_h, team_a])
    session.flush()

    match = Match(
        home_team_id=team_h.id,
        away_team_id=team_a.id,
        datetime=datetime(2026, 6, 11, 20, 0, tzinfo=UTC),
        round="Group Stage - 1",
        group="A",
        status="finished",
        home_score=2,
        away_score=1,
    )
    session.add(match)
    session.commit()

    _team_api_to_db.clear()
    _match_api_to_db.clear()
    _team_api_to_db[1] = team_h.id
    _team_api_to_db[6] = team_a.id
    _match_api_to_db[1001] = match.id

    stats = [
        MatchStats(
            match_api_id=1001,
            team_api_id=1,
            possession=0.58,
            shots_total=14,
            shots_on_target=6,
            corners=7,
        ),
        MatchStats(
            match_api_id=1001,
            team_api_id=6,
            possession=0.42,
            shots_total=9,
            shots_on_target=3,
            corners=4,
        ),
    ]
    count = save_match_stats(session, stats)
    assert count == 2

    saved = session.query(TeamForm).all()
    assert len(saved) == 2
    assert saved[0].team_id == team_h.id
    assert saved[0].possession == pytest.approx(0.58)
    assert saved[0].shots == 14


def test_save_match_stats_empty(session: Session) -> None:
    assert save_match_stats(session, []) == 0


# ── Persistence: save_injuries ───────────────────────────────────────


def test_save_injuries(session: Session) -> None:
    team = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL")
    session.add(team)
    session.commit()

    _team_api_to_db.clear()
    _team_api_to_db[1] = team.id

    injuries = [
        PlayerInjury(
            player_name="L. Messi",
            team_api_id=1,
            injury_type="Muscular",
            status="Doubtful",
            expected_return=datetime(2026, 6, 15, tzinfo=UTC),
        ),
        PlayerInjury(
            player_name="A. Di Maria",
            team_api_id=1,
            injury_type="Knee",
            status="Out",
            expected_return=None,
        ),
    ]
    count = save_injuries(session, injuries)
    assert count == 2

    saved = session.query(Injury).all()
    assert len(saved) == 2
    assert saved[0].player_name == "L. Messi"
    assert saved[0].team_id == team.id
    assert saved[0].injury_type == "Muscular"
    assert saved[1].player_name == "A. Di Maria"
    assert saved[1].status == "Out"


def test_save_injuries_update_existing(session: Session) -> None:
    team = Team(name="Argentina", fifa_code="ARG", confederation="CONMEBOL")
    session.add(team)
    session.commit()

    existing_injury = Injury(
        team_id=team.id,
        player_name="L. Messi",
        injury_type="Old Injury",
        status="Unknown",
    )
    session.add(existing_injury)
    session.commit()

    _team_api_to_db.clear()
    _team_api_to_db[1] = team.id

    injuries = [
        PlayerInjury(
            player_name="L. Messi",
            team_api_id=1,
            injury_type="Muscular",
            status="Doubtful",
        ),
    ]
    count = save_injuries(session, injuries)
    assert count == 1
    assert session.query(Injury).count() == 1

    updated = session.query(Injury).first()
    assert updated is not None
    assert updated.injury_type == "Muscular"
    assert updated.status == "Doubtful"


def test_save_injuries_empty(session: Session) -> None:
    assert save_injuries(session, []) == 0


# ── Pipeline test ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_world_cup_data_pipeline(session: Session) -> None:
    teams_fixture = _load_fixture("teams")
    fixtures_fixture = _load_fixture("fixtures")
    stats_fixture = _load_fixture("match_stats")

    client = APIFootballClient("dummy")

    responses = [teams_fixture, fixtures_fixture]
    # Need one stats response per finished match (7 finished matches in fixture)
    finished_count = sum(
        1 for m in APIFootballClient._parse_fixtures(fixtures_fixture)
        if m.status == MatchStatus.FINISHED
    )
    responses.extend([stats_fixture] * finished_count)

    call_idx = 0

    async def sequential_response(_self: APIFootballClient, endpoint: str, params: dict) -> dict:
        nonlocal call_idx
        result = responses[call_idx]
        call_idx += 1
        return result

    with patch.object(APIFootballClient, "_http_request", side_effect=responses):
        await extract_world_cup_data(client, session)

    assert session.query(Team).count() == 32
    assert session.query(Match).count() > 0


# ── Rate limiting test ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_raises_when_exceeded() -> None:
    client = APIFootballClient("dummy")
    client._request_count = 100
    with pytest.raises(RuntimeError, match="request limit"):
        await client._rate_limit()


# ── Client lifecycle ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_close() -> None:
    client = APIFootballClient("dummy")
    await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_client_close_no_client() -> None:
    client = APIFootballClient("dummy")
    await client.close()


# ── Edge cases ───────────────────────────────────────────────────────


def test_match_status_enum_values() -> None:
    assert MatchStatus.SCHEDULED.value == "scheduled"
    assert MatchStatus.LIVE.value == "live"
    assert MatchStatus.FINISHED.value == "finished"


def test_team_data_defaults() -> None:
    td = TeamData(api_id=1, name="Test", code="TST", country="Testland")
    assert td.founded is None
    assert td.logo_url is None
    assert td.venue_name is None
    assert td.venue_city is None


def test_match_data_defaults() -> None:
    md = MatchData(
        api_id=1,
        home_team_api_id=2,
        away_team_api_id=3,
        date=datetime.now(UTC),
        venue=None,
        round="Group Stage",
        status=MatchStatus.SCHEDULED,
    )
    assert md.home_score is None
    assert md.away_score is None
    assert md.home_penalties is None
    assert md.away_penalties is None


def test_match_stats_defaults() -> None:
    ms = MatchStats(match_api_id=1, team_api_id=2)
    assert ms.possession is None
    assert ms.shots_total is None
    assert ms.corners is None


def test_player_injury_defaults() -> None:
    pi = PlayerInjury(player_name="Test", team_api_id=1, injury_type="Test", status="Unknown")
    assert pi.expected_return is None


@pytest.mark.asyncio
async def test_get_team_squad_empty() -> None:
    client = APIFootballClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value={"error": "not found"})):
        squad = await client.get_team_squad(1)
    assert squad == []


@pytest.mark.asyncio
async def test_parse_fixtures_with_penalties() -> None:
    data = {
        "response": [{
            "fixture": {"id": 9999, "date": "2026-07-19T20:00:00Z", "status": {"short": "PEN"}},
            "league": {"round": "Final"},
            "teams": {"home": {"id": 1}, "away": {"id": 2}},
            "goals": {"home": 1, "away": 1},
            "score": {"penalty": {"home": 4, "away": 2}},
        }]
    }
    matches = APIFootballClient._parse_fixtures(data)
    assert len(matches) == 1
    assert matches[0].home_penalties == 4
    assert matches[0].away_penalties == 2


def test_save_teams_populates_module_mapping(session: Session) -> None:
    _team_api_to_db.clear()
    teams = [TeamData(api_id=77, name="Testopia", code="TTP", country="Testland")]
    team_map = save_teams(session, teams)
    assert _team_api_to_db[77] == team_map[77]
