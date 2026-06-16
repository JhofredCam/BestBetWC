"""Tests for The Odds API ETL module."""

import json
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from src.database.connection import Base, get_engine, get_session
from src.database.models import CorrectScoreOdds, Match, Odds, Team
from src.etl.odds_api import (
    CachedOddsClient,
    CorrectScoreSnapshot,
    OddsAPIClient,
    OddsSnapshot,
    get_closing_odds,
    save_correct_score_to_db,
    save_odds_to_db,
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


def _make_h2h_outcomes(home: float, draw: float, away: float) -> list[dict]:
    return [
        {"name": "Home Team", "price": home},
        {"name": "Draw", "price": draw},
        {"name": "Away Team", "price": away},
    ]


def _make_totals_outcomes(
    over_15: float, under_15: float,
    over_25: float, under_25: float,
    over_35: float, under_35: float,
) -> list[dict]:
    return [
        {"name": "Over", "price": over_15, "point": 1.5},
        {"name": "Under", "price": under_15, "point": 1.5},
        {"name": "Over", "price": over_25, "point": 2.5},
        {"name": "Under", "price": under_25, "point": 2.5},
        {"name": "Over", "price": over_35, "point": 3.5},
        {"name": "Under", "price": under_35, "point": 3.5},
    ]


def _make_btts_outcomes(yes: float, no: float) -> list[dict]:
    return [
        {"name": "Yes", "price": yes},
        {"name": "No", "price": no},
    ]


def _make_correct_score_outcomes(scores: list[tuple[str, float]]) -> list[dict]:
    return [{"name": name, "price": price} for name, price in scores]


def _make_bookmaker(
    key: str,
    title: str,
    h2h: tuple[float, float, float] | None = None,
    totals: tuple[float, float, float, float, float, float] | None = None,
    btts: tuple[float, float] | None = None,
    correct_scores: list[tuple[str, float]] | None = None,
) -> dict:
    markets: list[dict] = []
    if h2h is not None:
        markets.append({
            "key": "h2h",
            "last_update": "2026-06-14T12:00:00Z",
            "outcomes": _make_h2h_outcomes(*h2h),
        })
    if totals is not None:
        markets.append({
            "key": "totals",
            "last_update": "2026-06-14T12:00:00Z",
            "outcomes": _make_totals_outcomes(*totals),
        })
    if btts is not None:
        markets.append({
            "key": "btts",
            "last_update": "2026-06-14T12:00:00Z",
            "outcomes": _make_btts_outcomes(*btts),
        })
    if correct_scores is not None:
        markets.append({
            "key": "correct_score",
            "last_update": "2026-06-14T12:00:00Z",
            "outcomes": _make_correct_score_outcomes(correct_scores),
        })
    return {"key": key, "title": title, "markets": markets}


def sample_odds_response() -> dict:
    return {
        "id": "match_abc123",
        "bookmakers": [
            _make_bookmaker(
                "pinnacle", "Pinnacle",
                h2h=(2.10, 3.50, 3.20),
                totals=(1.40, 2.75, 1.80, 1.95, 2.50, 1.52),
                btts=(1.70, 2.10),
                correct_scores=[
                    ("1-0", 6.50), ("0-1", 8.00), ("1-1", 6.00),
                    ("2-0", 10.00), ("0-0", 9.00),
                ],
            ),
            _make_bookmaker(
                "bet365", "Bet365",
                h2h=(2.05, 3.40, 3.30),
                totals=(1.38, 2.85, 1.85, 1.90, 2.60, 1.48),
                btts=(1.65, 2.20),
                correct_scores=[("1-0", 6.00), ("0-0", 8.50)],
            ),
        ],
    }


# ── Normalization tests ──────────────────────────────────────────────


def test_implied_probability() -> None:
    assert OddsAPIClient.implied_probability(2.00) == pytest.approx(0.5)
    assert OddsAPIClient.implied_probability(4.00) == pytest.approx(0.25)
    assert OddsAPIClient.implied_probability(1.50) == pytest.approx(0.666666, rel=1e-4)


def test_normalize_two_outcome() -> None:
    prob_a, prob_b = OddsAPIClient.normalize_two_outcome(1.80, 2.00)
    assert prob_a + prob_b == pytest.approx(1.0, abs=0.001)
    assert prob_a > 0.0
    assert prob_b > 0.0


def test_normalize_probabilities_sums_to_one() -> None:
    probs = OddsAPIClient.normalize_probabilities(2.10, 3.50, 3.20)
    assert sum(probs) == pytest.approx(1.0, abs=0.001)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_normalize_probabilities_known_case() -> None:
    home, draw, away = OddsAPIClient.normalize_probabilities(2.0, 3.5, 4.0)
    assert home + draw + away == pytest.approx(1.0, abs=0.001)
    assert home > draw > away


def test_normalize_probabilities_all_equal() -> None:
    home, draw, away = OddsAPIClient.normalize_probabilities(3.0, 3.0, 3.0)
    assert home == pytest.approx(draw)
    assert draw == pytest.approx(away)
    assert home + draw + away == pytest.approx(1.0, abs=0.001)


def test_normalize_probabilities_heavy_favorite() -> None:
    home, draw, away = OddsAPIClient.normalize_probabilities(1.20, 6.00, 12.00)
    assert home > draw > away
    assert home + draw + away == pytest.approx(1.0, abs=0.001)
    assert home == pytest.approx(0.7776, abs=0.01)


def test_normalize_probabilities_all_in_range() -> None:
    test_cases = [
        (1.50, 4.00, 6.00),
        (2.00, 3.50, 3.50),
        (1.10, 8.00, 20.00),
        (3.00, 3.20, 2.40),
    ]
    for h, d, a in test_cases:
        home_p, draw_p, away_p = OddsAPIClient.normalize_probabilities(h, d, a)
        assert 0.0 <= home_p <= 1.0
        assert 0.0 <= draw_p <= 1.0
        assert 0.0 <= away_p <= 1.0
        assert home_p + draw_p + away_p == pytest.approx(1.0, abs=0.001)


# ── Parsing tests ────────────────────────────────────────────────────


def test_parse_h2h_valid() -> None:
    market = {"key": "h2h", "outcomes": _make_h2h_outcomes(2.10, 3.50, 3.20)}
    result = OddsAPIClient._parse_h2h(market)
    assert result == (2.10, 3.50, 3.20)


def test_parse_h2h_none() -> None:
    assert OddsAPIClient._parse_h2h(None) is None


def test_parse_h2h_insufficient_outcomes() -> None:
    market = {"key": "h2h", "outcomes": [{"name": "Home", "price": 2.0}]}
    assert OddsAPIClient._parse_h2h(market) is None


def test_parse_totals() -> None:
    market = {
        "key": "totals",
        "outcomes": _make_totals_outcomes(1.40, 2.75, 1.80, 1.95, 2.50, 1.52),
    }
    client = OddsAPIClient("dummy")
    result = client._parse_totals(market)
    assert result is not None
    assert "over_15_prob" in result
    assert "over_25_prob" in result
    assert "over_35_prob" in result
    for v in result.values():
        assert 0.0 < v < 1.0


def test_parse_totals_none() -> None:
    client = OddsAPIClient("dummy")
    assert client._parse_totals(None) is None


def test_parse_btts() -> None:
    market = {"key": "btts", "outcomes": _make_btts_outcomes(1.70, 2.10)}
    client = OddsAPIClient("dummy")
    result = client._parse_btts(market)
    assert result is not None
    assert "btts_yes_prob" in result
    assert "btts_no_prob" in result
    assert result["btts_yes_prob"] + result["btts_no_prob"] == pytest.approx(1.0, abs=0.001)


def test_parse_btts_none() -> None:
    client = OddsAPIClient("dummy")
    assert client._parse_btts(None) is None


def test_parse_correct_score() -> None:
    outcomes = _make_correct_score_outcomes([("1-0", 6.50), ("2-1", 12.0), ("0-0", 9.0)])
    market = {"key": "correct_score", "outcomes": outcomes}
    result = OddsAPIClient._parse_correct_score(
        market, "match_1", "Pinnacle", datetime.now(UTC),
    )
    assert len(result) == 3
    assert result[0].home_goals == 1
    assert result[0].away_goals == 0
    assert result[0].odds == 6.50
    assert result[0].prob == pytest.approx(1.0 / 6.50)


def test_parse_correct_score_invalid_format_skipped() -> None:
    outcomes = _make_correct_score_outcomes([("1-0", 6.50), ("invalid", 5.0)])
    market = {"key": "correct_score", "outcomes": outcomes}
    result = OddsAPIClient._parse_correct_score(
        market, "match_1", "Pinnacle", datetime.now(UTC),
    )
    assert len(result) == 1
    assert result[0].home_goals == 1


def test_parse_correct_score_none() -> None:
    assert OddsAPIClient._parse_correct_score(
        None, "match_1", "Pinnacle", datetime.now(UTC),
    ) == []


# ── Client tests with mocked HTTP ────────────────────────────────────


@pytest.mark.asyncio
async def test_get_upcoming_matches() -> None:
    mock_response = [
        {"id": "abc123", "home_team": "Argentina", "away_team": "Brazil"},
        {"id": "def456", "home_team": "Germany", "away_team": "France"},
    ]

    client = OddsAPIClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=mock_response)):
        matches = await client.get_upcoming_matches()
        assert len(matches) == 2
        assert matches[0]["id"] == "abc123"


@pytest.mark.asyncio
async def test_get_upcoming_matches_empty() -> None:
    client = OddsAPIClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value={"error": "bad"})):
        matches = await client.get_upcoming_matches()
        assert matches == []


@pytest.mark.asyncio
async def test_get_match_odds_all_markets() -> None:
    response = sample_odds_response()
    client = OddsAPIClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=response)):
        odds, scores = await client.get_match_odds("match_abc123")

    assert len(odds) == 2
    assert odds[0].bookmaker == "Pinnacle"
    assert odds[0].home_odds == 2.10
    assert odds[0].over_15_prob is not None
    assert odds[0].over_25_prob is not None
    assert odds[0].over_35_prob is not None
    assert odds[0].btts_yes_prob is not None
    assert odds[0].btts_no_prob is not None

    assert odds[1].home_prob + odds[1].draw_prob + odds[1].away_prob == pytest.approx(1.0)

    assert len(scores) == 7


@pytest.mark.asyncio
async def test_get_match_odds_missing_markets() -> None:
    response = {
        "id": "match_xyz",
        "bookmakers": [
            _make_bookmaker("test", "TestBook", h2h=(2.0, 3.5, 4.0)),
        ],
    }
    client = OddsAPIClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=response)):
        odds, scores = await client.get_match_odds("match_xyz")

    assert len(odds) == 1
    assert odds[0].over_15_prob is None
    assert odds[0].btts_yes_prob is None
    assert len(scores) == 0


@pytest.mark.asyncio
async def test_get_match_odds_empty_bookmakers() -> None:
    response = {"id": "match_xyz", "bookmakers": []}
    client = OddsAPIClient("dummy")
    with patch.object(client, "_make_request", AsyncMock(return_value=response)):
        odds, scores = await client.get_match_odds("match_xyz")

    assert odds == []
    assert scores == []


@pytest.mark.asyncio
async def test_get_all_odds() -> None:
    mock_matches = [
        {"id": "abc123", "home_team": "A", "away_team": "B"},
        {"id": "def456", "home_team": "C", "away_team": "D"},
    ]
    snapshot = OddsSnapshot(
        match_id="abc123", bookmaker="Test", timestamp=datetime.now(UTC),
        home_odds=2.0, draw_odds=3.5, away_odds=4.0,
        home_prob=0.5, draw_prob=0.2857, away_prob=0.25, margin=0.0357,
    )
    mock_odds = [
        ([snapshot], []),
        ([snapshot], []),
    ]

    client = OddsAPIClient("dummy")
    client._sport_key = "soccer_fifa_world_cup"
    with patch.object(client, "get_upcoming_matches", AsyncMock(return_value=mock_matches)):
        with patch.object(client, "get_match_odds", AsyncMock(side_effect=mock_odds)):
            all_odds, all_scores = await client.get_all_odds()

    assert len(all_odds) == 2
    assert all_odds[0].match_id == "abc123"


# ── Cache tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_avoids_request() -> None:
    mock_data = {"id": "match_1", "bookmakers": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = CachedOddsClient("dummy", cache_ttl=3600)
        client._cache_dir = tmp_dir
        client._sport_key = "soccer_world_cup"

        call_count = 0

        async def counting_request(inst: OddsAPIClient, endpoint: str, params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return mock_data

        with patch.object(OddsAPIClient, "_make_request", counting_request):
            _result = await client.get_match_odds("match_1")
            assert call_count == 1

            _result = await client.get_match_odds("match_1")
            assert call_count == 1
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


@pytest.mark.asyncio
async def test_cache_expiry_refetches() -> None:
    mock_data = {"id": "match_1", "bookmakers": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = CachedOddsClient("dummy", cache_ttl=0)
        client._cache_dir = tmp_dir
        client._sport_key = "soccer_world_cup"

        call_count = 0

        async def counting_request(inst: OddsAPIClient, endpoint: str, params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return mock_data

        with patch.object(OddsAPIClient, "_make_request", counting_request):
            _result = await client.get_match_odds("match_1")
            _result = await client.get_match_odds("match_1")
            assert call_count == 2
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


@pytest.mark.asyncio
async def test_cache_file_created() -> None:
    mock_data = {"id": "match_x", "bookmakers": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = CachedOddsClient("dummy", cache_ttl=3600)
        client._cache_dir = tmp_dir
        client._sport_key = "soccer_world_cup"

        async def fake_request(inst: OddsAPIClient, endpoint: str, params: dict) -> dict:
            return mock_data

        with patch.object(OddsAPIClient, "_make_request", fake_request):
            await client.get_match_odds("match_x")

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
    mock_data = {"id": "match_1", "bookmakers": []}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        client = CachedOddsClient("dummy", cache_ttl=3600)
        client._cache_dir = tmp_dir
        client._sport_key = "soccer_world_cup"

        async def fake_request(inst: OddsAPIClient, endpoint: str, params: dict) -> dict:
            return mock_data

        with patch.object(OddsAPIClient, "_make_request", fake_request):
            await client.get_match_odds("match_1")

        cache_files = list(tmp_dir.glob("*.json"))
        assert len(cache_files) == 1
        cache_files[0].write_text("not json", encoding="utf-8")

        call_count = 0
        async def counting_request(inst: OddsAPIClient, endpoint: str, params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return mock_data

        with patch.object(OddsAPIClient, "_make_request", counting_request):
            _result = await client.get_match_odds("match_1")
            assert call_count == 1
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


# ── Persistence tests ────────────────────────────────────────────────


def test_save_odds_to_db(session: Session) -> None:
    team_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    team_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([team_h, team_a])
    session.flush()

    match = Match(
        home_team_id=team_h.id, away_team_id=team_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(match)
    session.commit()

    snapshot = OddsSnapshot(
        match_id="api_abc", bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.10, draw_odds=3.50, away_odds=3.20,
        home_prob=0.46, draw_prob=0.28, away_prob=0.26,
        over_15_prob=0.64, over_25_prob=0.52, over_35_prob=0.38,
        btts_yes_prob=0.55, btts_no_prob=0.45, margin=0.035,
    )

    count = save_odds_to_db(session, [snapshot], match.id)
    assert count == 1

    saved = session.query(Odds).filter_by(match_id=match.id).first()
    assert saved is not None
    assert saved.bookmaker == "Pinnacle"
    assert saved.home_odds == 2.10
    assert saved.draw_odds == 3.50
    assert saved.away_odds == 3.20
    assert saved.over_15 == pytest.approx(0.64)
    assert saved.over_25 == pytest.approx(0.52)
    assert saved.over_35 == pytest.approx(0.38)
    assert saved.btts_yes == pytest.approx(0.55)
    assert saved.btts_no == pytest.approx(0.45)


def test_save_odds_to_db_multiple(session: Session) -> None:
    team_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    team_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([team_h, team_a])
    session.flush()

    match = Match(
        home_team_id=team_h.id, away_team_id=team_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(match)
    session.commit()

    snapshots = [
        OddsSnapshot(
            match_id="api_abc", bookmaker="Pinnacle",
            timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
            home_odds=2.10, draw_odds=3.50, away_odds=3.20,
            home_prob=0.46, draw_prob=0.28, away_prob=0.26,
        ),
        OddsSnapshot(
            match_id="api_abc", bookmaker="Bet365",
            timestamp=datetime(2026, 6, 14, 13, 0, tzinfo=UTC),
            home_odds=2.05, draw_odds=3.40, away_odds=3.30,
            home_prob=0.47, draw_prob=0.28, away_prob=0.25,
        ),
    ]

    count = save_odds_to_db(session, snapshots, match.id)
    assert count == 2
    assert session.query(Odds).filter_by(match_id=match.id).count() == 2


def test_save_correct_score_to_db(session: Session) -> None:
    team_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    team_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([team_h, team_a])
    session.flush()

    match = Match(
        home_team_id=team_h.id, away_team_id=team_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(match)
    session.commit()

    scores = [
        CorrectScoreSnapshot(
            match_id="api_abc", bookmaker="Pinnacle",
            timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
            home_goals=1, away_goals=0, odds=6.50, prob=0.1538,
        ),
        CorrectScoreSnapshot(
            match_id="api_abc", bookmaker="Pinnacle",
            timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
            home_goals=0, away_goals=0, odds=9.00, prob=0.1111,
        ),
    ]

    count = save_correct_score_to_db(session, scores, match.id)
    assert count == 2

    saved = session.query(CorrectScoreOdds).filter_by(match_id=match.id).all()
    assert len(saved) == 2
    assert saved[0].home_goals == 1
    assert saved[1].away_goals == 0


def test_get_closing_odds_returns_none_when_empty(session: Session) -> None:
    result = get_closing_odds(session, 999)
    assert result is None


def test_get_closing_odds_prefers_is_closing(session: Session) -> None:
    team_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    team_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([team_h, team_a])
    session.flush()

    match = Match(
        home_team_id=team_h.id, away_team_id=team_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(match)
    session.commit()

    early = Odds(
        match_id=match.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
        home_odds=2.20, draw_odds=3.40, away_odds=3.10,
        is_closing=False,
    )
    closing = Odds(
        match_id=match.id, bookmaker="Pinnacle",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.10, draw_odds=3.50, away_odds=3.20,
        is_closing=True,
    )
    session.add_all([early, closing])
    session.commit()

    result = get_closing_odds(session, match.id)
    assert result is not None
    assert result.bookmaker == "Pinnacle"
    assert result.home_odds == 2.10
    assert result.draw_odds == 3.50
    assert result.away_odds == 3.20
    assert result.home_prob + result.draw_prob + result.away_prob == pytest.approx(1.0, abs=0.001)


def test_get_closing_odds_fallback_when_no_is_closing(session: Session) -> None:
    team_h = Team(name="Home", fifa_code="HOM", confederation="UEFA")
    team_a = Team(name="Away", fifa_code="AWY", confederation="UEFA")
    session.add_all([team_h, team_a])
    session.flush()

    match = Match(
        home_team_id=team_h.id, away_team_id=team_a.id,
        datetime=datetime(2026, 6, 15, 20, 0),
        round="group", group="A", status="scheduled",
    )
    session.add(match)
    session.commit()

    odds = Odds(
        match_id=match.id, bookmaker="Bet365",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=UTC),
        home_odds=2.05, draw_odds=3.40, away_odds=3.30,
        is_closing=False,
    )
    session.add(odds)
    session.commit()

    result = get_closing_odds(session, match.id)
    assert result is not None
    assert result.bookmaker == "Bet365"


# ── Edge case tests ──────────────────────────────────────────────────


def test_empty_snapshots_save_returns_zero(session: Session) -> None:
    assert save_odds_to_db(session, [], 1) == 0
    assert save_correct_score_to_db(session, [], 1) == 0


def test_odds_snapshot_defaults() -> None:
    snapshot = OddsSnapshot(
        match_id="test", bookmaker="Test",
        timestamp=datetime.now(UTC),
        home_odds=2.0, draw_odds=3.5, away_odds=4.0,
        home_prob=0.5, draw_prob=0.2857, away_prob=0.25,
    )
    assert snapshot.over_15_prob is None
    assert snapshot.over_25_prob is None
    assert snapshot.btts_yes_prob is None
    assert snapshot.margin == 0.0


@pytest.mark.asyncio
async def test_odds_client_close() -> None:
    client = OddsAPIClient("dummy")
    await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_client_close_when_no_client() -> None:
    client = OddsAPIClient("dummy")
    await client.close()
