"""Async client for API-Football (RapidAPI) with caching and persistence."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from src.config import RAW_DATA_DIR
from src.database.models import Injury, Match, Team, TeamForm

CACHE_DIR = RAW_DATA_DIR / "cache" / "api-football"


class MatchStatus(Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"


@dataclass
class TeamData:
    api_id: int
    name: str
    code: str
    country: str
    founded: int | None = None
    logo_url: str | None = None
    venue_name: str | None = None
    venue_city: str | None = None


@dataclass
class MatchData:
    api_id: int
    home_team_api_id: int
    away_team_api_id: int
    date: datetime
    venue: str | None
    round: str
    status: MatchStatus
    home_score: int | None = None
    away_score: int | None = None
    home_penalties: int | None = None
    away_penalties: int | None = None


@dataclass
class MatchStats:
    match_api_id: int
    team_api_id: int
    possession: float | None = None
    shots_total: int | None = None
    shots_on_target: int | None = None
    shots_off_target: int | None = None
    corners: int | None = None
    fouls: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    passes_total: int | None = None
    passes_accurate: int | None = None


@dataclass
class PlayerInjury:
    player_name: str
    team_api_id: int
    injury_type: str
    status: str
    expected_return: datetime | None = None


_team_api_to_db: dict[int, int] = {}
_match_api_to_db: dict[int, int] = {}


class APIFootballClient:
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"

    def __init__(
        self,
        api_key: str,
        cache_dir: Path | None = None,
        league_id: int = 1,
        season: int = 2026,
    ) -> None:
        self.api_key = api_key
        self.league_id = league_id
        self.season = season
        self._semaphore = asyncio.Semaphore(1)
        self._client: httpx.AsyncClient | None = None
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_count = 0
        self._request_limit = 100
        self._backoff_base = 1.5

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.RAPIDAPI_HOST,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0, headers=self._headers
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_world_cup_teams(self) -> list[TeamData]:
        data = await self._make_request(
            "/teams",
            {"league": str(self.league_id), "season": str(self.season)},
        )
        return self._parse_teams(data)

    async def get_world_cup_fixtures(self) -> list[MatchData]:
        data = await self._make_request(
            "/fixtures",
            {"league": str(self.league_id), "season": str(self.season)},
        )
        return self._parse_fixtures(data)

    async def get_match_statistics(self, match_api_id: int) -> list[MatchStats]:
        data = await self._make_request(
            "/fixtures/statistics",
            {"fixture": str(match_api_id)},
        )
        return self._parse_statistics(data, match_api_id)

    async def get_team_fixtures(
        self, team_id: int, last: int = 10
    ) -> list[MatchData]:
        data = await self._make_request(
            "/fixtures",
            {"team": str(team_id), "last": str(last)},
        )
        return self._parse_fixtures(data)

    async def get_head_to_head(
        self, team_a_id: int, team_b_id: int, last: int = 5
    ) -> list[MatchData]:
        data = await self._make_request(
            "/fixtures/headtohead",
            {"h2h": f"{team_a_id}-{team_b_id}", "last": str(last)},
        )
        return self._parse_fixtures(data)

    async def get_team_injuries(self, team_id: int) -> list[PlayerInjury]:
        data = await self._make_request(
            "/injuries",
            {"team": str(team_id), "season": str(self.season)},
        )
        return self._parse_injuries(data, team_id)

    async def get_team_squad(self, team_id: int) -> list[dict[str, Any]]:
        data = await self._make_request(
            "/players/squads",
            {"team": str(team_id)},
        )
        if isinstance(data, dict) and "response" in data:
            response_data: list[dict[str, Any]] = data["response"]
            return response_data
        return []

    def _cache_key(self, endpoint: str, params: dict[str, str]) -> str:
        raw = f"{endpoint}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _read_cache(self, cache_key: str) -> Any | None:
        cache_file = self._cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        return cached.get("data")

    def _write_cache(self, cache_key: str, data: Any) -> None:
        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "data": data}, f)
        except OSError:
            pass

    async def _http_request(self, endpoint: str, params: dict[str, str]) -> Any:
        await self._rate_limit()

        async with self._semaphore:
            client = await self._get_client()
            url = f"{self.BASE_URL}{endpoint}"
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        self._request_count += 1
        return data

    async def _make_request(self, endpoint: str, params: dict[str, str]) -> Any:
        cache_key = self._cache_key(endpoint, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        data = await self._http_request(endpoint, params)
        self._write_cache(cache_key, data)
        return data

    async def _rate_limit(self) -> None:
        if self._request_count >= self._request_limit:
            raise RuntimeError(
                f"Daily request limit of {self._request_limit} reached"
            )
        delay = self._backoff_base ** min(self._request_count, 10)
        await asyncio.sleep(delay)

    @staticmethod
    def _parse_teams(data: Any) -> list[TeamData]:
        if isinstance(data, dict) and "response" in data:
            items = data["response"]
        elif isinstance(data, list):
            items = data
        else:
            return []

        teams: list[TeamData] = []
        for item in items:
            team = item.get("team", item)
            venue = item.get("venue", {}) if isinstance(item, dict) else {}
            teams.append(
                TeamData(
                    api_id=team["id"],
                    name=team["name"],
                    code=team.get("code", ""),
                    country=team.get("country", ""),
                    founded=team.get("founded"),
                    logo_url=team.get("logo"),
                    venue_name=venue.get("name") if isinstance(venue, dict) else None,
                    venue_city=venue.get("city") if isinstance(venue, dict) else None,
                )
            )
        return teams

    @staticmethod
    def _parse_fixtures(data: Any) -> list[MatchData]:
        if isinstance(data, dict) and "response" in data:
            items = data["response"]
        elif isinstance(data, list):
            items = data
        else:
            return []

        matches: list[MatchData] = []
        for item in items:
            fixture = item.get("fixture", item)
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            league_info = item.get("league", {})
            score = item.get("score", {})

            status_str = (
                fixture.get("status", {})
                .get("short", "NS")
                if isinstance(fixture.get("status"), dict)
                else fixture.get("status", "NS")
            )

            status_map = {
                "NS": MatchStatus.SCHEDULED,
                "TBD": MatchStatus.SCHEDULED,
                "1H": MatchStatus.LIVE,
                "HT": MatchStatus.LIVE,
                "2H": MatchStatus.LIVE,
                "ET": MatchStatus.LIVE,
                "P": MatchStatus.LIVE,
                "FT": MatchStatus.FINISHED,
                "AET": MatchStatus.FINISHED,
                "PEN": MatchStatus.FINISHED,
                "SUSP": MatchStatus.SCHEDULED,
                "INT": MatchStatus.SCHEDULED,
                "PST": MatchStatus.SCHEDULED,
                "CANC": MatchStatus.SCHEDULED,
                "ABD": MatchStatus.SCHEDULED,
                "AWD": MatchStatus.FINISHED,
                "WO": MatchStatus.FINISHED,
            }
            status = status_map.get(status_str, MatchStatus.SCHEDULED)

            home_penalties = None
            away_penalties = None
            if isinstance(score, dict):
                p = score.get("penalty", {})
                if isinstance(p, dict):
                    home_penalties = p.get("home")
                    away_penalties = p.get("away")

            round_name = (
                league_info.get("round", "")
                if isinstance(league_info, dict)
                else item.get("round", "")
            )

            match_date = None
            if isinstance(fixture, dict) and fixture.get("date"):
                from datetime import UTC

                try:
                    match_date = datetime.fromisoformat(
                        fixture["date"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    match_date = datetime.now(UTC)

            if match_date is None:
                from datetime import UTC

                match_date = datetime.now(UTC)

            matches.append(
                MatchData(
                    api_id=fixture["id"] if isinstance(fixture, dict) else item.get("id", 0),
                    home_team_api_id=teams.get("home", {}).get("id", 0)
                    if isinstance(teams, dict)
                    else 0,
                    away_team_api_id=teams.get("away", {}).get("id", 0)
                    if isinstance(teams, dict)
                    else 0,
                    date=match_date,
                    venue=(
                        fixture.get("venue", {}).get("name")
                        if isinstance(fixture, dict)
                        else None
                    ),
                    round=round_name,
                    status=status,
                    home_score=goals.get("home") if isinstance(goals, dict) else None,
                    away_score=goals.get("away") if isinstance(goals, dict) else None,
                    home_penalties=home_penalties,
                    away_penalties=away_penalties,
                )
            )
        return matches

    @staticmethod
    def _parse_statistics(data: Any, match_api_id: int) -> list[MatchStats]:
        if isinstance(data, dict) and "response" in data:
            items = data["response"]
        elif isinstance(data, list):
            items = data
        else:
            return []

        stats_list: list[MatchStats] = []
        for team_stats in items:
            team = team_stats.get("team", {})
            team_api_id = team.get("id", 0) if isinstance(team, dict) else 0
            statistics = team_stats.get("statistics", [])

            stat_map: dict[str, Any] = {}
            for s in statistics:
                stype = s.get("type", "")
                value = s.get("value")
                if value is not None:
                    try:
                        if stype in (
                            "Ball Possession",
                            "Pass Accuracy",
                        ):
                            value = (
                                float(str(value).replace("%", "")) / 100
                                if "%" in str(value)
                                else float(value) / 100
                                if isinstance(value, str) and "." not in value
                                else float(value)
                            )
                        else:
                            value = int(value) if value is not None else None
                    except (ValueError, TypeError):
                        value = None
                stat_map[stype] = value

            stats_list.append(
                MatchStats(
                    match_api_id=match_api_id,
                    team_api_id=team_api_id,
                    possession=stat_map.get("Ball Possession"),
                    shots_total=stat_map.get("Total Shots"),
                    shots_on_target=stat_map.get("Shots on Goal"),
                    shots_off_target=stat_map.get("Shots off Goal"),
                    corners=stat_map.get("Corner Kicks"),
                    fouls=stat_map.get("Fouls"),
                    yellow_cards=stat_map.get("Yellow Cards"),
                    red_cards=stat_map.get("Red Cards"),
                    passes_total=stat_map.get("Total Passes"),
                    passes_accurate=stat_map.get("Accurate Passes"),
                )
            )
        return stats_list

    @staticmethod
    def _parse_injuries(data: Any, team_api_id: int) -> list[PlayerInjury]:
        if isinstance(data, dict) and "response" in data:
            items = data["response"]
        elif isinstance(data, list):
            items = data
        else:
            return []

        injuries: list[PlayerInjury] = []
        for item in items:
            player = item.get("player", item)
            injury_info = player.get("injury", {}) if isinstance(player, dict) else {}

            if not isinstance(injury_info, dict):
                continue

            expected_return = None
            return_date = injury_info.get("return")
            if return_date:
                try:
                    expected_return = datetime.fromisoformat(
                        str(return_date).replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            injuries.append(
                PlayerInjury(
                    player_name=player.get("name", "") if isinstance(player, dict) else "",
                    team_api_id=team_api_id,
                    injury_type=injury_info.get("type", "Unknown"),
                    status=injury_info.get("status", "Unknown"),
                    expected_return=expected_return,
                )
            )
        return injuries


def save_teams(session: Session, teams: list[TeamData]) -> dict[int, int]:
    team_map: dict[int, int] = {}
    for td in teams:
        existing = (
            session.query(Team).filter(Team.name == td.name, Team.fifa_code == td.code).first()
        )
        if existing is not None:
            existing.name = td.name
            existing.fifa_code = td.code
            team_map[td.api_id] = existing.id
        else:
            new_team = Team(
                name=td.name,
                fifa_code=td.code,
                confederation="",
                elo_rating=None,
                fifa_rank=None,
                group=None,
            )
            session.add(new_team)
            session.flush()
            team_map[td.api_id] = new_team.id

    session.commit()
    global _team_api_to_db
    _team_api_to_db.update(team_map)
    return team_map


def save_matches(
    session: Session, matches: list[MatchData], team_map: dict[int, int]
) -> list[int]:
    db_ids: list[int] = []
    for md in matches:
        home_db_id = team_map.get(md.home_team_api_id)
        away_db_id = team_map.get(md.away_team_api_id)
        if home_db_id is None or away_db_id is None:
            continue

        existing = (
            session.query(Match)
            .filter(
                Match.home_team_id == home_db_id,
                Match.away_team_id == away_db_id,
                Match.datetime == md.date,
            )
            .first()
        )
        if existing is not None:
            existing.status = md.status.value
            existing.home_score = md.home_score
            existing.away_score = md.away_score
            existing.round = md.round
            existing.venue = md.venue
            db_ids.append(existing.id)
        else:
            group_part = ""
            if " - " in md.round:
                group_part = md.round.split(" - ")[-1]

            new_match = Match(
                home_team_id=home_db_id,
                away_team_id=away_db_id,
                datetime=md.date,
                venue=md.venue,
                city=None,
                round=md.round,
                group=group_part if group_part.isalpha() and len(group_part) <= 2 else None,
                status=md.status.value,
                home_score=md.home_score,
                away_score=md.away_score,
            )
            session.add(new_match)
            session.flush()
            db_ids.append(new_match.id)

    session.commit()
    global _match_api_to_db
    _match_api_to_db.update(
        {md.api_id: db_id for md, db_id in zip(matches, db_ids)}
    )
    return db_ids


def save_match_stats(session: Session, stats: list[MatchStats]) -> int:
    count = 0
    for stat in stats:
        team_db_id = _team_api_to_db.get(stat.team_api_id)
        match_db_id = _match_api_to_db.get(stat.match_api_id)
        if team_db_id is None or match_db_id is None:
            continue

        existing = (
            session.query(TeamForm)
            .filter(
                TeamForm.team_id == team_db_id,
                TeamForm.match_id == match_db_id,
            )
            .first()
        )
        if existing is not None:
            existing.possession = stat.possession
            existing.shots = stat.shots_total
            existing.shots_on_target = stat.shots_on_target
        else:
            form = TeamForm(
                team_id=team_db_id,
                match_id=match_db_id,
                goals_scored=0,
                goals_conceded=0,
                xg=None,
                xga=None,
                possession=stat.possession,
                shots=stat.shots_total,
                shots_on_target=stat.shots_on_target,
                result="",
                is_home=False,
            )
            session.add(form)
        count += 1

    session.commit()
    return count


def save_injuries(session: Session, injuries: list[PlayerInjury]) -> int:
    count = 0
    for inj in injuries:
        team_db_id = _team_api_to_db.get(inj.team_api_id)
        if team_db_id is None:
            continue

        existing = (
            session.query(Injury)
            .filter(
                Injury.team_id == team_db_id,
                Injury.player_name == inj.player_name,
            )
            .first()
        )
        if existing is not None:
            existing.injury_type = inj.injury_type
            existing.status = inj.status
            existing.expected_return = inj.expected_return
        else:
            new_injury = Injury(
                team_id=team_db_id,
                player_name=inj.player_name,
                injury_type=inj.injury_type,
                status=inj.status,
                expected_return=inj.expected_return,
            )
            session.add(new_injury)
        count += 1

    session.commit()
    return count


async def extract_world_cup_data(
    client: APIFootballClient, session: Session
) -> None:
    teams = await client.get_world_cup_teams()
    team_map = save_teams(session, teams)

    fixtures = await client.get_world_cup_fixtures()
    match_ids = save_matches(session, fixtures, team_map)

    for match_data, _db_id in zip(fixtures, match_ids):
        if match_data.status == MatchStatus.FINISHED:
            stats = await client.get_match_statistics(match_data.api_id)
            save_match_stats(session, stats)
