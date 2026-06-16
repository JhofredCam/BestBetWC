"""Async client for The Odds API with caching and persistence."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from src.config import RAW_DATA_DIR
from src.database.models import CorrectScoreOdds, Odds

CACHE_DIR = RAW_DATA_DIR / "cache"


@dataclass
class OddsSnapshot:
    match_id: str
    bookmaker: str
    timestamp: datetime
    home_odds: float
    draw_odds: float
    away_odds: float
    home_prob: float
    draw_prob: float
    away_prob: float
    over_15_prob: float | None = None
    over_25_prob: float | None = None
    over_35_prob: float | None = None
    btts_yes_prob: float | None = None
    btts_no_prob: float | None = None
    margin: float = 0.0


@dataclass
class CorrectScoreSnapshot:
    match_id: str
    bookmaker: str
    timestamp: datetime
    home_goals: int
    away_goals: int
    odds: float
    prob: float


class OddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    WORLD_CUP_KEY_PATTERNS = [
        "soccer_fifa_world_cup",
        "soccer_world_cup_winner",
        "soccer_fifa_wc",
        "soccer_wc",
        "soccer_world_cup",
    ]

    def __init__(
        self,
        api_key: str,
        cache_dir: Path | None = None,
        sport_key: str = "",
    ) -> None:
        self.api_key = api_key
        self._sport_key = sport_key
        self._semaphore = asyncio.Semaphore(5)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_available_sports(self) -> list[dict[str, Any]]:
        params = {"apiKey": self.api_key}
        data = await self._make_request("/sports", params)
        return data if isinstance(data, list) else []

    async def find_world_cup_key(self) -> str:
        if self._sport_key:
            return self._sport_key
        sports = await self.get_available_sports()
        for sport in sports:
            key = sport.get("key", "")
            if any(pattern == key for pattern in self.WORLD_CUP_KEY_PATTERNS):
                self._sport_key = key
                return key
        for sport in sports:
            key = sport.get("key", "")
            title = sport.get("title", "")
            if "world cup" in key.lower() or "world cup" in title.lower():
                self._sport_key = key
                return key
        self._sport_key = "soccer_world_cup"
        return self._sport_key

    async def _get_sport_key(self) -> str:
        if self._sport_key:
            return self._sport_key
        return await self.find_world_cup_key()

    async def get_upcoming_matches(self, sport: str | None = None) -> list[dict[str, Any]]:
        key = sport or await self._get_sport_key()
        params = {"apiKey": self.api_key}
        data = await self._make_request(f"/sports/{key}/events", params)
        return data if isinstance(data, list) else []

    async def get_match_odds(
        self,
        match_id: str,
        regions: str = "eu",
        markets: str = "h2h,totals,btts,correct_score",
    ) -> tuple[list[OddsSnapshot], list[CorrectScoreSnapshot]]:
        key = await self._get_sport_key()
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
        }
        data = await self._make_request(
            f"/sports/{key}/events/{match_id}/odds", params
        )
        return self._parse_odds_response(data, match_id)

    async def get_all_odds(
        self,
    ) -> tuple[list[OddsSnapshot], list[CorrectScoreSnapshot]]:
        key = await self._get_sport_key()
        try:
            matches = await self.get_upcoming_matches()
        except httpx.HTTPStatusError:
            return await self._get_all_odds_direct(key)

        if not matches:
            return await self._get_all_odds_direct(key)

        all_odds: list[OddsSnapshot] = []
        all_scores: list[CorrectScoreSnapshot] = []
        for match in matches:
            try:
                odds, scores = await self.get_match_odds(match["id"])
            except httpx.HTTPStatusError:
                continue
            all_odds.extend(odds)
            all_scores.extend(scores)
        return all_odds, all_scores

    async def _get_all_odds_direct(
        self, sport_key: str,
    ) -> tuple[list[OddsSnapshot], list[CorrectScoreSnapshot]]:
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h,totals,btts,correct_score",
        }
        data = await self._make_request(f"/sports/{sport_key}/odds", params)
        all_odds: list[OddsSnapshot] = []
        all_scores: list[CorrectScoreSnapshot] = []
        if isinstance(data, list):
            for event in data:
                match_id = str(event.get("id", ""))
                b_odds, b_scores = self._parse_odds_response(event, match_id)
                all_odds.extend(b_odds)
                all_scores.extend(b_scores)
        return all_odds, all_scores

    @staticmethod
    def normalize_probabilities(
        home: float, draw: float, away: float,
    ) -> tuple[float, float, float]:
        margin = 1.0 / home + 1.0 / draw + 1.0 / away
        home_prob = (1.0 / home) / margin
        draw_prob = (1.0 / draw) / margin
        away_prob = (1.0 / away) / margin
        return home_prob, draw_prob, away_prob

    @staticmethod
    def implied_probability(odds: float) -> float:
        return 1.0 / odds

    @staticmethod
    def normalize_two_outcome(odds_a: float, odds_b: float) -> tuple[float, float]:
        margin = 1.0 / odds_a + 1.0 / odds_b
        prob_a = (1.0 / odds_a) / margin
        prob_b = (1.0 / odds_b) / margin
        return prob_a, prob_b

    async def _make_request(self, endpoint: str, params: dict[str, str]) -> Any:
        async with self._semaphore:
            await asyncio.sleep(1.0)
            client = await self._get_client()
            url = f"{self.BASE_URL}{endpoint}"
            response = await client.get(url, params=params)
            if response.status_code >= 400:
                detail = ""
                try:
                    detail = response.text[:500]
                except Exception:
                    pass
                raise httpx.HTTPStatusError(
                    f"{response.status_code}: {detail}",
                    request=response.request,
                    response=response,
                )
            return response.json()

    def _parse_odds_response(
        self, data: Any, match_id: str,
    ) -> tuple[list[OddsSnapshot], list[CorrectScoreSnapshot]]:
        odds_snapshots: list[OddsSnapshot] = []
        score_snapshots: list[CorrectScoreSnapshot] = []

        bookmakers = data.get("bookmakers", []) if isinstance(data, dict) else []
        timestamp = datetime.now(UTC)

        for bm in bookmakers:
            bookmaker = bm.get("title", bm.get("key", "unknown"))
            markets = bm.get("markets", [])
            markets_by_key: dict[str, dict[str, Any]] = {m["key"]: m for m in markets}

            h2h_odds = self._parse_h2h(markets_by_key.get("h2h"))
            totals = self._parse_totals(markets_by_key.get("totals"))
            btts = self._parse_btts(markets_by_key.get("btts"))
            correct_scores = self._parse_correct_score(
                markets_by_key.get("correct_score"), match_id, bookmaker, timestamp,
            )

            if h2h_odds is not None:
                home_odds, draw_odds, away_odds = h2h_odds
                home_prob, draw_prob, away_prob = self.normalize_probabilities(
                    home_odds, draw_odds, away_odds,
                )
                margin = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds

                snapshot = OddsSnapshot(
                    match_id=match_id,
                    bookmaker=bookmaker,
                    timestamp=timestamp,
                    home_odds=home_odds,
                    draw_odds=draw_odds,
                    away_odds=away_odds,
                    home_prob=home_prob,
                    draw_prob=draw_prob,
                    away_prob=away_prob,
                    over_15_prob=totals.get("over_15_prob") if totals else None,
                    over_25_prob=totals.get("over_25_prob") if totals else None,
                    over_35_prob=totals.get("over_35_prob") if totals else None,
                    btts_yes_prob=btts.get("btts_yes_prob") if btts else None,
                    btts_no_prob=btts.get("btts_no_prob") if btts else None,
                    margin=margin,
                )
                odds_snapshots.append(snapshot)

            score_snapshots.extend(correct_scores)

        return odds_snapshots, score_snapshots

    @staticmethod
    def _parse_h2h(h2h_market: dict[str, Any] | None) -> tuple[float, float, float] | None:
        if h2h_market is None:
            return None
        outcomes = h2h_market.get("outcomes", [])
        if len(outcomes) < 3:
            return None
        return (
            float(outcomes[0]["price"]),
            float(outcomes[1]["price"]),
            float(outcomes[2]["price"]),
        )

    def _parse_totals(
        self, totals_market: dict[str, Any] | None,
    ) -> dict[str, float] | None:
        if totals_market is None:
            return None
        outcomes = totals_market.get("outcomes", [])

        prices: dict[float, dict[str, float]] = {}
        for outcome in outcomes:
            point = outcome.get("point")
            if point is None:
                continue
            name = outcome.get("name", "").lower()
            price = float(outcome["price"])
            if point not in prices:
                prices[point] = {}
            prices[point][name] = price

        result: dict[str, float] = {}
        point_key_map = {1.5: "over_15_prob", 2.5: "over_25_prob", 3.5: "over_35_prob"}

        for point, pair in prices.items():
            over_price = pair.get("over")
            under_price = pair.get("under")
            if over_price is None or under_price is None:
                continue
            over_prob, _ = self.normalize_two_outcome(over_price, under_price)
            key = point_key_map.get(point)
            if key:
                result[key] = over_prob

        return result if result else None

    def _parse_btts(
        self, btts_market: dict[str, Any] | None,
    ) -> dict[str, float] | None:
        if btts_market is None:
            return None
        outcomes = btts_market.get("outcomes", [])
        yes_price: float | None = None
        no_price: float | None = None
        for outcome in outcomes:
            name = outcome.get("name", "").lower()
            price = float(outcome["price"])
            if name == "yes":
                yes_price = price
            elif name == "no":
                no_price = price

        if yes_price is not None and no_price is not None:
            yes_prob, no_prob = self.normalize_two_outcome(yes_price, no_price)
            return {"btts_yes_prob": yes_prob, "btts_no_prob": no_prob}
        return None

    @staticmethod
    def _parse_correct_score(
        cs_market: dict[str, Any] | None,
        match_id: str,
        bookmaker: str,
        timestamp: datetime,
    ) -> list[CorrectScoreSnapshot]:
        if cs_market is None:
            return []
        outcomes = cs_market.get("outcomes", [])
        snapshots: list[CorrectScoreSnapshot] = []
        for outcome in outcomes:
            name = outcome.get("name", "")
            price = float(outcome["price"])
            try:
                parts = name.split("-")
                home_goals = int(parts[0])
                away_goals = int(parts[1])
            except (ValueError, IndexError):
                continue
            snapshots.append(
                CorrectScoreSnapshot(
                    match_id=match_id,
                    bookmaker=bookmaker,
                    timestamp=timestamp,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    odds=price,
                    prob=OddsAPIClient.implied_probability(price),
                )
            )
        return snapshots


class CachedOddsClient(OddsAPIClient):
    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,
        sport_key: str = "",
    ) -> None:
        super().__init__(api_key, sport_key=sport_key)
        self.cache_ttl = cache_ttl
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

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
        age = time.time() - cached.get("timestamp", 0)
        if age > self.cache_ttl:
            return None
        return cached["data"]

    def _write_cache(self, cache_key: str, data: Any) -> None:
        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "data": data}, f)
        except OSError:
            pass

    async def _make_request(self, endpoint: str, params: dict[str, str]) -> Any:
        api_params = {k: v for k, v in params.items() if k != "apiKey"}
        cache_key = self._cache_key(endpoint, api_params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached
        data = await super()._make_request(endpoint, params)
        self._write_cache(cache_key, data)
        return data


def save_odds_to_db(
    session: Session, snapshots: list[OddsSnapshot], match_db_id: int,
) -> int:
    count = 0
    for snapshot in snapshots:
        odds = Odds(
            match_id=match_db_id,
            bookmaker=snapshot.bookmaker,
            timestamp=snapshot.timestamp,
            home_odds=snapshot.home_odds,
            draw_odds=snapshot.draw_odds,
            away_odds=snapshot.away_odds,
            over_15=snapshot.over_15_prob,
            over_25=snapshot.over_25_prob,
            over_35=snapshot.over_35_prob,
            btts_yes=snapshot.btts_yes_prob,
            btts_no=snapshot.btts_no_prob,
            is_closing=False,
        )
        session.add(odds)
        count += 1
    session.flush()
    return count


def save_correct_score_to_db(
    session: Session, scores: list[CorrectScoreSnapshot], match_db_id: int,
) -> int:
    count = 0
    for score in scores:
        cs = CorrectScoreOdds(
            match_id=match_db_id,
            bookmaker=score.bookmaker,
            timestamp=score.timestamp,
            home_goals=score.home_goals,
            away_goals=score.away_goals,
            odds=score.odds,
        )
        session.add(cs)
        count += 1
    session.flush()
    return count


def get_closing_odds(session: Session, match_id: int) -> OddsSnapshot | None:
    db_odds = (
        session.query(Odds)
        .filter(Odds.match_id == match_id, Odds.is_closing == True)  # noqa: E712
        .order_by(Odds.timestamp.desc())
        .first()
    )
    if db_odds is None:
        db_odds = (
            session.query(Odds)
            .filter(Odds.match_id == match_id)
            .order_by(Odds.timestamp.desc())
            .first()
        )

    if db_odds is None:
        return None

    home_prob, draw_prob, away_prob = OddsAPIClient.normalize_probabilities(
        db_odds.home_odds, db_odds.draw_odds, db_odds.away_odds,
    )
    margin = 1.0 / db_odds.home_odds + 1.0 / db_odds.draw_odds + 1.0 / db_odds.away_odds

    return OddsSnapshot(
        match_id=str(db_odds.match_id),
        bookmaker=db_odds.bookmaker,
        timestamp=db_odds.timestamp,
        home_odds=db_odds.home_odds,
        draw_odds=db_odds.draw_odds,
        away_odds=db_odds.away_odds,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        over_15_prob=db_odds.over_15,
        over_25_prob=db_odds.over_25,
        over_35_prob=db_odds.over_35,
        btts_yes_prob=db_odds.btts_yes,
        btts_no_prob=db_odds.btts_no,
        margin=margin,
    )
