"""Web scraper for pollamundial.org using Playwright with CSV fallback.

CSS Selectors documented in this module. If the platform changes, update
the SELECTORS dict and verify against HTML fixtures in tests/fixtures/polla/.
"""

from __future__ import annotations

import asyncio
import csv
import io
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.database.models import Participant, ParticipantPrediction

load_dotenv()

_ = load_dotenv

SELECTORS: dict[str, str] = {

    "participants_table": "table.participants-table",
    "participants_row": "table.participants-table tbody tr",
    "participants_username": "td.username",
    "participants_points": "td:nth-child(3)",
    "participants_exact": "td:nth-child(4)",
    "participants_results": "td:nth-child(5)",


    "predictions_table": "table.predictions-table",
    "predictions_row": "table.predictions-table tbody tr.prediction-row",
    "predictions_username": "td.username",
    "predictions_home_goals": "td.home-goals",
    "predictions_away_goals": "td.away-goals",
    "predictions_timestamp": "td.pred-timestamp",


    "matches_table": "table.matches-table",
    "matches_row": "table.matches-table tbody tr.match-row",
    "matches_id": "td.match-id",
    "matches_home_team": "td.home-team",
    "matches_away_team": "td.away-team",
    "matches_datetime": "td.match-datetime",
    "matches_round": "td.match-round",
    "matches_result": "td.match-result",


    "standings_table": "table.standings-table",
    "standings_row": "table.standings-table tbody tr.standing-row",
    "standings_position": "td.position",
    "standings_username": "td.username",
    "standings_total_points": "td.total-points",
    "standings_round_points": "td.round-points",


    "all_predictions_section": "div.match-section",
    "all_predictions_match_id": "div.match-section[data-match-id]",
}


@dataclass
class PollaParticipant:
    platform_id: str
    name: str
    total_points: int
    position: int
    exact_scores: int
    correct_results: int


@dataclass
class PollaPrediction:
    platform_match_id: str
    participant_platform_id: str
    home_goals: int
    away_goals: int
    timestamp: datetime


@dataclass
class PollaMatch:
    platform_id: str
    home_team: str
    away_team: str
    datetime: datetime
    round: str
    home_score: int | None = None
    away_score: int | None = None


@dataclass
class PollaStandings:
    participant_id: str
    position: int
    total_points: int
    round_points: int


class PollaScraper:
    """Scrapes pollamundial.org for participant predictions and standings."""

    BASE_URL = "https://pollamundial.org"

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self._last_request_time: float = 0.0
        self._rate_limit_seconds: float = 1.0
        self._browser: Any = None
        self._playwright: Any = None
        self._page: Any = None

    async def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_seconds:
            await asyncio.sleep(self._rate_limit_seconds - elapsed)
        self._last_request_time = time.monotonic()

    async def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page()

    async def _scrape_page(self, url: str, wait_selector: str) -> str:
        await self._rate_limit()
        await self._ensure_browser()
        await self._page.goto(url, wait_until="networkidle")
        try:
            await self._page.wait_for_selector(wait_selector, timeout=10000)
        except Exception:
            pass
        return await self._page.content()

    async def login(self, username: str, password: str) -> bool:
        raise NotImplementedError("Login not yet implemented for pollamundial.org")

    async def scrape_participants(self, html: str | None = None) -> list[PollaParticipant]:
        if html is None:
            html = await self._scrape_page(
                f"{self.BASE_URL}/participants",
                SELECTORS["participants_table"],
            )
        return self._parse_participants(html)

    async def scrape_predictions(
        self, match_id: str, html: str | None = None
    ) -> list[PollaPrediction]:
        if html is None:
            html = await self._scrape_page(
                f"{self.BASE_URL}/matches/{match_id}/predictions",
                SELECTORS["predictions_table"],
            )
        return self._parse_predictions(html, match_id)

    async def scrape_all_predictions(
        self, html: str | None = None
    ) -> dict[str, list[PollaPrediction]]:
        if html is None:
            html = await self._scrape_page(
                f"{self.BASE_URL}/predictions",
                SELECTORS["all_predictions_section"],
            )
        return self._parse_all_predictions(html)

    async def scrape_matches(self, html: str | None = None) -> list[PollaMatch]:
        if html is None:
            html = await self._scrape_page(
                f"{self.BASE_URL}/matches",
                SELECTORS["matches_table"],
            )
        return self._parse_matches(html)

    async def scrape_standings(self, html: str | None = None) -> list[PollaStandings]:
        if html is None:
            html = await self._scrape_page(
                f"{self.BASE_URL}/standings",
                SELECTORS["standings_table"],
            )
        return self._parse_standings(html)

    async def scrape_historical_results(self, html: str | None = None) -> list[dict[str, Any]]:
        if html is None:
            html = await self._scrape_page(
                f"{self.BASE_URL}/results",
                SELECTORS["matches_table"],
            )
        matches = self._parse_matches(html)
        return [
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

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._page = None

    # ── Parsing methods ──────────────────────────────────────────────

    def _parse_participants(self, html: str) -> list[PollaParticipant]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select(SELECTORS["participants_row"])
        participants: list[PollaParticipant] = []
        for row in rows:
            try:
                username_el = row.select_one(SELECTORS["participants_username"])
                if username_el is None:
                    continue
                name = username_el.get_text(strip=True)

                position_cell = row.select_one("td:first-child")
                position = int(position_cell.get_text(strip=True)) if position_cell else len(
                    participants
                ) + 1

                points_el = row.select_one(SELECTORS["participants_points"])
                total_points = int(points_el.get_text(strip=True)) if points_el else 0

                exact_el = row.select_one(SELECTORS["participants_exact"])
                exact_scores = int(exact_el.get_text(strip=True)) if exact_el else 0

                results_el = row.select_one(SELECTORS["participants_results"])
                correct_results = int(results_el.get_text(strip=True)) if results_el else 0

                participants.append(
                    PollaParticipant(
                        platform_id=name,
                        name=name,
                        total_points=total_points,
                        position=position,
                        exact_scores=exact_scores,
                        correct_results=correct_results,
                    )
                )
            except (ValueError, AttributeError):
                continue
        return participants

    def _parse_predictions(self, html: str, match_id: str) -> list[PollaPrediction]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select(SELECTORS["predictions_row"])
        predictions: list[PollaPrediction] = []
        for row in rows:
            try:
                username_el = row.select_one(SELECTORS["predictions_username"])
                home_goals_el = row.select_one(SELECTORS["predictions_home_goals"])
                away_goals_el = row.select_one(SELECTORS["predictions_away_goals"])
                timestamp_el = row.select_one(SELECTORS["predictions_timestamp"])

                if None in (username_el, home_goals_el, away_goals_el, timestamp_el):
                    continue

                name = username_el.get_text(strip=True)
                home_goals = int(home_goals_el.get_text(strip=True))
                away_goals = int(away_goals_el.get_text(strip=True))
                ts_str = timestamp_el.get_text(strip=True)
                timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")

                predictions.append(
                    PollaPrediction(
                        platform_match_id=match_id,
                        participant_platform_id=name,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        timestamp=timestamp,
                    )
                )
            except (ValueError, AttributeError):
                continue
        return predictions

    def _parse_all_predictions(self, html: str) -> dict[str, list[PollaPrediction]]:
        soup = BeautifulSoup(html, "lxml")
        sections = soup.select(SELECTORS["all_predictions_section"])
        result: dict[str, list[PollaPrediction]] = {}
        for section in sections:
            match_id = section.get("data-match-id", "")
            if isinstance(match_id, list):
                match_id = match_id[0] if match_id else ""
            if not match_id:
                continue
            result[str(match_id)] = self._parse_predictions(str(section), str(match_id))
        return result

    def _parse_matches(self, html: str) -> list[PollaMatch]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select(SELECTORS["matches_row"])
        matches: list[PollaMatch] = []
        for row in rows:
            try:
                id_el = row.select_one(SELECTORS["matches_id"])
                home_el = row.select_one(SELECTORS["matches_home_team"])
                away_el = row.select_one(SELECTORS["matches_away_team"])
                dt_el = row.select_one(SELECTORS["matches_datetime"])
                round_el = row.select_one(SELECTORS["matches_round"])
                result_el = row.select_one(SELECTORS["matches_result"])

                if None in (id_el, home_el, away_el, dt_el, round_el, result_el):
                    continue

                platform_id = id_el.get_text(strip=True)
                home_team = home_el.get_text(strip=True)
                away_team = away_el.get_text(strip=True)
                dt_str = dt_el.get_text(strip=True)
                match_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                round_name = round_el.get_text(strip=True)
                result_str = result_el.get_text(strip=True)

                home_score: int | None = None
                away_score: int | None = None
                if result_str and result_str != "-":
                    parts = result_str.split("-")
                    if len(parts) == 2:
                        try:
                            home_score = int(parts[0].strip())
                            away_score = int(parts[1].strip())
                        except ValueError:
                            pass

                matches.append(
                    PollaMatch(
                        platform_id=platform_id,
                        home_team=home_team,
                        away_team=away_team,
                        datetime=match_dt,
                        round=round_name,
                        home_score=home_score,
                        away_score=away_score,
                    )
                )
            except (ValueError, AttributeError):
                continue
        return matches

    def _parse_standings(self, html: str) -> list[PollaStandings]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select(SELECTORS["standings_row"])
        standings: list[PollaStandings] = []
        for row in rows:
            try:
                position_el = row.select_one(SELECTORS["standings_position"])
                username_el = row.select_one(SELECTORS["standings_username"])
                total_el = row.select_one(SELECTORS["standings_total_points"])
                round_el = row.select_one(SELECTORS["standings_round_points"])

                if None in (position_el, username_el, total_el, round_el):
                    continue

                standings.append(
                    PollaStandings(
                        participant_id=username_el.get_text(strip=True),
                        position=int(position_el.get_text(strip=True)),
                        total_points=int(total_el.get_text(strip=True)),
                        round_points=int(round_el.get_text(strip=True)),
                    )
                )
            except (ValueError, AttributeError):
                continue
        return standings


# ── CSV Fallback ─────────────────────────────────────────────────────────


class PollaScraperFallback:
    """CSV fallback when web scraping fails."""

    CSV_COLUMNS = [
        "participant_name",
        "match_id",
        "home_goals",
        "away_goals",
        "timestamp",
    ]

    def load_from_csv(self, path: Path) -> list[PollaPrediction]:
        predictions: list[PollaPrediction] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    predictions.append(
                        PollaPrediction(
                            platform_match_id=row["match_id"],
                            participant_platform_id=row["participant_name"],
                            home_goals=int(row["home_goals"]),
                            away_goals=int(row["away_goals"]),
                            timestamp=datetime.strptime(
                                row["timestamp"], "%Y-%m-%d %H:%M"
                            ),
                        )
                    )
                except (ValueError, KeyError):
                    continue
        return predictions

    def export_template(self, matches: list[PollaMatch]) -> Path:
        path = Path("polla_predictions_template.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_COLUMNS)
        return path


# ── Persistence ──────────────────────────────────────────────────────────


def sync_participants(
    session: Session, participants: list[PollaParticipant]
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for p in participants:
        existing = (
            session.query(Participant)
            .filter_by(platform_id=p.platform_id)
            .first()
        )
        if existing is not None:
            existing.name = p.name
            mapping[p.platform_id] = existing.id
        else:
            db_participant = Participant(name=p.name, platform_id=p.platform_id)
            session.add(db_participant)
            session.flush()
            mapping[p.platform_id] = db_participant.id
    session.commit()
    return mapping


def save_predictions(
    session: Session,
    predictions: list[PollaPrediction],
    participant_map: dict[str, int],
    match_map: dict[str, int] | None = None,
) -> int:
    count = 0
    match_db_id = None
    if match_map is not None and predictions:
        match_db_id = match_map.get(predictions[0].platform_match_id)
    for pred in predictions:
        participant_db_id = participant_map.get(pred.participant_platform_id)
        if participant_db_id is None:
            continue
        existing = (
            session.query(ParticipantPrediction)
            .filter_by(
                match_id=match_db_id,
                participant_id=participant_db_id,
            )
            .first()
        )
        if existing is not None:
            existing.home_goals = pred.home_goals
            existing.away_goals = pred.away_goals
            existing.timestamp = pred.timestamp
        else:
            db_pred = ParticipantPrediction(
                match_id=match_db_id,  # type: ignore[arg-type]
                participant_id=participant_db_id,
                home_goals=pred.home_goals,
                away_goals=pred.away_goals,
                timestamp=pred.timestamp,
            )
            session.add(db_pred)
        count += 1
    session.commit()
    return count


def get_participant_history(
    session: Session, participant_id: int
) -> list[ParticipantPrediction]:
    return (
        session.query(ParticipantPrediction)
        .filter_by(participant_id=participant_id)
        .all()
    )
