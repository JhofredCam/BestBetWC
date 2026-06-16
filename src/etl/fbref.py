"""FBref web scraper for xG and advanced match statistics."""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from bs4 import BeautifulSoup, Tag

from src.config import RAW_DATA_DIR

CACHE_DIR = RAW_DATA_DIR / "cache" / "fbref"
DEFAULT_RATE_LIMIT = 3
DEFAULT_RATE_WINDOW = 60
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


@dataclass
class FBrefMatchStats:
    match_url: str
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_xga: float
    away_xga: float
    home_possession: float
    away_possession: float
    home_shots: int
    away_shots: int
    home_shots_on_target: int
    away_shots_on_target: int
    home_passes: int
    away_passes: int
    home_deep_completions: int = 0
    away_deep_completions: int = 0
    home_ppda: float | None = None
    away_ppda: float | None = None
    home_score: int | None = None
    away_score: int | None = None
    match_date: str | None = None


class FBrefScraper:
    BASE_URL = "https://fbref.com/en"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: httpx.AsyncClient | None = None
        self._request_times: list[float] = []
        self._rate_limit = DEFAULT_RATE_LIMIT
        self._rate_window = DEFAULT_RATE_WINDOW

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": USER_AGENT},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _rate_limit_wait(self) -> None:
        now = time.time()
        self._request_times = [
            t for t in self._request_times if t > now - self._rate_window
        ]
        if len(self._request_times) >= self._rate_limit:
            oldest = self._request_times[0]
            wait = oldest - (now - self._rate_window) + 0.5
            if wait > 0:
                await asyncio.sleep(wait)
        self._request_times.append(time.time())

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _read_cache(self, cache_key: str) -> str | None:
        cache_file = self._cache_dir / f"{cache_key}.html"
        if not cache_file.exists():
            return None
        try:
            return cache_file.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_cache(self, cache_key: str, html: str) -> None:
        cache_file = self._cache_dir / f"{cache_key}.html"
        try:
            cache_file.write_text(html, encoding="utf-8")
        except OSError:
            pass

    async def _fetch_url(self, url: str) -> str:
        cache_key = self._cache_key(url)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        await self._rate_limit_wait()
        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        html = response.text

        self._write_cache(cache_key, html)
        return html

    async def scrape_world_cup_matches(self, year: int = 2022) -> list[FBrefMatchStats]:
        comp_url = f"{self.BASE_URL}/comps/1/{year}/schedule/{year}-World-Cup-Scores-and-Fixtures"
        html = await self._fetch_url(comp_url)
        return self._parse_match_table(html)

    async def scrape_team_season_stats(self, team_name: str) -> dict[str, Any]:
        team_slug = team_name.lower().replace(" ", "-")
        url = f"{self.BASE_URL}/squads/{team_slug}/{team_name}-Stats"
        html = await self._fetch_url(url)
        return self._parse_xg_from_shots_table(html)

    async def scrape_match_stats(self, match_url: str) -> FBrefMatchStats | None:
        full_url = f"{self.BASE_URL}{match_url}" if match_url.startswith("/") else match_url
        html = await self._fetch_url(full_url)
        return self._parse_match_detail(html, match_url)

    def _parse_match_table(self, html: str) -> list[FBrefMatchStats]:
        soup = BeautifulSoup(html, "lxml")
        table_el = soup.find("table", id=lambda x: x and "sched" in x)
        if table_el is None:
            table_el = soup.find("table", class_=lambda c: c and "stats_table" in c)
        if table_el is None or not isinstance(table_el, Tag):
            return []

        table = cast(Tag, table_el)
        results: list[FBrefMatchStats] = []
        rows = cast(list[Tag], table.find_all("tr"))

        for row in rows:
            if not isinstance(row, Tag):
                continue
            row_classes = cast(list[str], row.get("class") or [])
            if "thead" in row_classes:
                continue
            cells = row.find_all(["th", "td"])
            if len(cells) < 8:
                continue

            match_link = row.find("a", href=lambda h: h and "/matches/" in h)
            if match_link is None:
                continue

            match_href = str(match_link.get("href", "") if isinstance(match_link, Tag) else "")

            home_team_el = row.find("td", {"data-stat": "home_team"})
            away_team_el = row.find("td", {"data-stat": "away_team"})
            home_xg_el = row.find("td", {"data-stat": "home_xg"})
            away_xg_el = row.find("td", {"data-stat": "away_xg"})
            home_score_el = row.find("td", {"data-stat": "home_score"})
            away_score_el = row.find("td", {"data-stat": "away_score"})
            date_el = row.find("td", {"data-stat": "date"})

            home_team = home_team_el.get_text(strip=True) if home_team_el else ""
            away_team = away_team_el.get_text(strip=True) if away_team_el else ""

            try:
                home_xg = float(home_xg_el.get_text(strip=True)) if home_xg_el else 0.0
            except (ValueError, TypeError):
                home_xg = 0.0
            try:
                away_xg = float(away_xg_el.get_text(strip=True)) if away_xg_el else 0.0
            except (ValueError, TypeError):
                away_xg = 0.0
            try:
                home_score = int(home_score_el.get_text(strip=True)) if home_score_el else None
            except (ValueError, TypeError):
                home_score = None
            try:
                away_score = int(away_score_el.get_text(strip=True)) if away_score_el else None
            except (ValueError, TypeError):
                away_score = None

            match_date = date_el.get_text(strip=True) if date_el else None

            if home_team and away_team:
                results.append(
                    FBrefMatchStats(
                        match_url=match_href,
                        home_team=home_team,
                        away_team=away_team,
                        home_xg=home_xg,
                        away_xg=away_xg,
                        home_xga=away_xg,
                        away_xga=home_xg,
                        home_possession=0.0,
                        away_possession=0.0,
                        home_shots=0,
                        away_shots=0,
                        home_shots_on_target=0,
                        away_shots_on_target=0,
                        home_passes=0,
                        away_passes=0,
                        home_score=home_score,
                        away_score=away_score,
                        match_date=match_date,
                    )
                )

        return results

    def _parse_match_detail(self, html: str, match_url: str) -> FBrefMatchStats | None:
        soup = BeautifulSoup(html, "lxml")

        home_team_el = soup.find("a", href=lambda h: h and "/squads/" in h)
        away_team_el = soup.find_all("a", href=lambda h: h and "/squads/" in h)
        home_team = home_team_el.get_text(strip=True) if home_team_el else ""
        away_team = away_team_el[1].get_text(strip=True) if len(away_team_el) > 1 else ""

        if not home_team or not away_team:
            return None

        shots_table = soup.find("table", id=lambda x: x and "shots" in x)
        home_xg = 0.0
        away_xg = 0.0
        home_shots = 0
        away_shots = 0
        home_sot = 0
        away_sot = 0

        if isinstance(shots_table, Tag):
            tbody = shots_table.find("tbody")
            if isinstance(tbody, Tag):
                for row in cast(list[Tag], tbody.find_all("tr")):
                    cells = row.find_all(["th", "td"])
                    if len(cells) < 5:
                        continue
                    team_text = cells[0].get_text(strip=True)
                    try:
                        xg_val = float(cells[-1].get_text(strip=True))
                    except (ValueError, IndexError):
                        xg_val = 0.0
                    try:
                        sot_text = cells[4].get_text(strip=True)
                        is_sot = sot_text.lower() == "yes"
                    except (IndexError, ValueError):
                        is_sot = False

                    if home_team in team_text or team_text in home_team:
                        home_xg += xg_val
                        home_shots += 1
                        if is_sot:
                            home_sot += 1
                    elif away_team in team_text or team_text in away_team:
                        away_xg += xg_val
                        away_shots += 1
                        if is_sot:
                            away_sot += 1

        home_possession = 50.0
        away_possession = 50.0
        possession_el = soup.find("td", {"data-stat": "possession"})
        if possession_el:
            text = possession_el.get_text(strip=True)
            try:
                home_possession = float(text.replace("%", ""))
                away_possession = 100.0 - home_possession
            except ValueError:
                pass

        home_passes = 0
        away_passes = 0
        passes_table = soup.find("table", id=lambda x: x and "passing" in x)
        if isinstance(passes_table, Tag):
            tbody = passes_table.find("tbody")
            if isinstance(tbody, Tag):
                for row in cast(list[Tag], tbody.find_all("tr")):
                    att_el = row.find("td", {"data-stat": "passes_total"})
                    team_text = row.get_text()
                    if isinstance(att_el, Tag):
                        try:
                            p = int(att_el.get_text(strip=True))
                        except (ValueError, TypeError):
                            p = 0
                        if home_team in team_text:
                            home_passes = p
                        elif away_team in team_text:
                            away_passes = p

        return FBrefMatchStats(
            match_url=match_url,
            home_team=home_team,
            away_team=away_team,
            home_xg=home_xg,
            away_xg=away_xg,
            home_xga=away_xg,
            away_xga=home_xg,
            home_possession=home_possession,
            away_possession=away_possession,
            home_shots=home_shots,
            away_shots=away_shots,
            home_shots_on_target=home_sot,
            away_shots_on_target=away_sot,
            home_passes=home_passes,
            away_passes=away_passes,
        )

    def _parse_xg_from_shots_table(self, html: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", id=lambda x: x and "stats_standard" in x)
        if table is None or not isinstance(table, Tag):
            return {}

        result: dict[str, Any] = {}
        tbody = table.find("tbody")
        if tbody is None or not isinstance(tbody, Tag):
            return result

        for row in cast(list[Tag], tbody.find_all("tr")):
            xg_el = row.find("td", {"data-stat": "xg"})
            xga_el = row.find("td", {"data-stat": "xg_against"})
            pos_el = row.find("td", {"data-stat": "possession"})
            shots_el = row.find("td", {"data-stat": "shots_total"})

            try:
                result["xg"] = float(xg_el.get_text(strip=True)) if xg_el else 0.0
            except (ValueError, TypeError):
                result["xg"] = 0.0
            try:
                result["xga"] = float(xga_el.get_text(strip=True)) if xga_el else 0.0
            except (ValueError, TypeError):
                result["xga"] = 0.0
            try:
                result["possession"] = float(pos_el.get_text(strip=True)) if pos_el else 0.0
            except (ValueError, TypeError):
                result["possession"] = 0.0
            try:
                result["shots"] = int(shots_el.get_text(strip=True)) if shots_el else 0
            except (ValueError, TypeError):
                result["shots"] = 0
            break

        return result
