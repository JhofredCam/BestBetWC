# SPEC-007: API-Football ETL

## Status: COMPLETED

## Objective

Cliente async para API-Football (RapidAPI) que extraiga datos deportivos:
resultados históricos, alineaciones, estadísticas de partidos, información de
lesiones y datos de selecciones nacionales.

## Dependencies

- **SPEC-001** (config: `API_FOOTBALL_KEY`)
- **SPEC-002** (models: `Team`, `Match`, `TeamForm`, `Injury`)

## Context

API-Football proporciona la capa de datos deportivos "duros": resultados reales,
goles, posesión, tiros. Estos datos alimentan:
- El modelo Dixon-Coles (resultados históricos para `fit()`)
- Features de rendimiento (forma reciente, rachas)
- Features de contexto (lesiones, suspensiones)

API-Football tiene mejor cobertura de selecciones que otras APIs gratuitas.
Free tier: 100 req/día en RapidAPI.

## Technical Design

### `src/etl/api_football.py`

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class MatchStatus(Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"

@dataclass
class TeamData:
    api_id: int
    name: str
    code: str              # FIFA code: "BRA", "ARG"
    country: str
    founded: int | None
    logo_url: str | None
    venue_name: str | None
    venue_city: str | None

@dataclass
class MatchData:
    api_id: int
    home_team_api_id: int
    away_team_api_id: int
    date: datetime
    venue: str | None
    round: str             # "Group Stage - 1", "Round of 16"
    status: MatchStatus
    home_score: int | None
    away_score: int | None
    home_penalties: int | None
    away_penalties: int | None

@dataclass
class MatchStats:
    match_api_id: int
    team_api_id: int
    possession: float | None
    shots_total: int | None
    shots_on_target: int | None
    shots_off_target: int | None
    corners: int | None
    fouls: int | None
    yellow_cards: int | None
    red_cards: int | None
    passes_total: int | None
    passes_accurate: int | None

@dataclass
class PlayerInjury:
    player_name: str
    team_api_id: int
    injury_type: str
    status: str            # "doubtful", "out"
    expected_return: datetime | None

class APIFootballClient:
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

    def __init__(self, api_key: str, cache_dir: Path | None = None): ...

    async def get_world_cup_teams(self) -> list[TeamData]: ...

    async def get_world_cup_fixtures(self) -> list[MatchData]: ...

    async def get_match_statistics(self, match_api_id: int) -> list[MatchStats]: ...

    async def get_team_fixtures(
        self, team_id: int, last: int = 10
    ) -> list[MatchData]: ...
    """Últimos N partidos de una selección (para forma reciente)."""

    async def get_head_to_head(
        self, team_a_id: int, team_b_id: int, last: int = 5
    ) -> list[MatchData]: ...

    async def get_team_injuries(self, team_id: int) -> list[PlayerInjury]: ...

    async def get_team_squad(self, team_id: int) -> list[dict]: ...
```

### Funciones de persistencia

```python
def save_teams(session: Session, teams: list[TeamData]) -> dict[int, int]:
    """Inserta/actualiza equipos. Retorna mapping api_id -> db_id."""

def save_matches(session: Session, matches: list[MatchData], team_map: dict) -> list[int]:
    """Inserta/actualiza partidos. Retorna lista de db_ids."""

def save_match_stats(session: Session, stats: list[MatchStats]) -> int: ...

def save_injuries(session: Session, injuries: list[PlayerInjury]) -> int: ...
```

### Pipeline de extracción principal

```python
async def extract_world_cup_data(
    client: APIFootballClient, session: Session
) -> None:
    teams = await client.get_world_cup_teams()
    team_map = save_teams(session, teams)

    fixtures = await client.get_world_cup_fixtures()
    match_ids = save_matches(session, fixtures, team_map)

    # Para cada partido FINISHED, extraer stats
    for match_data, db_id in zip(fixtures, match_ids):
        if match_data.status == MatchStatus.FINISHED:
            stats = await client.get_match_statistics(match_data.api_id)
            save_match_stats(session, stats)
```

## Acceptance Criteria

- [ ] `get_world_cup_teams()` retorna las 48 selecciones (o 32 si es dataset histórico)
- [ ] `get_world_cup_fixtures()` retorna calendario completo con rondas
- [ ] `get_match_statistics()` retorna stats para partidos finalizados
- [ ] `get_team_fixtures(team_id, last=10)` retorna exactamente 10 partidos o menos
- [ ] `get_head_to_head()` retorna enfrentamientos directos entre dos selecciones
- [ ] Cache en `data/raw/cache/api-football/` evita requests duplicados
- [ ] Las funciones `save_*` usan insert-or-update para evitar duplicados
- [ ] Test: mock HTTP responses con fixtures de API real
- [ ] Test: funciones de persistencia con SQLite en memoria
- [ ] Rate limiting: respeta 100 req/día con backoff exponencial

## Files to Create

```
src/etl/api_football.py
tests/test_api_football.py
tests/fixtures/api_football/  (JSON de respuestas mock)
```

## Git Workflow

```bash
git checkout -b feature/spec-007-api-football

git add src/etl/api_football.py
git commit -m "feat(SPEC-007): add API-Football client for match data extraction"

git add tests/test_api_football.py tests/fixtures/api_football/
git commit -m "test(SPEC-007): add API-Football client tests with mocked fixtures"

pytest tests/test_api_football.py -v
ruff check src/etl/api_football.py

git checkout main
git merge feature/spec-007-api-football
```

## Notes

- API-Football usa RapidAPI como proxy; el header `x-rapidapi-key` es obligatorio
- Endpoint `/fixtures?league=1&season=2026` para Mundial (league_id=1)
- Endpoint `/teams?league=1&season=2026` para selecciones
- Endpoint `/fixtures/headtohead?h2h={team_a}-{team_b}` para H2H
- Las lesiones vienen de `/injuries?team={id}&season=2026`
- NO hardcodear `league=1`; hacerlo configurable para poder testear con otras ligas
