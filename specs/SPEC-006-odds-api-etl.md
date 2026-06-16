# SPEC-006: The Odds API ETL

## Status: PLANNED

## Objective

Cliente async para The Odds API que extraiga cuotas de apuestas (1X2, over/under,
BTTS, correct score, asian handicap) para cada partido del Mundial 2026, las
normalice a probabilidades implícitas y las persista en la base de datos.

## Dependencies

- **SPEC-001** (config: `THE_ODDS_API_KEY`)
- **SPEC-002** (models: `Odds`, `CorrectScoreOdds`)

## Context

Las cuotas de apuestas son la fuente de información más valiosa antes de cualquier
modelo estadístico. El mercado de apuestas agrega inteligencia colectiva y es
generalmente más preciso que cualquier modelo individual para predecir resultados.

Esta spec debe:
1. Conectarse a The Odds API (50+ casas de apuestas)
2. Extraer cuotas para todos los partidos disponibles
3. Normalizar a probabilidades implícitas (removiendo el margen de la casa)
4. Persistir snapshots con timestamp para tracking de movimiento de cuotas
5. Cachear respuestas para minimizar requests (free tier: 500 req/mes)

## Technical Design

### `src/etl/odds_api.py`

```python
import httpx
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OddsSnapshot:
    match_id: str
    bookmaker: str
    timestamp: datetime
    home_odds: float
    draw_odds: float
    away_odds: float
    home_prob: float      # Probabilidad implícita normalizada
    draw_prob: float
    away_prob: float
    over_15_prob: float | None
    over_25_prob: float | None
    over_35_prob: float | None
    btts_yes_prob: float | None
    btts_no_prob: float | None
    margin: float          # Overround de la casa

@dataclass
class CorrectScoreOdds:
    match_id: str
    bookmaker: str
    timestamp: datetime
    home_goals: int
    away_goals: int
    odds: float
    prob: float

class OddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, cache_dir: Path | None = None): ...

    async def get_upcoming_matches(self, sport: str = "soccer_world_cup") -> list[dict]: ...

    async def get_match_odds(
        self, match_id: str, regions: str = "eu",
        markets: str = "h2h,totals,btts,correct_score",
    ) -> list[OddsSnapshot]: ...

    async def get_all_odds(self) -> list[OddsSnapshot]: ...

    def normalize_probabilities(
        self, home: float, draw: float, away: float
    ) -> tuple[float, float, float]: ...
    """Convierte cuotas decimales a probabilidades, eliminando el overround."""

    def implied_probability(self, odds: float) -> float:
        """1 / odds, ajustado por margen."""
```

### Método de normalización

Para remover el overround (margen de la casa) y obtener probabilidades que sumen 1:

```
margin = 1/home_odds + 1/draw_odds + 1/away_odds
home_prob = (1 / home_odds) / margin
draw_prob = (1 / draw_odds) / margin
away_prob = (1 / away_odds) / margin
```

### Cache strategy

```python
class CachedOddsClient(OddsAPIClient):
    """Wrapper que persiste respuestas JSON en data/raw/cache/"""

    def __init__(self, api_key: str, cache_ttl: int = 3600): ...
    # TTL de 1 hora: las cuotas no cambian tan rápido antes del partido
```

### Funciones de persistencia

```python
def save_odds_to_db(session: Session, snapshots: list[OddsSnapshot]) -> int: ...
def save_correct_score_to_db(session: Session, scores: list[CorrectScoreOdds]) -> int: ...
def get_closing_odds(session: Session, match_id: int) -> OddsSnapshot | None: ...
```

## Acceptance Criteria

- [ ] `OddsAPIClient.get_match_odds()` retorna todos los mercados para un partido
- [ ] `normalize_probabilities()` produce probabilidades que suman 1.0 (±0.001)
- [ ] El cache en `data/raw/cache/` evita requests duplicados en el mismo TTL
- [ ] `save_odds_to_db()` persiste correctamente en la tabla `odds`
- [ ] Las probabilidades normalizadas son >= 0 y <= 1
- [ ] Test: mock HTTP responses para no depender de API key real
- [ ] Test: normalización con casos conocidos (ej: 2.0, 3.5, 4.0)
- [ ] Rate limiting: no excede 10 req/min (respetar free tier)

## Files to Create

```
src/etl/odds_api.py
tests/test_odds_api.py
```

## Git Workflow

```bash
git checkout -b feature/spec-006-odds-api

git add src/etl/odds_api.py
git commit -m "feat(SPEC-006): add The Odds API client with async support"

git add tests/test_odds_api.py
git commit -m "test(SPEC-006): add odds API client tests with mocked responses"

pytest tests/test_odds_api.py -v
ruff check src/etl/odds_api.py

git checkout main
git merge feature/spec-006-odds-api
```

## Notes

- Free tier: 500 requests/month. Cache agresivo es obligatorio.
- La API devuelve `h2h` (1X2), `totals` (over/under), `btts`, y `correct_score` como mercados separados.
- Usar `regions=eu` para cuotas decimales (más fáciles de procesar que americanas).
- El campo `bookmakers[].markets[].outcomes[]` contiene las cuotas individuales.
- NO hardcodear la API key. Leer de `THE_ODDS_API_KEY` en `.env`.
