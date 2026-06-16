# SPEC-011: Polla Scraper

## Status: PLANNED

## Objective

Web scraper para pollamundial.org que extraiga: pronósticos de los 15 participantes,
resultados de partidos pasados, clasificación en tiempo real, y reglas de la polla.

## Dependencies

- **SPEC-002** (models: `Participant`, `ParticipantPrediction`, `Score`)
- **SPEC-007** (API-Football para mapear nombres de equipos)

## Context

La capa de teoría de juegos necesita datos reales de los otros 14 participantes:
qué marcadores eligen, con qué frecuencia aciertan, qué sesgos muestran. Sin web
scraping, el ownership estimation (SPEC-014) debe operar con priors no informativos,
lo cual reduce significativamente el valor de la capa de game theory.

El scraper debe ser **defensivo**:
- Rate limiting conservador (1 req/seg)
- User-Agent rotativo
- Detección de cambios en la estructura HTML
- Fallback: extracción manual via CSV si el scraping falla

## Technical Design

### `src/etl/polla_scraper.py`

```python
from playwright.async_api import Page, Browser
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PollaParticipant:
    platform_id: str       # ID en pollamundial.org
    name: str              # Nombre de usuario
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
    home_score: int | None
    away_score: int | None

@dataclass
class PollaStandings:
    participant_id: str
    position: int
    total_points: int
    round_points: int

class PollaScraper:
    BASE_URL = "https://pollamundial.org"  # URL real a verificar

    def __init__(self, headless: bool = True): ...

    async def login(self, username: str, password: str) -> bool:
        """Login si la plataforma lo requiere."""

    async def scrape_participants(self) -> list[PollaParticipant]:
        """Extrae lista de participantes y su posición actual."""

    async def scrape_predictions(
        self, match_id: str | None = None
    ) -> list[PollaPrediction]:
        """Extrae pronósticos de TODOS los participantes para un partido."""

    async def scrape_all_predictions(
        self,
    ) -> dict[str, list[PollaPrediction]]:
        """Extrae pronósticos para todos los partidos disponibles."""

    async def scrape_matches(self) -> list[PollaMatch]:
        """Extrae calendario de partidos de la plataforma."""

    async def scrape_standings(self) -> list[PollaStandings]:
        """Extrae clasificación actual."""

    async def scrape_historical_results(self) -> list[dict]:
        """Extrae resultados de partidos ya jugados (validación)."""

    async def close(self) -> None:
        """Cierra el navegador."""

class PollaScraperFallback:
    """
    Fallback: si el scraping falla, permitir carga manual de CSV.
    Formato esperado:
    participant_name, match_id, home_goals, away_goals, timestamp
    """

    def load_from_csv(self, path: Path) -> list[PollaPrediction]: ...
    def export_template(self, matches: list[PollaMatch]) -> Path: ...
```

### Funciones de persistencia

```python
def sync_participants(
    session: Session, participants: list[PollaParticipant]
) -> dict[str, int]:
    """Sincroniza participantes. Retorna mapping platform_id -> db_id."""

def save_predictions(
    session: Session, predictions: list[PollaPrediction],
    participant_map: dict[str, int],
) -> int: ...

def get_participant_history(
    session: Session, participant_id: int,
) -> list[ParticipantPrediction]: ...
```

### Estrategia de scraping con Playwright

```python
async def _scrape_page(self, url: str) -> str:
    """
    1. Navegar a URL
    2. Esperar que la tabla de predicciones cargue
    3. Extraer HTML
    4. Parsear con BeautifulSoup
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=self.headless)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        # Esperar elemento específico
        await page.wait_for_selector("table.predictions", timeout=10000)
        html = await page.content()
        await browser.close()
        return html
```

## Acceptance Criteria

- [ ] `scrape_participants()` retorna lista de ~15 participantes
- [ ] `scrape_predictions(match_id)` retorna 15 pronósticos (uno por participante)
- [ ] Las predicciones incluyen home_goals y away_goals correctos
- [ ] `scrape_standings()` retorna clasificación ordenada por posición
- [ ] Rate limiting: mínimo 1 segundo entre requests
- [ ] Si el scraping falla, `load_from_csv()` funciona como fallback
- [ ] `export_template()` genera un CSV vacío con las columnas correctas
- [ ] Los datos persisten correctamente en `participant_predictions`
- [ ] Test: mock de Playwright con HTML fixtures de la plataforma
- [ ] Test: fallback CSV funciona con datos de ejemplo
- [ ] Documentación de selectores CSS usados (por si la web cambia)

## Files to Create

```
src/etl/polla_scraper.py
tests/test_polla_scraper.py
tests/fixtures/polla/           (HTML de ejemplo de pollamundial.org)
```

## Git Workflow

```bash
git checkout -b feature/spec-011-polla-scraper

git add src/etl/polla_scraper.py
git commit -m "feat(SPEC-011): add pollamundial.org scraper with Playwright"

git add tests/test_polla_scraper.py tests/fixtures/polla/
git commit -m "test(SPEC-011): add polla scraper tests with mocked HTML fixtures"

pytest tests/test_polla_scraper.py -v
ruff check src/etl/polla_scraper.py

git checkout main
git merge feature/spec-011-polla-scraper
```

## Notes

- Verificar URL real de pollamundial.org (¿es la plataforma correcta?)
- Si la web requiere login, implementar con variables de entorno POLLA_USERNAME/POLLA_PASSWORD
- Los selectores CSS SON FRÁGILES. Documentar exactamente qué se espera en cada selector
- Si Playwright es muy pesado como dependencia, evaluar `httpx` + `beautifulsoup4` primero
- El fallback CSV es CRÍTICO. Si no se puede scrap, al menos se pueden cargar datos manualmente
- La plataforma puede cambiar entre ahora y el Mundial 2026
