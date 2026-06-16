# SPEC-008: Data Pipeline (FBref + Feature Engineering)

## Status: COMPLETED

## Objective

Pipeline completo que: (1) extrae xG y estadísticas avanzadas de FBref/Understat
via web scraping, (2) combina datos de todas las fuentes (odds API, API-Football,
FBref), y (3) genera el feature vector final para cada partido.

## Dependencies

- **SPEC-002** (database models)
- **SPEC-006** (odds API data)
- **SPEC-007** (API-Football data)

## Context

FBref es la fuente más completa de xG (expected goals) gratuito para selecciones.
Understat complementa con datos históricos. El pipeline debe:

1. Scrapear FBref para xG, xGA, posesión, tiros (por partido y acumulados)
2. Unificar datos de las 3 fuentes (odds + API-Football + FBref)
3. Calcular features derivadas: diferencias, ratios, rolling averages
4. Producir el `MatchFeatureVector` que alimenta los modelos de predicción

## Technical Design

### `src/etl/fbref.py`

```python
from dataclasses import dataclass

@dataclass
class FBrefMatchStats:
    match_url: str
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_xga: float          # xG en contra
    away_xga: float
    home_possession: float
    away_possession: float
    home_shots: int
    away_shots: int
    home_shots_on_target: int
    away_shots_on_target: int
    home_passes: int
    away_passes: int
    home_deep_completions: int  # Pases completados en área rival
    away_deep_completions: int
    home_ppda: float | None     # Passes per defensive action (presión)
    away_ppda: float | None

class FBrefScraper:
    BASE_URL = "https://fbref.com/en"

    def __init__(self, cache_dir: Path | None = None): ...

    async def scrape_world_cup_matches(self, year: int = 2022) -> list[FBrefMatchStats]:
        """Scrapea todos los partidos de un Mundial. Usa 2022 para backtesting."""

    async def scrape_team_season_stats(self, team_name: str) -> dict:
        """Estadísticas acumuladas de una selección en el último año."""

    def _parse_match_table(self, html: str) -> list[FBrefMatchStats]: ...
    def _parse_xg_from_shots_table(self, html: str) -> dict: ...
```

### Scraping Strategy

- Usar `httpx` + `beautifulsoup4` para requests HTTP y parseo HTML
- FBref bloquea scrapers agresivos → rate limit de 3 req/min, User-Agent realista
- Cachear HTML crudo en `data/raw/cache/fbref/` para desarrollo
- Las URLs de partidos siguen el patrón: `/en/matches/{match_id}/Team-A-Team-B-Date`

### `src/features/pipeline.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class MatchFeatureVector:
    """Feature vector completo para un partido, listo para el modelo."""

    match_id: int
    timestamp: datetime

    # === Mercado de apuestas ===
    market_home_prob: float
    market_draw_prob: float
    market_away_prob: float
    market_btts_yes_prob: float | None
    market_over_15_prob: float | None
    market_over_25_prob: float | None
    market_over_35_prob: float | None
    market_margin: float           # Overround promedio
    market_num_bookmakers: int
    market_disagreement: float     # Std de probabilidades entre casas
    odds_movement_home: float      # Delta apertura → cierre
    odds_movement_draw: float
    odds_movement_away: float
    correct_score_entropy: float   # Entropía del mercado de correct score

    # === Rendimiento deportivo ===
    elo_home: float | None
    elo_away: float | None
    elo_diff: float | None
    fifa_rank_home: int | None
    fifa_rank_away: int | None
    fifa_rank_diff: int | None
    form_pts_5_home: float
    form_pts_5_away: float
    form_pts_10_home: float
    form_pts_10_away: float
    goals_scored_5_home: float
    goals_scored_5_away: float
    goals_conceded_5_home: float
    goals_conceded_5_away: float
    xg_home: float | None
    xg_away: float | None
    xga_home: float | None
    xga_away: float | None
    xg_diff: float | None
    xg_ratio_home: float | None    # xG_home / (xG_home + xGA_away)
    home_performance_factor: float # Rendimiento como local vs global
    away_performance_factor: float
    h2h_wins_home: int
    h2h_draws: int
    h2h_wins_away: int
    h2h_avg_goals: float | None
    possession_diff: float | None
    shots_diff: float | None
    shots_on_target_diff: float | None

    # === Contexto ===
    match_importance: float        # 0-1 basado en clasificación y ronda
    rest_days_home: int | None
    rest_days_away: int | None
    is_knockout: bool
    round_weight: float            # Peso de la ronda para bonos
    must_win_home: float           # 0-1 necesidad de ganar
    must_win_away: float
    already_qualified_home: bool
    already_qualified_away: bool

    def to_array(self) -> np.ndarray:
        """Convierte a array numpy para ML models."""
    def to_dict(self) -> dict[str, float]:
        """Convierte a diccionario para modelos que lo requieran."""

class FeaturePipeline:
    def __init__(self, session: Session): ...

    def build_features(self, match_id: int) -> MatchFeatureVector:
        """Pipeline completo: extrae de BD, calcula derivadas, retorna vector."""

    def build_features_batch(self, match_ids: list[int]) -> list[MatchFeatureVector]: ...

    def _get_market_features(self, match_id: int) -> dict: ...
    def _get_performance_features(self, match_id: int) -> dict: ...
    def _get_context_features(self, match_id: int) -> dict: ...
    def _calculate_match_importance(self, match: Match) -> float: ...
    def _calculate_form(self, team_id: int, last_n: int) -> float:
        """Puntos obtenidos en los últimos N partidos."""
```

### Cálculo de match_importance

```python
def _calculate_match_importance(self, match: Match) -> float:
    """0-1 donde 1 es partido decisivo"""
    if match.round in ("round_of_16", "quarter", "semi", "final"):
        return 1.0  # Eliminatorias: máxima importancia
    # Fase de grupos: basado en si hay algo en juego
    # (depende de resultados de otros partidos del grupo)
    return 0.5  # Default para fase de grupos
```

### Cálculo de correct_score_entropy

```python
def _calculate_entropy(probs: np.ndarray) -> float:
    """Entropía de Shannon; alta = mercado incierto sobre el marcador."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))
```

## Acceptance Criteria

### FBref Scraper

- [ ] `scrape_world_cup_matches(2022)` retorna 64 partidos con xG
- [ ] xG y xGA son floats positivos
- [ ] Rate limiting respeta 3 req/min
- [ ] Cache persiste HTML en disco

### Feature Pipeline

- [ ] `build_features(match_id)` retorna `MatchFeatureVector` con todos los campos
- [ ] Features de mercado calculadas a partir de `closing_odds` (las últimas)
- [ ] Features de rendimiento usan solo datos ANTERIORES al partido (no leakage)
- [ ] `to_array()` produce un numpy array 1D con orden fijo
- [ ] `to_dict()` produce un diccionario con nombres de features como keys
- [ ] Test: features de partido conocido tienen valores esperados
- [ ] Test: validación temporal: features no incluyen datos del futuro
- [ ] Test: `MatchFeatureVector` se puede serializar/deserializar

## Files to Create

```
src/etl/fbref.py
src/features/__init__.py
src/features/pipeline.py
src/features/market.py
src/features/performance.py
src/features/context.py
tests/test_fbref.py
tests/test_feature_pipeline.py
tests/fixtures/fbref/          (HTML de ejemplo)
```

## Git Workflow

```bash
git checkout -b feature/spec-008-data-pipeline

# Commit 1: FBref scraper
git add src/etl/fbref.py tests/test_fbref.py tests/fixtures/fbref/
git commit -m "feat(SPEC-008): add FBref scraper for xG and advanced stats"

# Commit 2: Feature pipeline
git add src/features/ tests/test_feature_pipeline.py
git commit -m "feat(SPEC-008): add feature engineering pipeline with market, performance, context features"

pytest tests/test_fbref.py tests/test_feature_pipeline.py -v
ruff check src/etl/fbref.py src/features/

git checkout main
git merge feature/spec-008-data-pipeline
```

## Notes

- FBref URLs para Mundial 2022: `https://fbref.com/en/comps/1/{match_id}/...`
- Las tablas de xG están en el div `#shots_{team_id}` en formato HTML complejo
- Para el Mundial 2026, crear las URLs cuando estén disponibles
- El feature `correct_score_entropy` requiere los datos de SPEC-006 (correct score odds)
- NO usar datos de partidos futuros en el cálculo de features (temporal leakage)
- La función `_calculate_form` debe filtrar por fecha < fecha_del_partido
