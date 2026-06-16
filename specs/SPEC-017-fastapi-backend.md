# SPEC-017: FastAPI Backend

## Status: PLANNED

## Objective

Backend REST API con FastAPI que exponga todas las funcionalidades del sistema:
predicción de partidos, recomendaciones de estrategia, simulación Monte Carlo,
backtesting, y consultas de la polla. Sirve como API para la UI web (SPEC-018)
y cualquier otra interfaz futura.

## Dependencies

- **SPEC-002** (database layer)
- **SPEC-003** (`DixonColes`, `MatchPrediction`)
- **SPEC-004** (`ExpectedScoreCalculator`)
- **SPEC-005** (`StrategySelector`)
- **SPEC-006** (`OddsAPIClient`) - opcional, si hay API key
- **SPEC-014** (`MonteCarloEngine`) - opcional
- **SPEC-015** (`BacktestEngine`) - opcional

## Context

El CLI actual (`bestbet predict`) requiere abrir una terminal y pasar parámetros
manualmente. Un backend FastAPI permite:
- Interfaz web (SPEC-018)
- Automatización via HTTP
- Múltiples usuarios consultando predicciones
- Integración con notificaciones (alertas pre-partido)
- Endpoints async para ETL (actualización de datos desde APIs)

## Technical Design

### `src/api/`

```
src/api/
├── __init__.py
├── app.py              # FastAPI app factory
├── dependencies.py     # Dependency injection (DB session, models, etc.)
├── routers/
│   ├── __init__.py
│   ├── predictions.py  # /api/predictions
│   ├── matches.py      # /api/matches
│   ├── strategies.py   # /api/strategies
│   ├── simulation.py   # /api/simulation
│   ├── backtesting.py  # /api/backtesting
│   ├── standings.py    # /api/standings
│   └── profiles.py     # /api/profiles
└── schemas.py          # Pydantic models
```

### `src/api/app.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routers import (
    predictions, matches, strategies, simulation,
    backtesting, standings, profiles,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: inicializar DB, cargar modelos
    yield
    # Shutdown: cerrar conexiones

def create_app() -> FastAPI:
    app = FastAPI(
        title="BestBetWC API",
        description="Sistema Inteligente para Optimizar Pronósticos en Polla Mundialista 2026",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
    app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
    app.include_router(strategies.router, prefix="/api/strategies", tags=["Strategies"])
    app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
    app.include_router(backtesting.router, prefix="/api/backtesting", tags=["Backtesting"])
    app.include_router(standings.router, prefix="/api/standings", tags=["Standings"])
    app.include_router(profiles.router, prefix="/api/profiles", tags=["Profiles"])

    return app

app = create_app()
```

### `src/api/dependencies.py`

```python
from fastapi import Depends
from sqlalchemy.orm import Session

from src.database.connection import get_session
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector

def get_db() -> Session:
    """Dependency: SQLAlchemy session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()

def get_dixon_coles() -> DixonColes:
    """Dependency: modelo DixonColes cargado/pre-entrenado."""
    # Cargar de caché o entrenar con datos de BD
    ...

def get_ep_calculator() -> ExpectedScoreCalculator:
    """Dependency: calculadora de EP con reglas actuales."""
    return ExpectedScoreCalculator()

def get_strategy_selector() -> StrategySelector:
    """Dependency: selector de estrategia."""
    return StrategySelector()
```

### `src/api/schemas.py`

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# === Request schemas ===

class PredictRequest(BaseModel):
    home_team: str = Field(..., description="Nombre del equipo local")
    away_team: str = Field(..., description="Nombre del equipo visitante")
    home_lambda: float = Field(default=1.5, ge=0.1, le=10.0,
                                description="Goles esperados del local")
    away_lambda: float = Field(default=1.0, ge=0.1, le=10.0,
                                description="Goles esperados del visitante")
    current_position: int = Field(default=1, ge=1, le=15,
                                   description="Posición actual en la polla")

class PredictFromOddsRequest(BaseModel):
    """Predicción usando cuotas del mercado (requiere SPEC-006)."""
    match_id: int

class SimulateRequest(BaseModel):
    home_lambda: float = 1.5
    away_lambda: float = 1.0
    simulations: int = Field(default=10000, ge=100, le=100000)

class BacktestRequest(BaseModel):
    year: int = Field(default=2022, ge=2010, le=2026)
    strategy: str = "optimal_ep"

class UpdateDataRequest(BaseModel):
    source: str = Field(default="all", pattern="^(odds|football|fbref|all)$")

# === Response schemas ===

class ScoreProbability(BaseModel):
    home_goals: int
    away_goals: int
    probability: float
    ep_total: float
    ep_exact: float
    ep_result: float
    ep_goals: float
    ep_unique: float

class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    most_likely_score: str          # "2-1"
    most_likely_score_prob: float
    expected_home_goals: float
    expected_away_goals: float
    top_predictions: list[ScoreProbability]
    recommendation: StrategyRecommendationResponse
    score_matrix: list[list[float]]  # Matriz completa serializada

class StrategyRecommendationResponse(BaseModel):
    prediction: str                  # "2-1"
    ep_total: float
    strategy_mode: str
    reasoning: str
    risk_score: float
    upside_potential: float
    risk_of_ruin: float

class StrategyModesResponse(BaseModel):
    """Mapeo de posiciones a modos de estrategia."""
    modes: dict[str, str]  # {"1": "minimize_risk", "2-5": "balanced", ...}

class SimulationResult(BaseModel):
    score: str
    ep_mean: float
    ep_min: float
    ep_max: float
    result_hit_rate: float
    exact_hit_rate: float
    std_dev: float

class SimulateResponse(BaseModel):
    home_lambda: float
    away_lambda: float
    num_simulations: int
    results: list[SimulationResult]

class BacktestResultItem(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    predicted_score: str
    actual_score: str
    points: float
    correct_result: bool
    exact_score: bool

class BacktestResponse(BaseModel):
    year: int
    strategy: str
    total_points: float
    exact_scores: int
    correct_results: int
    log_loss: float
    brier_score: float
    match_details: list[BacktestResultItem]

class MatchSummary(BaseModel):
    id: int
    home_team: str
    away_team: str
    datetime: datetime
    round: str
    status: str
    home_score: Optional[int]
    away_score: Optional[int]

class StandingEntry(BaseModel):
    position: int
    participant_name: str
    total_points: int
    exact_scores: int
    correct_results: int

class ProfileResponse(BaseModel):
    participant_name: str
    archetype: str
    conservative_score: float
    aggressive_score: float
    market_follower_score: float
    intuition_score: float
    favorite_bias: float
    home_bias: float
    result_accuracy: float
    exact_accuracy: float
    avg_points_per_match: float

class ErrorResponse(BaseModel):
    detail: str
    code: str
```

### `src/api/routers/predictions.py`

```python
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def predict_match(
    request: PredictRequest,
    ep_calculator: ExpectedScoreCalculator = Depends(get_ep_calculator),
    strategy_selector: StrategySelector = Depends(get_strategy_selector),
):
    """Genera predicción y recomendación para un partido."""

@router.get("/match/{match_id}", response_model=PredictionResponse)
async def predict_match_by_id(
    match_id: int,
    position: int = 1,
    db: Session = Depends(get_db),
    ep_calculator: ExpectedScoreCalculator = Depends(get_ep_calculator),
):
    """Predicción para un partido de la BD usando features reales."""

@router.get("/upcoming", response_model=list[PredictionResponse])
async def predict_upcoming_matches(
    position: int = 1,
    db: Session = Depends(get_db),
):
    """Predicciones para todos los partidos pendientes."""
```

### `src/api/routers/strategies.py`

```python
router = APIRouter()

@router.get("/modes", response_model=StrategyModesResponse)
async def get_strategy_modes():
    """Retorna el mapeo de posiciones → modos de estrategia."""

@router.get("/optimal/{position}")
async def get_optimal_strategy(
    position: int,
    home_lambda: float = 1.5,
    away_lambda: float = 1.0,
):
    """Recomendación de estrategia para una posición específica."""
```

### `src/api/routers/simulation.py`

```python
router = APIRouter()

@router.post("/match", response_model=SimulateResponse)
async def simulate_match(request: SimulateRequest):
    """Simula N iteraciones de un partido y compara estrategias."""

@router.post("/tournament")
async def simulate_tournament(
    strategy: str = "optimal_ep",
    simulations: int = 10000,
):
    """Simula un torneo completo (requiere SPEC-014)."""
```

### `src/api/routers/backtesting.py`

```python
router = APIRouter()

@router.post("/", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Ejecuta backtesting en un Mundial histórico."""

@router.get("/strategies")
async def list_backtest_strategies():
    """Lista estrategias disponibles para backtesting."""
```

### `src/api/routers/standings.py`

```python
router = APIRouter()

@router.get("/", response_model=list[StandingEntry])
async def get_standings(
    round: str | None = None,
    db: Session = Depends(get_db),
):
    """Clasificación actual de la polla."""

@router.get("/participant/{participant_id}")
async def get_participant_performance(
    participant_id: int,
    db: Session = Depends(get_db),
):
    """Detalle de rendimiento de un participante."""
```

### `src/api/routers/profiles.py`

```python
router = APIRouter()

@router.get("/{participant_id}", response_model=ProfileResponse)
async def get_profile(
    participant_id: int,
    db: Session = Depends(get_db),
):
    """Perfil de un participante (teoría de juegos)."""

@router.get("/", response_model=list[ProfileResponse])
async def get_all_profiles(
    db: Session = Depends(get_db),
):
    """Perfiles de todos los participantes."""
```

### Endpoint de actualización de datos

```python
# src/api/routers/data.py
router = APIRouter()

@router.post("/update")
async def update_data(
    request: UpdateDataRequest,
    db: Session = Depends(get_db),
):
    """Actualiza datos desde APIs externas."""
    ...

@router.get("/status")
async def data_status(db: Session = Depends(get_db)):
    """Estado de los datos: últimos partidos, cuotas disponibles, etc."""
    ...
```

## Acceptance Criteria

### General
- [ ] `uvicorn src.api.app:app` levanta el servidor en `localhost:8000`
- [ ] Swagger UI disponible en `/docs` con todos los endpoints documentados
- [ ] CORS configurado para desarrollo local
- [ ] Health check en `/health` retorna 200
- [ ] Todos los endpoints tienen type hints y response models

### Predictions
- [ ] `POST /api/predictions` retorna `PredictionResponse` completo
- [ ] `GET /api/predictions/match/{id}` usa datos de BD si están disponibles
- [ ] `GET /api/predictions/upcoming` retorna lista de próximos partidos
- [ ] Error 404 si el match_id no existe
- [ ] Error 422 si los parámetros son inválidos

### Strategies
- [ ] `GET /api/strategies/modes` retorna mapeo completo de posiciones
- [ ] `GET /api/strategies/optimal/{position}` varía según posición

### Simulation
- [ ] `POST /api/simulation/match` acepta 100-100000 simulaciones
- [ ] Timeout configurable para simulaciones grandes
- [ ] Respuesta incluye EP_mean, EP_std, hit rates

### Backtesting
- [ ] `POST /api/backtesting` acepta años 2010-2026
- [ ] `GET /api/backtesting/strategies` lista estrategias disponibles

### Standings & Profiles
- [ ] `GET /api/standings` retorna clasificación ordenada
- [ ] `GET /api/profiles` retorna todos los perfiles
- [ ] Endpoints vacíos (sin datos) retornan lista vacía (no 500)

### Tests
- [ ] Test: cliente HTTP (`httpx` o `TestClient`) contra la app
- [ ] Test: cada endpoint retorna schema correcto
- [ ] Test: validación de request (Pydantic) funciona
- [ ] Test: manejo de errores (404, 422, 500)
- [ ] Test de integración con DixonColes y EP calculator reales

## Files to Create

```
src/api/__init__.py
src/api/app.py
src/api/dependencies.py
src/api/schemas.py
src/api/routers/__init__.py
src/api/routers/predictions.py
src/api/routers/matches.py
src/api/routers/strategies.py
src/api/routers/simulation.py
src/api/routers/backtesting.py
src/api/routers/standings.py
src/api/routers/profiles.py
src/api/routers/data.py
tests/test_api.py
tests/test_api_predictions.py
tests/test_api_strategies.py
```

## Git Workflow

```bash
git checkout -b feature/spec-017-fastapi-backend

# Commit 1: app factory + schemas
git add src/api/__init__.py src/api/app.py src/api/schemas.py src/api/dependencies.py
git commit -m "feat(SPEC-017): add FastAPI app factory, schemas, and dependency injection"

# Commit 2: prediction + strategy routers
git add src/api/routers/__init__.py src/api/routers/predictions.py src/api/routers/strategies.py
git commit -m "feat(SPEC-017): add prediction and strategy API routers"

# Commit 3: simulation + backtesting routers
git add src/api/routers/simulation.py src/api/routers/backtesting.py
git commit -m "feat(SPEC-017): add simulation and backtesting API routers"

# Commit 4: standings + profiles + data routers
git add src/api/routers/standings.py src/api/routers/profiles.py src/api/routers/data.py src/api/routers/matches.py
git commit -m "feat(SPEC-017): add standings, profiles, matches, and data API routers"

# Commit 5: tests
git add tests/test_api.py tests/test_api_predictions.py tests/test_api_strategies.py
git commit -m "test(SPEC-017): add FastAPI integration tests"

pytest tests/test_api*.py -v
ruff check src/api/

git checkout main
git merge feature/spec-017-fastapi-backend
```

## Notes

- Agregar `fastapi`, `uvicorn`, `pydantic` a `pyproject.toml`
- Las dependencias usan FastAPI's `Depends` para inyectar DB session y modelos
- Los modelos de ML (DixonColes, XGBoost) se cargan una vez en startup (lifespan)
- Para operaciones largas (simulación 100k), considerar BackgroundTasks o polling
- Swagger en `/docs` permite probar todos los endpoints interactivamente
- La app debe ser stateless: el estado viene de la BD y los modelos son read-only
- Rate limiting para endpoints pesados (simulation, backtesting)
- NO exponer API keys en responses
- Response models usan `response_model_by_alias` si es necesario para camelCase
- El health check `/health` debe verificar conectividad con BD
