# SPEC-015: Backtesting Framework

## Status: PLANNED

## Objective

Framework de backtesting con validación temporal estricta que evalúa el rendimiento
histórico del sistema usando Mundiales pasados (2014, 2018, 2022), simulando la
polla con las reglas actuales y comparando múltiples estrategias.

## Dependencies

- **SPEC-003** (`DixonColes`, `MatchPrediction`)
- **SPEC-004** (`ExpectedScoreCalculator`)
- **SPEC-005** (`StrategySelector`)
- **SPEC-014** (`TournamentSimulator`)

## Context

El backtesting es la única forma de validar que el sistema realmente mejora la
puntuación esperada. No se puede usar cross-validation aleatorio porque los partidos
tienen dependencia temporal (la forma de un equipo depende de partidos anteriores).

El backtesting debe:
1. Cargar datos reales de Mundiales 2014, 2018, 2022
2. Para cada partido, entrenar el modelo SOLO con datos anteriores a ese partido
3. Generar predicciones y calcular EP
4. Acumular puntuación total
5. Comparar contra estrategias baseline

## Technical Design

### `src/validation/backtesting.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class BacktestConfig:
    validation_years: list[int] = field(default_factory=lambda: [2014, 2018, 2022])
    max_goals: int = 7
    temporal_split: bool = True  # SIEMPRE True
    min_train_matches: int = 20  # Mínimo de partidos para entrenar

@dataclass
class BacktestMatch:
    match_id: str
    date: datetime
    home_team: str
    away_team: str
    round: str
    home_goals: int
    away_goals: int
    features: dict[str, float] | None  # Features disponibles en ese momento

@dataclass
class BacktestResult:
    strategy_name: str
    tournament: str          # "World Cup 2022"
    total_points: float
    points_per_match: list[float]
    exact_scores: int
    correct_results: int
    prediction_history: list[dict]  # Predicción + resultado por partido
    log_loss: float
    brier_score: float
    calibration_error: float
    ranked_probability_score: float
    closings_line_value: float | None  # Requiere cuotas históricas

@dataclass
class BacktestReport:
    tournament: str
    strategies: dict[str, BacktestResult]
    baseline_result: BacktestResult
    relative_performance: dict[str, float]  # % mejora vs baseline
    summary: str

class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None): ...

    def load_tournament_data(
        self, year: int
    ) -> list[BacktestMatch]:
        """Carga datos históricos de un Mundial. Ordenados cronológicamente."""

    def run_backtest(
        self,
        year: int,
        strategy_fn: callable,
        strategy_name: str,
    ) -> BacktestResult:
        """
        Backtest de una estrategia en un torneo.

        Algoritmo:
        1. Cargar partidos ordenados por fecha
        2. Para cada partido i:
           a. train_matches = partidos[0:i]  (SOLO anteriores)
           b. Entrenar modelo con train_matches
           c. Generar predicción para partido i
           d. Comparar con resultado real
           e. Calcular puntos y métricas
        3. Acumular y retornar BacktestResult
        """

    def run_backtest_expanding_window(
        self,
        year: int,
        strategy_fn: callable,
        strategy_name: str,
        window_size: int = 50,
    ) -> BacktestResult:
        """
        Variante con ventana deslizante: solo usa los últimos N partidos
        para entrenar (más relevante para equipos que cambian con el tiempo).
        """

    def run_all_tournaments(
        self,
        strategy_fn: callable,
        strategy_name: str,
    ) -> dict[int, BacktestResult]:
        """Ejecuta backtest en todos los torneos configurados."""

    def compare_strategies(
        self,
        strategies: dict[str, callable],
        year: int = 2022,
    ) -> BacktestReport:
        """
        Compara múltiples estrategias en el mismo torneo.

        Estrategias baseline incluidas:
        - always_favorite: siempre predecir victoria del favorito 1-0
        - market_consensus: siempre seguir cuotas (si disponibles)
        - optimal_ep: usar EP calculator sin ownership
        - optimal_ep_contrarian: usar EP calculator con ownership estimado
        """

    def calculate_metrics(
        self, predictions: list[np.ndarray],
        actuals: list[tuple[int, int]],
    ) -> dict[str, float]:
        """
        Calcula métricas de evaluación:
        - Log Loss
        - Brier Score
        - Calibration Error (ECE)
        - Ranked Probability Score
        """

    def calculate_calibration_error(
        self, probs: np.ndarray, actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error (ECE)."""
```

### Estrategias baseline para comparación

```python
def always_favorite_strategy(
    match_pred: MatchPrediction,
    team_home: str, team_away: str,
) -> tuple[int, int]:
    """Siempre predice victoria del favorito (según mercado) con 1-0."""
    if match_pred.home_win_prob > match_pred.away_win_prob:
        return (1, 0)
    elif match_pred.away_win_prob > match_pred.home_win_prob:
        return (0, 1)
    else:
        return (1, 1)

def optimal_ep_strategy(
    match_pred: MatchPrediction,
    ep_calculator: ExpectedScoreCalculator,
) -> tuple[int, int]:
    """Maximiza EP sin considerar ownership."""
    optimal = ep_calculator.find_optimal_prediction(match_pred)
    return (optimal.home_goals, optimal.away_goals)

def optimal_ep_contrarian_strategy(
    match_pred: MatchPrediction,
    ep_calculator: ExpectedScoreCalculator,
    ownership_matrix: np.ndarray,
) -> tuple[int, int]:
    """Maximiza EP considerando ownership."""
    optimal = ep_calculator.find_optimal_prediction(match_pred, ownership_matrix)
    return (optimal.home_goals, optimal.away_goals)

def adaptive_strategy(
    match_pred: MatchPrediction,
    strategy_selector: StrategySelector,
    current_position: int,
    ownership_matrix: np.ndarray,
) -> tuple[int, int]:
    """Estrategia adaptativa según posición en la tabla."""
    rec = strategy_selector.get_recommendation(
        match_pred, current_position, 15, ownership_matrix
    )
    return (rec.prediction.home_goals, rec.prediction.away_goals)
```

### Cálculo de Expected Calibration Error

```python
def calculate_ece(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    """
    Agrupa predicciones en bins por confianza.
    Para cada bin: |accuracy - confidence|
    Ponderado por tamaño del bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = actuals[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf) / len(probs)
    return ece
```

### Log Loss para marcadores

```python
def calculate_score_matrix_log_loss(
    score_matrices: list[np.ndarray],
    actual_scores: list[tuple[int, int]],
) -> float:
    """
    Log Loss multi-clase para matrices de marcador.
    -log(P(i,j)) para el marcador real.
    """
    ll = 0.0
    for matrix, (h, a) in zip(score_matrices, actual_scores):
        p = max(matrix[h, a], 1e-15)
        ll += np.log(p)
    return -ll / len(score_matrices)
```

## Acceptance Criteria

- [ ] `load_tournament_data(2022)` carga 64 partidos en orden cronológico
- [ ] `run_backtest(2022, ...)` usa SOLO datos anteriores a cada partido
- [ ] Temporal leakage test: predicción del partido 1 no usa datos del partido 64
- [ ] `compare_strategies()` muestra que optimal_ep > always_favorite
- [ ] `calculate_calibration_error()` retorna ECE en [0, 1]
- [ ] Log Loss para siempre-predecir-favorito > Log Loss del modelo entrenado
- [ ] Brier Score < 0.25 para el modelo (mejor que random)
- [ ] `BacktestReport` incluye summary legible
- [ ] Test: expanding window usa máximo window_size partidos
- [ ] Test: backtest sin suficientes datos iniciales usa priors no informativos
- [ ] Test: métricas calculadas son consistentes con cálculo manual

## Files to Create

```
src/validation/__init__.py
src/validation/backtesting.py
data/raw/world_cup_2022.csv      (datos de ejemplo)
data/raw/world_cup_2018.csv
data/raw/world_cup_2014.csv
tests/test_backtesting.py
```

## Git Workflow

```bash
git checkout -b feature/spec-015-backtesting

git add src/validation/ data/raw/world_cup_20*.csv
git commit -m "feat(SPEC-015): add backtesting framework with temporal validation"

git add tests/test_backtesting.py
git commit -m "test(SPEC-015): add backtesting tests with historical World Cup data"

pytest tests/test_backtesting.py -v
ruff check src/validation/

git checkout main
git merge feature/spec-015-backtesting
```

## Notes

- Datos de Mundiales pasados disponibles en: https://www.kaggle.com/datasets/
- Formato CSV mínimo: date,home_team,away_team,home_goals,away_goals,round
- Para métricas de calibración, se necesita al menos 30 partidos de test
- El CLV (Closing Line Value) requiere cuotas históricas: difícil de conseguir
- Priorizar Log Loss y Expected Score como métricas principales
- La validación temporal es NO NEGOCIABLE. El README debe advertirlo.
- Datasets de ejemplo: incluir CSV pequeño (5 partidos) para que los tests corran offline
