# SPEC-014: Monte Carlo Tournament Simulator

## Status: COMPLETED

## Objective

Simulador Monte Carlo que genera torneos completos para evaluar estrategias,
estimar distribuciones de posiciones finales, calcular Win Probability, y
comparar el rendimiento esperado de diferentes estrategias de pronóstico.

## Dependencies

- **SPEC-003** (`DixonColes`, `MatchPrediction`)
- **SPEC-004** (`ExpectedScoreCalculator`)
- **SPEC-010** (`ModelEnsemble`) - opcional, fuente de predicciones
- **SPEC-012** (ownership estimates) - opcional

## Context

El simulador es la herramienta principal para responder preguntas estratégicas:
- "¿Cuál es mi probabilidad de ganar la polla si sigo esta estrategia?"
- "¿Cuánto arriesgo si elijo un marcador contrarian?"
- "¿Qué posición esperada tendré después de la fecha 3?"
- "¿Vale la pena sacrificar puntos ahora para diferenciarme después?"

## Technical Design

### `src/simulation/tournament.py`

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimulationConfig:
    num_simulations: int = 10000
    max_goals: int = 7
    seed: int | None = 42
    track_progress: bool = True  # Mostrar barra de progreso

@dataclass
class TournamentResult:
    """
    Resultado de UNA simulación de torneo completo.
    Contiene TODOS los resultados de TODOS los partidos.
    """
    match_results: list[tuple[int, int]]           # (home_goals, away_goals) por partido
    group_standings: dict[str, list[str]]            # Grupo → [equipos ordenados]
    knockout_results: list[tuple[str, str, int, int]] # (home, away, goles_h, goles_a)

@dataclass
class StrategyResult:
    """
    Resultado de evaluar UNA estrategia en UN torneo simulado.
    """
    strategy_name: str
    total_points: float
    match_points: list[float]  # Puntos por partido
    bracket_bonus: float
    position: int              # Posición final (1-15)
    exact_scores: int
    correct_results: int

@dataclass
class SimulationReport:
    """
    Reporte agregado de N simulaciones para una estrategia.
    """
    strategy_name: str
    mean_points: float
    std_points: float
    median_points: float
    min_points: float
    max_points: float
    win_probability: float          # Probabilidad de terminar 1°
    top3_probability: float         # Probabilidad de terminar en top 3
    last_probability: float         # Probabilidad de terminar último
    expected_rank: float            # Posición esperada
    rank_distribution: dict[int, float]  # Posición → probabilidad
    points_percentiles: dict[float, float]  # Percentil → puntos
    risk_of_ruin: float             # Probabilidad de bajar de posición

class TournamentSimulator:
    def __init__(
        self,
        config: SimulationConfig | None = None,
    ): ...

    def set_match_predictions(
        self, predictions: dict[str, MatchPrediction]
    ) -> None:
        """Configura las predicciones de cada partido (match_id → MatchPrediction)."""

    def simulate_tournament(self) -> TournamentResult:
        """
        Simula UN torneo completo.

        1. Fase de grupos:
           Para cada partido, samplear (h,a) de score_matrix
           Calcular tabla de cada grupo
           Determinar clasificados (top 2 por grupo + 8 mejores 3ros)

        2. Eliminatorias:
           Para cada ronda, samplear resultados de partidos del bracket
           Avanzar ganadores

        3. Retornar TournamentResult con TODO
        """

    def simulate_n_tournaments(
        self, n: int | None = None,
    ) -> list[TournamentResult]:
        """Simula N torneos independientes."""

    def evaluate_strategy(
        self,
        strategy_name: str,
        strategy_fn: callable,  # (match_id, prediction) → (home_goals, away_goals)
        tournament_results: list[TournamentResult],
        opponent_strategies: list[callable] | None = None,
    ) -> SimulationReport:
        """
        Evalúa una estrategia en todos los torneos simulados.

        Para cada torneo:
        1. Para cada partido, la estrategia elige un marcador
        2. Se compara con el resultado simulado del partido
        3. Se calculan puntos según reglas de la polla (SPEC-004)
        4. Se compara con estrategias de oponentes
        5. Se determina posición final

        Retorna SimulationReport con estadísticas agregadas.
        """

    def compare_strategies(
        self,
        strategies: dict[str, callable],
        n_simulations: int = 10000,
    ) -> dict[str, SimulationReport]:
        """Compara múltiples estrategias en las mismas simulaciones."""

    def calculate_win_probability(
        self,
        my_points: list[float],
        opponent_points: list[list[float]],
    ) -> float:
        """Probabilidad de que mis puntos > puntos de todos los oponentes."""
```

### `src/simulation/participants.py`

```python
@dataclass
class SimulatedParticipant:
    """Participante simulado para evaluación de estrategias."""
    name: str
    profile: PlayerProfile
    strategy_mode: StrategyMode

class ParticipantSimulator:
    """Genera predicciones de participantes simulados basados en perfiles."""

    def __init__(self, profiler: PlayerProfiler | None = None): ...

    def simulate_predictions(
        self,
        participant: SimulatedParticipant,
        match_predictions: dict[str, MatchPrediction],
    ) -> dict[str, tuple[int, int]]:
        """Genera predicciones del participante para todos los partidos."""

    def simulate_all(
        self,
        participants: list[SimulatedParticipant],
        matches: list[Match],
        match_predictions: dict[str, MatchPrediction],
    ) -> dict[int, dict[str, tuple[int, int]]]: ...

    @staticmethod
    def conservative_strategy(
        match_pred: MatchPrediction,
    ) -> tuple[int, int]: ...

    @staticmethod
    def aggressive_strategy(
        match_pred: MatchPrediction,
    ) -> tuple[int, int]: ...

    @staticmethod
    def market_follower_strategy(
        match_pred: MatchPrediction,
        market_probs: np.ndarray,
    ) -> tuple[int, int]: ...

    @staticmethod
    def random_strategy(
        match_pred: MatchPrediction,
    ) -> tuple[int, int]: ...
```

### `src/simulation/monte_carlo.py`

```python
class MonteCarloEngine:
    """
    Motor principal de Monte Carlo. Orquesta todas las simulaciones.
    """

    def __init__(
        self,
        tournament_sim: TournamentSimulator,
        participant_sim: ParticipantSimulator,
        ep_calculator: ExpectedScoreCalculator,
        config: SimulationConfig,
    ): ...

    def run_full_simulation(
        self,
        matches: list[Match],
        match_predictions: dict[str, MatchPrediction],
        my_strategies: dict[str, callable],
        opponent_profiles: list[PlayerProfile],
        n_simulations: int = 10000,
    ) -> dict[str, SimulationReport]:
        """
        Simulación completa:

        1. Simular N torneos
        2. Para cada estrategia mía:
           a. Generar predicciones para cada partido
           b. Simular predicciones de oponentes
           c. Calcular puntos para todos
           d. Determinar posiciones
        3. Agregar resultados en SimulationReport por estrategia
        """

    def run_what_if(
        self,
        scenario: str,
        matches: list[Match],
        match_predictions: dict[str, MatchPrediction],
        n_simulations: int = 5000,
    ) -> SimulationReport:
        """
        Escenarios what-if:
        - "¿Qué pasa si elijo todos los favoritos?"
        - "¿Qué pasa si voy contrarian en el 50% de partidos?"
        - "¿Qué pasa si acierto 3 exactos seguidos?"
        """
```

## Acceptance Criteria

- [ ] `simulate_tournament()` retorna `TournamentResult` con todos los partidos
- [ ] `simulate_n_tournaments(1000)` termina en < 30 segundos
- [ ] `evaluate_strategy()` retorna `SimulationReport` con todas las métricas
- [ ] `win_probability` está en [0, 1] y es consistente
- [ ] `expected_rank` es un float entre 1 y 15
- [ ] `rank_distribution` suma 1.0
- [ ] `compare_strategies()` permite comparar A vs B
- [ ] Semilla fija → resultados reproducibles
- [ ] Test: estrategia perfecta (elige resultado real) gana siempre vs estrategia aleatoria
- [ ] Test: en 1000 simulaciones, ~15% de posiciones 1° para estrategia media
- [ ] Test: `SimulationReport` contiene todas las métricas listadas
- [ ] Type hints completos

## Files to Create

```
src/simulation/__init__.py
src/simulation/tournament.py
src/simulation/participants.py
src/simulation/monte_carlo.py
tests/test_tournament_simulation.py
tests/test_participant_simulation.py
tests/test_monte_carlo.py
```

## Git Workflow

```bash
git checkout -b feature/spec-014-monte-carlo

git add src/simulation/tournament.py
git commit -m "feat(SPEC-014): add tournament simulator with group and knockout phases"

git add src/simulation/participants.py tests/test_participant_simulation.py
git commit -m "feat(SPEC-014): add participant simulator with strategy archetypes"

git add src/simulation/monte_carlo.py tests/test_monte_carlo.py
git commit -m "feat(SPEC-014): add Monte Carlo engine for full tournament evaluation"

pytest tests/test_tournament_simulation.py tests/test_participant_simulation.py tests/test_monte_carlo.py -v
ruff check src/simulation/

git checkout main
git merge feature/spec-014-monte-carlo
```

## Notes

- Usar `numpy.random.default_rng(seed)` para reproducibilidad
- Las simulaciones son embarazosamente paralelizables (cada torneo es independiente)
- Para MVP, procesamiento secuencial. Para producción, usar `concurrent.futures`
- El cuello de botella es samplear de score_matrix para 64 partidos × N simulaciones
- Cachear los samples pre-generados si N es muy grande (>100k)
- La fase de grupos del Mundial 2026 tiene 12 grupos de 4 equipos (72 partidos de grupo)
- Total partidos: 72 (grupos) + 32 (16avos) + 16 + 8 + 4 + 2 + 1 = 135? 
  No, verificar: 48 equipos, 12 grupos de 4 → 6 partidos por grupo = 72 partidos de grupo.
  32 clasificados a 16avos → 16 → 8 → 4 → 2 → 1 final. Total = 72 + 16 + 8 + 4 + 2 + 1 = 103 partidos.
