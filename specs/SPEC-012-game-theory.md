# SPEC-012: Game Theory Layer (Player Profiling + Ownership Estimation)

## Status: PLANNED

## Objective

Construir la capa de teoría de juegos que: (1) perfila a cada uno de los 15 participantes
de la polla según sus patrones de predicción, (2) estima la probabilidad de que cada
marcador sea elegido por otros participantes (ownership matrix), y (3) calcula métricas
de diferenciación (contrarian value, leverage score).

## Dependencies

- **SPEC-002** (models: `ParticipantProfile`, `ParticipantPrediction`)
- **SPEC-004** (`ExpectedScoreCalculator`, `ExpectedScoreResult`)
- **SPEC-011** (`PollaScraper` - datos de participantes)

## Context

En una polla de 15 participantes, el ganador no es quien mejor predice, sino quien
toma mejores decisiones RELATIVAS a los demás. Esta capa es potencialmente MÁS IMPORTANTE
que mejorar el modelo de predicción de 70% a 75% de accuracy.

La capa de teoría de juegos opera en dos fases:
1. **Profiling**: para cada participante, estimar su estilo de juego
2. **Ownership**: para cada partido futuro, estimar qué marcadores elegirán los demás

## Technical Design

### `src/game_theory/profiling.py`

```python
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class PlayerArchetype(Enum):
    CONSERVATIVE = "conservative"       # Favoritos, marcadores bajos
    AGGRESSIVE = "aggressive"           # Upsets, goleadas
    MARKET_FOLLOWER = "market_follower" # Sigue cuotas
    INTUITION = "intuition"             # Patrones no predecibles
    HOMER = "homer"                     # Sesgo regional (favorece equipo propio)

@dataclass
class PlayerProfile:
    participant_id: int
    name: str

    # Scores de arquetipo (suman 1.0)
    conservative_score: float
    aggressive_score: float
    market_follower_score: float
    intuition_score: float

    # Sesgos específicos
    favorite_bias: float              # 0-1: sobrevalora favoritos
    home_bias: float                  # 0-1: favorece al local
    draw_aversion: float              # 0-1: evita predecir empates
    exact_score_freq: float           # Frecuencia de acertar exactos
    avg_goals_predicted: float        # Media de goles por partido en sus pronósticos
    popular_team_score: dict[str, float]  # Equipos que favorece
    underdog_team_score: dict[str, float] # Equipos que subestima

    # Historial
    total_predictions: int
    result_accuracy: float            # % acierto de resultado
    exact_accuracy: float             # % acierto de exacto
    avg_points_per_match: float
    predicted_matches: list[dict]     # Últimos N partidos predichos

    def to_dict(self) -> dict: ...

class PlayerProfiler:
    def __init__(self, session: Session): ...

    def profile_all(self) -> dict[int, PlayerProfile]:
        """Perfila a todos los participantes."""

    def profile_participant(
        self, participant_id: int
    ) -> PlayerProfile: ...

    def calculate_archetype_scores(
        self, predictions: list[ParticipantPrediction],
        match_contexts: list[dict],
    ) -> tuple[float, float, float, float]:
        """
        Calcula los scores de arquetipo basado en patrones:

        conservative_score:
            - Alta frecuencia de predecir favoritos
            - Marcadores bajos (1-0, 2-0, 2-1)
            - Pocos empates

        aggressive_score:
            - Alta frecuencia de upsets
            - Marcadores altos (3-2, 4-1)

        market_follower_score:
            - Alta correlación con cuotas de mercado
            - Baja desviación del consenso

        intuition_score:
            - Baja correlación con cualquier patrón
            - Alta varianza en predicciones
        """

    def calculate_biases(
        self, predictions: list[ParticipantPrediction],
    ) -> dict:
        """Estima sesgos: home_bias, draw_aversion, favorite_bias."""

    def detect_team_preferences(
        self, predictions: list[ParticipantPrediction],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Detecta equipos que el participante sobrevalora/infravalora."""

    def update_profiles(self) -> None:
        """Actualiza todos los perfiles en la BD."""
```

### `src/game_theory/ownership.py`

```python
@dataclass
class OwnershipEstimate:
    score_matrix: np.ndarray         # Ownership por marcador
    most_popular_score: tuple[int, int]
    ownership_of_most_popular: float
    entropy: float                   # Diversidad de predicciones
    unique_opportunities: list[tuple[int, int]]  # Marcadores con ownership < 10%

class OwnershipEstimator:
    def __init__(self, profiler: PlayerProfiler): ...

    def estimate(
        self, match: Match, team_home: str, team_away: str,
        market_probs: np.ndarray | None = None,
    ) -> OwnershipEstimate:
        """
        Estima ownership matrix para un partido.

        Algoritmo:
        1. Para cada participante, obtener su PlayerProfile
        2. Generar predicción simulada del participante basada en su perfil
        3. Contar frecuencia de cada marcador entre los 14 participantes
        4. Normalizar a probabilidades

        Si market_probs está disponible, el market_follower_score
        ajusta la predicción hacia el consenso del mercado.
        """

    def _simulate_participant_prediction(
        self, profile: PlayerProfile,
        home_team: str, away_team: str,
        market_probs: np.ndarray | None,
    ) -> tuple[int, int]:
        """
        Simula qué marcador elegiría este participante basado en su perfil.

        Usa una mezcla de:
        - Probabilidades del mercado (peso: market_follower_score)
        - Favoritismo al local (peso: home_bias)
        - Tendencia a marcadores bajos/altos (peso: conservative/aggressive)
        - Sesgos de equipos específicos
        """

    def estimate_batch(
        self, matches: list[Match],
    ) -> dict[int, OwnershipEstimate]: ...

    def get_contrarian_value(
        self, estimate: OwnershipEstimate,
        model_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Contrarian value = model_probs[i,j] × (1 - ownership[i,j])

        Alto cuando el marcador es probable (según el modelo) pero
        poco popular (según ownership).
        """
```

### `src/game_theory/opponent_model.py`

```python
class OpponentModel:
    """
    Modelo predictivo de lo que cada oponente va a elegir.
    Extiende PlayerProfiler con predicción forward-looking.
    """

    def __init__(self, profiler: PlayerProfiler): ...

    def predict_opponent_choice(
        self, participant_id: int,
        match: Match, team_home: str, team_away: str,
    ) -> tuple[int, int]:
        """Predice el marcador que elegirá un oponente específico."""

    def predict_all_opponents(
        self, match: Match,
        team_home: str, team_away: str,
        exclude_participant: int | None = None,
    ) -> list[tuple[int, int]]:
        """Predice los marcadores de TODOS los oponentes."""

    def build_ownership_matrix(
        self, match: Match,
        team_home: str, team_away: str,
        max_goals: int = 7,
    ) -> np.ndarray:
        """Construye ownership matrix completa para un partido."""
```

## Acceptance Criteria

### Player Profiling

- [ ] `profile_all()` retorna 15 perfiles (uno por participante)
- [ ] `conservative_score + aggressive_score + market_follower_score + intuition_score ≈ 1.0`
- [ ] `calculate_archetype_scores()` identifica correctamente patrones conocidos
- [ ] `detect_team_preferences()` identifica equipos sobre/infravalorados
- [ ] `update_profiles()` persiste en `participant_profiles`
- [ ] Test: perfil conservador tiene conservative_score > 0.5
- [ ] Test: perfil de datos sintéticos con patrón conocido

### Ownership Estimation

- [ ] `estimate()` retorna ownership matrix que suma ~1.0 (como distribución)
- [ ] `most_popular_score` es coherente con los perfiles de los participantes
- [ ] `unique_opportunities` lista marcadores con ownership < 10%
- [ ] `get_contrarian_value()` produce valores en [0, 1]
- [ ] Test: con 14 perfiles idénticos, ownership del favorito > 60%

## Files to Create

```
src/game_theory/profiling.py
src/game_theory/ownership.py
src/game_theory/opponent_model.py
tests/test_profiling.py
tests/test_ownership.py
tests/test_opponent_model.py
```

## Git Workflow

```bash
git checkout -b feature/spec-012-game-theory

# Commit 1: profiling
git add src/game_theory/profiling.py tests/test_profiling.py
git commit -m "feat(SPEC-012): implement player profiling with archetype detection"

# Commit 2: ownership
git add src/game_theory/ownership.py tests/test_ownership.py
git commit -m "feat(SPEC-012): implement ownership estimation and contrarian value"

# Commit 3: opponent model
git add src/game_theory/opponent_model.py tests/test_opponent_model.py
git commit -m "feat(SPEC-012): implement opponent choice prediction model"

pytest tests/test_profiling.py tests/test_ownership.py tests/test_opponent_model.py -v
ruff check src/game_theory/

git checkout main
git merge feature/spec-012-game-theory
```

## Notes

- El profiling usa datos históricos (SPEC-011). Si no hay historial, todos los perfiles
  son no informativos (todos los scores = 0.25)
- La simulación de oponentes NO es determinística; usar seeds para reproducibilidad
- El `market_follower_score` requiere acceso a cuotas (SPEC-006) para calcular correlación
- Los sesgos de equipos (`popular_team_score`) se actualizan incrementalmente
- Considerar usar Dirichlet distribution para modelar la incertidumbre en los perfiles
