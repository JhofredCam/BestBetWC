# SPEC-013: Bracket Optimizer

## Status: COMPLETED

## Objective

Optimizador de predicciones para los bonos por ronda (bracket completo). Dado que
los bonos son binarios (todo o nada: 10 pts en 16avos si aciertas TODOS los
clasificados), el optimizador debe balancear la probabilidad de acertar el bracket
completo contra la diferenciación de otros participantes.

## Dependencies

- **SPEC-003** (`MatchPrediction` para cada partido)
- **SPEC-004** (`ExpectedScoreCalculator`)
- **SPEC-012** (ownership estimates de otros participantes)

## Context

Las reglas de la polla otorgan bonos por acertar TODOS los equipos clasificados a
cada ronda eliminatoria:

| Ronda | Bono | # Equipos a acertar |
|---|---|---|
| 16avos | 10 pts | 32 equipos |
| 8vos | 8 pts | 16 equipos |
| 4tos | 4 pts | 8 equipos |
| Semis | 2 pts | 4 equipos |
| Final | 5 pts | 2 equipos |

El bono de 16avos (10 pts) es el más valioso. Acertar el bracket completo de 32
equipos es extremadamente improbable, pero el objetivo no es acertarlo siempre,
sino maximizar el Expected Score total (incluyendo el Expected Bracket Bonus).

### Expected Bracket Bonus

```
EBB_ronda = P(acertar_todos_los_de_ronda) × bono_ronda

P(acertar_todos) = ∏ P(clasifica_equipo_i) para los N equipos de la ronda

Donde P(clasifica_equipo_i) se deriva de:
- Probabilidades de cada partido de grupos
- Probabilidades de cada partido de eliminatoria anterior
```

## Technical Design

### `src/optimization/bracket.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class BracketConfig:
    bonus_16: int = 10
    bonus_8: int = 8
    bonus_4: int = 4
    bonus_semi: int = 2
    bonus_final: int = 5
    num_groups: int = 12       # Mundial 2026: 12 grupos de 4
    teams_per_group: int = 4
    advancing_per_group: int = 2  # Clasifican 1° y 2° + 8 mejores 3°

@dataclass
class BracketPrediction:
    round_of_32: list[str]     # 32 equipos que pasan a 16avos
    round_of_16: list[str]     # 16 equipos que pasan a 8vos
    quarter_finalists: list[str]
    semi_finalists: list[str]
    finalists: list[str]
    champion: str | None
    prob_perfect: float        # Probabilidad de bracket perfecto
    expected_bonus: float      # Expected bracket bonus total

@dataclass
class GroupPrediction:
    group_name: str
    standings: list[tuple[str, float]]  # (team, prob_advancing)
    winner_prob: dict[str, float]
    runner_up_prob: dict[str, float]

class BracketOptimizer:
    def __init__(self, config: BracketConfig | None = None): ...

    def predict_group_standings(
        self,
        group_matches: list[Match],
        match_predictions: dict[str, MatchPrediction],
        num_simulations: int = 10000,
    ) -> GroupPrediction:
        """
        Simula fase de grupos para estimar probabilidades de clasificación.

        Para cada simulación:
        1. Samplear resultado de cada partido del grupo
        2. Calcular tabla de posiciones
        3. Determinar clasificados
        4. Acumular frecuencias
        """

    def predict_bracket(
        self,
        all_matches: list[Match],
        match_predictions: dict[str, MatchPrediction],
    ) -> BracketPrediction:
        """
        Simula el torneo completo para estimar:
        - Probabilidad de cada equipo de llegar a cada ronda
        - Expected bracket bonus total
        - Bracket más probable
        """

    def optimize_bracket_predictions(
        self,
        bracket_pred: BracketPrediction,
        ownership_estimates: dict[str, np.ndarray],
    ) -> BracketPrediction:
        """
        Ajusta las predicciones de bracket considerando ownership.

        Si el 80% de los participantes predice a Brasil en la final,
        y Brasil tiene 30% de probabilidad real:
        - Predicción estándar: Brasil (maximiza prob)
        - Predicción óptima: podría ser Argentina si hay suficiente
          probabilidad + diferenciación
        """

    def calculate_expected_bracket_bonus(
        self, bracket_pred: BracketPrediction
    ) -> float:
        """
        EBB = P(R32_perfecto) × 10
            + P(R16_perfecto) × 8
            + P(QF_perfecto) × 4
            + P(SF_perfecto) × 2
            + P(F_perfecto) × 5
        """

    def bracket_contrarian_value(
        self,
        my_bracket: BracketPrediction,
        opponent_brackets: list[BracketPrediction],
    ) -> float:
        """Cuánto me diferencio del bracket promedio de los oponentes."""

class KnockoutSimulator:
    """Simula rondas eliminatorias dado un bracket de clasificados."""

    def __init__(self, match_predictions: dict[str, MatchPrediction]): ...

    def simulate_round(
        self, teams: list[str], match_predictions: dict[str, MatchPrediction],
    ) -> list[str]:
        """Retorna ganadores de una ronda eliminatoria."""

    def simulate_full_bracket(
        self, round_of_32: list[str],
    ) -> BracketPrediction: ...
```

### Algoritmo de simulación de fase de grupos

```python
def simulate_group_stage(
    group_teams: list[str],
    match_predictions: dict[tuple[str, str], MatchPrediction],
    num_simulations: int = 10000,
) -> dict[str, float]:
    """
    Para cada simulación:
    1. Para cada partido del grupo, samplear (i,j) de score_matrix
    2. Construir tabla: equipo, pts, GF, GC, DG
    3. Ordenar por pts → DG → GF
    4. Top 2 avanzan (más posibles mejores 3ros)
    5. Acumular: cuántas veces cada equipo clasifica
    """
    advancing_count = {team: 0 for team in group_teams}

    for _ in range(num_simulations):
        table = {team: {"pts": 0, "gf": 0, "gc": 0} for team in group_teams}

        for (home, away), pred in match_predictions.items():
            h_goals, a_goals = sample_from_matrix(pred.score_matrix)
            table[home]["gf"] += h_goals
            table[home]["gc"] += a_goals
            table[away]["gf"] += a_goals
            table[away]["gc"] += h_goals

            if h_goals > a_goals:
                table[home]["pts"] += 3
            elif h_goals == a_goals:
                table[home]["pts"] += 1
                table[away]["pts"] += 1
            else:
                table[away]["pts"] += 3

        sorted_table = sorted(
            table.items(),
            key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["gc"], x[1]["gf"]),
            reverse=True,
        )

        for team, _ in sorted_table[:2]:
            advancing_count[team] += 1

    return {team: count / num_simulations for team, count in advancing_count.items()}
```

## Acceptance Criteria

- [ ] `predict_group_standings()` retorna probabilidades de clasificación para cada equipo
- [ ] Las probabilidades de clasificación suman ~2.0 por grupo (2 clasificados)
- [ ] `predict_bracket()` retorna `BracketPrediction` con expected_bonus
- [ ] `calculate_expected_bracket_bonus()` desglosa el EBB por ronda
- [ ] Simulación de 10,000 iteraciones termina en < 5 segundos
- [ ] `optimize_bracket_predictions()` ajusta según ownership
- [ ] `bracket_contrarian_value()` mide diferenciación del bracket
- [ ] Test: grupo con un favorito claro, su prob de clasificar > 90%
- [ ] Test: EBB total nunca excede la suma de todos los bonos
- [ ] Test: simulación determinística con seed fijo
- [ ] Test: bracket para Mundial 2022 reproduce resultados reales en ~60% de equipos

## Files to Create

```
src/optimization/bracket.py
tests/test_bracket.py
```

## Git Workflow

```bash
git checkout -b feature/spec-013-bracket-optimizer

git add src/optimization/bracket.py
git commit -m "feat(SPEC-013): implement bracket optimizer with group stage simulation"

git add tests/test_bracket.py
git commit -m "test(SPEC-013): add bracket optimizer tests with group simulation"

pytest tests/test_bracket.py -v
ruff check src/optimization/bracket.py

git checkout main
git merge feature/spec-013-bracket-optimizer
```

## Notes

- Mundial 2026 tiene formato expandido: 12 grupos de 4, 32 clasifican a 16avos
  (1° y 2° de cada grupo = 24, más 8 mejores 3ros)
- La regla de "8 mejores 3ros" complica la simulación: no basta top 2
- Para la simulación, usar `np.random.choice` con la score_matrix como distribución
- El bracket eliminatorio sigue un cuadro predefinido. Modelar el cuadro.
- Los bonos de ronda son ADICIONALES a los puntos por partido
- Priorizar optimización de 16avos (10 pts) y final (5 pts) por ser los bonos más altos
