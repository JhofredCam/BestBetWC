# SPEC-004: Expected Score Calculator

## Status: COMPLETED

## Objective

Dada una distribución de probabilidad de marcadores (`MatchPrediction`) y las reglas de
la polla, calcular el **Expected Score (EP)** para cada marcador candidato y encontrar
el pronóstico que maximiza la puntuación esperada.

## Dependencies

- **SPEC-001** (config: `PollaRules`)
- **SPEC-003** (`MatchPrediction`, `DixonColes`)

## Context

Esta es LA capa crítica. El objetivo NO es predecir el marcador más probable, sino
el que maximiza la puntuación esperada según las reglas de la polla.

### Fórmula de Expected Score

Para un marcador candidato `(i, j)`:

```
EP(i,j) = P(exacto) × 5
        + P(resultado_sin_exacto) × 2
        + P(home_goals = i) × 1
        + P(away_goals = j) × 1
        + P(exacto) × (1 - ownership(i,j))^14 × 2
```

Donde:
- `P(exacto) = score_matrix[i, j]`
- `P(resultado_sin_exacto) = P(ganador/empate) - P(exacto)`
- `P(home_goals = i) = sum_j score_matrix[i, j]`
- `P(away_goals = j) = sum_i score_matrix[i, j]`
- El último término es el bono de predicción única (2 pts si eres el único)

## Technical Design

### `src/optimization/expected_score.py`

```python
@dataclass
class ExpectedScoreResult:
    home_goals: int
    away_goals: int
    ep_total: float          # Expected Score total
    ep_exact: float           # Componente de marcador exacto
    ep_result: float          # Componente de resultado sin exacto
    ep_goals_home: float      # Componente de goles local
    ep_goals_away: float      # Componente de goles visitante
    ep_unique: float          # Componente de predicción única
    prob_exact: float         # P(exacto)
    prob_result: float        # P(resultado) - P(exacto)
    prob_goals_home: float    # P(home_goals = i)
    prob_goals_away: float    # P(away_goals = j)
    ownership_estimate: float # Ownership estimado
    contrarian_value: float   # Valor contrarian = P(exacto) × (1 - ownership)

class ExpectedScoreCalculator:
    def __init__(self, rules: PollaRules | None = None): ...

    def calculate_ep(
        self, prediction: MatchPrediction,
        pred_home: int, pred_away: int,
        ownership_estimate: float = 0.0,
    ) -> ExpectedScoreResult: ...

    def find_optimal_prediction(
        self, prediction: MatchPrediction,
        ownership_matrix: np.ndarray | None = None,
    ) -> ExpectedScoreResult: ...

    def rank_all_predictions(
        self, prediction: MatchPrediction,
        ownership_matrix: np.ndarray | None = None,
    ) -> list[ExpectedScoreResult]: ...
```

### Notas importantes

- **El marcador exacto REEMPLAZA los puntos por resultado** (5 pts, no 5+2)
- **Los goles acertados son ADICIONALES** (se suman tanto a exacto como a resultado)
- **El bono de predicción única se aplica solo si hay exacto**
- **Ownership matrix** es opcional; si no se provee, ownership=0 para todos los marcadores

## Acceptance Criteria

- [x] `ep_total == ep_exact + ep_result + ep_goals_home + ep_goals_away + ep_unique`
- [x] Marcadores de alta probabilidad tienen mayor EP (sin ownership)
- [x] Ownership > 0 reduce `ep_unique` y puede cambiar el óptimo
- [x] `find_optimal_prediction` encuentra el marcador con máximo EP
- [x] `rank_all_predictions` devuelve 64 resultados ordenados por EP descendente
- [x] Las reglas son configurables via `PollaRules`
- [x] Test coverage: 100% (`src/optimization/expected_score.py`)

## Files

| File | Purpose |
|---|---|
| `src/optimization/expected_score.py` | Calculadora de EP |
| `tests/test_expected_score.py` | 10 tests unitarios |

## Git Commits

```bash
git add src/optimization/expected_score.py tests/test_expected_score.py
git commit -m "feat(SPEC-004): implement expected score calculator with unique prediction bonus"
```

## Verification

```bash
pytest tests/test_expected_score.py -v
```
