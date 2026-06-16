# SPEC-005: Strategy Selector

## Status: COMPLETED

## Objective

Sistema de estrategia adaptativa que selecciona el modo de juego (conservador, balanceado,
diferenciación, alto riesgo) según la posición actual del jugador en la tabla de la polla.

## Dependencies

- **SPEC-001** (config: `StrategyConfig`)
- **SPEC-004** (`ExpectedScoreCalculator`)

## Technical Design

### `src/optimization/strategy.py`

```python
class StrategyMode(Enum):
    MINIMIZE_RISK = "minimize_risk"
    BALANCED = "balanced"
    DIFFERENTIATION = "differentiation"
    HIGH_RISK = "high_risk"

@dataclass
class StrategyRecommendation:
    prediction: ExpectedScoreResult
    strategy_mode: StrategyMode
    reasoning: str
    risk_score: float
    upside_potential: float
    risk_of_ruin: float

class StrategySelector:
    def __init__(self, config: StrategyConfig | None = None): ...
    def determine_mode(self, current_position: int, total_participants: int) -> StrategyMode: ...
    def get_recommendation(
        self, prediction: MatchPrediction,
        current_position: int, total_participants: int,
        ownership_matrix: np.ndarray | None = None,
    ) -> StrategyRecommendation: ...
```

### Matriz de decisión

| Posición | Modo | Selección |
|---|---|---|
| 1 | MINIMIZE_RISK | Marcadores con P(resultado) + P(exacto) > 30%, max EP |
| 2-5 | BALANCED | Top 5 marcadores por EP, max EP |
| 6-10 | DIFFERENTIATION | Marcadores con ownership < 20% y P(exacto) > 5%, max contrarian_value |
| 11-15 | HIGH_RISK | Marcadores con ownership < 10% y P(exacto) > 2%, max contrarian_value × EP |

### Heurísticas de selección por modo

```python
def _select_minimize_risk(predictions):  # Filter: prob_result+prob_exact > 0.3
def _select_balanced(predictions):       # Top 5 by EP
def _select_differentiation(predictions):# Filter: ownership < 0.2, prob_exact > 0.05
def _select_high_risk(predictions):      # Filter: ownership < 0.1, prob_exact > 0.02
```

## Acceptance Criteria

- [x] Posición 1 → `StrategyMode.MINIMIZE_RISK`
- [x] Posición 3 → `StrategyMode.BALANCED`
- [x] Posición 8 → `StrategyMode.DIFFERENTIATION`
- [x] Posición 13 → `StrategyMode.HIGH_RISK`
- [x] `get_recommendation` retorna `StrategyRecommendation` válida
- [x] Liderando tiene menor `risk_score` que trailing
- [x] La recomendación incluye `reasoning` no vacío
- [x] Test coverage: 86% (`src/optimization/strategy.py`)

## Files

| File | Purpose |
|---|---|
| `src/optimization/strategy.py` | Selector de estrategia adaptativa |
| `tests/test_strategy.py` | 6 tests unitarios |

## Git Commits

```bash
git add src/optimization/strategy.py tests/test_strategy.py
git commit -m "feat(SPEC-005): implement adaptive strategy selector based on position"
```

## Verification

```bash
pytest tests/test_strategy.py -v
```
