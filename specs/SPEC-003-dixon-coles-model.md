# SPEC-003: Dixon-Coles Model

## Status: COMPLETED

## Objective

Implementar el modelo Dixon-Coles para estimar la distribución de probabilidad
conjunta de marcadores (home_goals, away_goals) para cualquier partido dados
los parámetros de fuerza de ataque/defensa de cada equipo.

## Dependencies

- **SPEC-001** (config: `max_goals`)

## Technical Design

### `src/models/dixon_coles.py`

El modelo produce una matriz de probabilidad `score_matrix[i, j]` donde `i` son
goles del local y `j` goles del visitante, para `i, j in [0, max_goals]`.

Corrige el sesgo del Poisson independiente en marcadores bajos (0-0, 1-0, 0-1, 1-1)
mediante un factor τ (tau):

```
P(home=i, away=j) = Poisson(i|λ_home) × Poisson(j|μ_away) × τ(i, j, λ_home, μ_away, ρ)
```

Donde:
- `λ_home = exp(home_advantage + attack_home - defense_away)`
- `μ_away = exp(attack_away - defense_home)`
- `ρ ≈ -0.13` es el parámetro de correlación de baja inflación

### Clases implementadas

```python
@dataclass
class MatchPrediction:
    home_goals_dist: np.ndarray       # Distribución marginal de goles local
    away_goals_dist: np.ndarray       # Distribución marginal de goles visitante
    score_matrix: np.ndarray          # Matriz conjunta (max_goals+1 × max_goals+1)
    home_win_prob: float              # P(local gana)
    draw_prob: float                  # P(empate)
    away_win_prob: float              # P(visitante gana)
    expected_home_goals: float        # λ esperado
    expected_away_goals: float        # μ esperado
    most_likely_score: tuple[int, int]
    most_likely_score_prob: float

class DixonColes:
    def __init__(self, max_goals: int = 7): ...
    def predict_match(self, team_home: str, team_away: str) -> MatchPrediction: ...
    def fit(self, matches: list[dict]) -> None: ...
    def predict_from_params(self, lambda_h: float, mu_a: float, rho: float | None = None) -> MatchPrediction: ...
```

### Uso

```python
# Con parámetros directos (sin fit previo)
model = DixonColes(max_goals=7)
pred = model.predict_from_params(lambda_h=1.5, mu_a=1.0)
print(pred.score_matrix[1, 0])  # P(1-0)

# Con fit de datos históricos
model.fit([
    {"home_team": "Brasil", "away_team": "Argentina", "home_goals": 2, "away_goals": 1},
    ...
])
pred = model.predict_match("Brasil", "Argentina")
```

## Acceptance Criteria

- [x] `score_matrix.sum() == 1.0` (±1e-6)
- [x] `home_win_prob + draw_prob + away_win_prob == 1.0` (±1e-6)
- [x] Con `lambda_h > mu_a`, `home_win_prob > away_win_prob`
- [x] Con `lambda_h == mu_a`, `draw_prob > 0.2`
- [x] El factor ρ modifica las probabilidades de marcadores bajos
- [x] `fit()` aprende parámetros de ataque/defensa correctamente
- [x] Distribuciones marginales suman 1.0
- [x] `most_likely_score` es coherente con los parámetros
- [x] Test coverage: 100% (`src/models/dixon_coles.py`)

## Files

| File | Purpose |
|---|---|
| `src/models/dixon_coles.py` | Implementación completa |
| `tests/test_dixon_coles.py` | 7 tests unitarios |

## Git Commits

```bash
git add src/models/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat(SPEC-003): implement Dixon-Coles model for score distribution prediction"
```

## Verification

```bash
pytest tests/test_dixon_coles.py -v
```
