# SPEC-001: Project Setup

## Status: COMPLETED

## Objective

Establecer la estructura base del proyecto Python: dependencias, configuración,
variables de entorno, y estándares de desarrollo (linting, type checking, testing).

## Dependencies

- Ninguna (spec raíz)

## Files Created

| File | Purpose |
|---|---|
| `pyproject.toml` | Dependencias, scripts, config de ruff/mypy/pytest |
| `.env.example` | Variables de entorno requeridas (API keys, DB URL) |
| `src/config.py` | `PollaRules`, `ModelConfig`, `StrategyConfig` dataclasses |
| `AGENTS.md` | Instrucciones para agentes de IA |
| `specs/README.md` | Guía de desarrollo y workflow git |

## Technical Design

### `src/config.py`

```python
@dataclass
class PollaRules:
    result_correct_pts: int = 2
    result_incorrect_pts: int = 0
    exact_score_pts: int = 5
    goals_home_correct_pts: int = 1
    goals_away_correct_pts: int = 1
    unique_prediction_bonus: int = 2
    round_bonus_16: int = 10
    round_bonus_8: int = 8
    round_bonus_4: int = 4
    round_bonus_semi: int = 2
    round_bonus_final: int = 5
    num_participants: int = 15
    max_goals: int = 7

@dataclass
class ModelConfig:
    dixon_coles_rho: float = -0.13
    ensemble_weights: dict[str, float]
    calibration_method: str = "isotonic"
    validation_folds: int = 5

@dataclass
class StrategyConfig:
    leading_threshold: int = 1
    middle_range: tuple[int, int] = (2, 5)
    behind_range: tuple[int, int] = (6, 10)
    trailing_range: tuple[int, int] = (11, 15)
    risk_aversion_leading: float = 0.8
    risk_aversion_trailing: float = 0.2
    contrarian_weight: float = 0.3

POLLA_RULES = PollaRules()
MODEL_CONFIG = ModelConfig()
STRATEGY_CONFIG = StrategyConfig()
```

### Environment Variables (`.env.example`)

```
THE_ODDS_API_KEY=your_key_here
API_FOOTBALL_KEY=your_key_here
DATABASE_URL=sqlite:///data/bestbetwc.db
```

## Acceptance Criteria

- [x] `pip install -e .` instala todas las dependencias
- [x] `pip install -e ".[dev]"` instala pytest, ruff, mypy, black
- [x] `ruff check src/` pasa sin errores
- [x] `mypy src/` pasa (cuando haya código tipado)
- [x] `pytest` descubre tests correctamente
- [x] Las reglas de la polla son configurables vía `PollaRules` dataclass

## Verification

```bash
pip install -e ".[dev]"
pytest -v
ruff check src/
```

## Git Commits for this spec

```bash
git add pyproject.toml .env.example src/config.py AGENTS.md specs/README.md
git commit -m "feat(SPEC-001): project setup with config, deps, and dev tooling"
```
