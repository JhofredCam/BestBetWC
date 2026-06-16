# SPEC-009: Gradient Boosting Model

## Status: COMPLETED

## Objective

Implementar un modelo XGBoost que prediga la distribución de marcadores usando
el feature vector completo (mercado + rendimiento + contexto). El modelo debe
estimar P(home_goals=i, away_goals=j) para cada marcador posible.

## Dependencies

- **SPEC-008** (`MatchFeatureVector`, `FeaturePipeline`)

## Context

Mientras Dixon-Coles (SPEC-003) captura la estructura paramétrica básica del fútbol,
un modelo de gradient boosting puede capturar interacciones no lineales complejas
entre features heterogéneas (cuotas, xG, forma, contexto).

El approach recomendado es:
1. No predecir directamente el marcador (problema de clasificación multi-clase con 64+ clases)
2. En su lugar, predecir las distribuciones marginales P(home_goals=k) y P(away_goals=k) por separado
3. Combinar marginales asumiendo independencia (Poisson bivariado) para obtener la conjunta

Alternativa: usar XGBoost como corrector del output de Dixon-Coles (residual fitting).

## Technical Design

### `src/models/gradient_boost.py`

```python
import xgboost as xgb
from dataclasses import dataclass
import numpy as np

@dataclass
class GBModelConfig:
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int = 20

class GradientBoostModel:
    """Wrapper alrededor de XGBoost para predecir distribuciones de goles."""

    def __init__(self, config: GBModelConfig | None = None): ...

    def fit(
        self,
        X: np.ndarray,          # Features matrix (n_matches × n_features)
        y_home: np.ndarray,     # Home goals (n_matches,)
        y_away: np.ndarray,     # Away goals (n_matches,)
        sample_weights: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict:                  # Training history
        """
        Entrena DOS modelos XGBoost independientes:
        - model_home: predice P(home_goals = k | features) para k ∈ [0, max_goals]
        - model_away: predice P(away_goals = k | features) para k ∈ [0, max_goals]
        """

    def predict_score_distribution(
        self, X: np.ndarray, max_goals: int = 7
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna distribuciones marginales:
        - home_probs: shape (n_samples, max_goals+1)
        - away_probs: shape (n_samples, max_goals+1)

        La distribución conjunta se obtiene como:
        P(i,j) = P(home=i) × P(away=j)  (independencia)
        """

    def predict_match(
        self, features: MatchFeatureVector, max_goals: int = 7
    ) -> MatchPrediction:
        """Wrapper que retorna MatchPrediction compatible con DixonColes."""

    def get_feature_importance(self) -> dict[str, float]:
        """Feature importance del modelo (para interpretabilidad)."""

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### Estrategia de entrenamiento

```python
def temporal_train_test_split(
    matches: list[MatchFeatureVector],
    train_ratio: float = 0.8,
) -> tuple[list, list]:
    """
    Split TEMPORAL, no aleatorio.
    Los primeros 80% de partidos cronológicamente son train,
    el último 20% es test.
    """
    sorted_matches = sorted(matches, key=lambda m: m.timestamp)
    split_idx = int(len(sorted_matches) * train_ratio)
    return sorted_matches[:split_idx], sorted_matches[split_idx:]


def calibrate_model(
    model: GradientBoostModel,
    X_cal: np.ndarray,
    y_cal_home: np.ndarray,
    y_cal_away: np.ndarray,
) -> GradientBoostModel:
    """
    Post-calibración usando Platt scaling o isotonic regression
    para mejorar la calibración probabilística.
    """
```

### Approach alternativo: residual fitting

```python
class ResidualGBModel:
    """
    XGBoost como corrector de Dixon-Coles.
    Entrena sobre los RESIDUOS: y_true - y_pred_dixon_coles.
    El output final es: pred_dixon_coles + pred_xgboost_residual.
    Menos propenso a overfitting que un modelo standalone.
    """

    def fit(
        self,
        X: np.ndarray,
        dixon_preds_home: np.ndarray,  # Probabilidades de Dixon-Coles
        dixon_preds_away: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
    ) -> dict: ...

    def predict_score_distribution(
        self, X: np.ndarray,
        dixon_preds_home: np.ndarray,
        dixon_preds_away: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...
```

## Acceptance Criteria

- [ ] `fit()` entrena dos modelos (home y away) y retorna training history
- [ ] `predict_score_distribution()` retorna probabilidades que suman ~1.0 por muestra
- [ ] `predict_match()` retorna `MatchPrediction` con score_matrix válida
- [ ] Temporal split: train/test por orden cronológico, no aleatorio
- [ ] Feature importance accesible post-entrenamiento
- [ ] Modelo entrenado en Mundial 2018 predice Mundial 2022 con Log Loss < 0.9
- [ ] Early stopping funciona correctamente (no overfitting)
- [ ] Test: modelo con 2 features triviales tiene feature importance coherente
- [ ] Test: `predict_match()` produce score_matrix que suma 1.0
- [ ] Test: save/load roundtrip preserva predicciones idénticas
- [ ] Type hints en todas las funciones públicas

## Files to Create

```
src/models/gradient_boost.py
tests/test_gradient_boost.py
```

## Git Workflow

```bash
git checkout -b feature/spec-009-gradient-boost

git add src/models/gradient_boost.py
git commit -m "feat(SPEC-009): add XGBoost model for score distribution prediction"

git add tests/test_gradient_boost.py
git commit -m "test(SPEC-009): add gradient boost model tests with temporal validation"

pytest tests/test_gradient_boost.py -v
ruff check src/models/gradient_boost.py

git checkout main
git merge feature/spec-009-gradient-boost
```

## Notes

- NO usar `train_test_split` aleatorio de sklearn. Implementar split temporal manual.
- El target para XGBoost es multi-class: `num_class = max_goals + 1 = 8`
- `objective="multi:softprob"` ya retorna probabilidades normalizadas
- Considerar `scale_pos_weight` si los datos están muy desbalanceados (muchos 0-0, 1-0)
- La calibración probabilística de XGBoost puede ser pobre sin post-procesamiento
- Guardar modelos en `data/models/gb_home.json` y `data/models/gb_away.json`
