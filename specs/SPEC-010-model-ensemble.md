# SPEC-010: Model Ensemble

## Status: COMPLETED

## Objective

Combinar las predicciones de múltiples modelos (Dixon-Coles, mercado de apuestas,
XGBoost, Bayesiano) en un solo `MatchPrediction` mediante un ensamble ponderado
con pesos configurables.

## Dependencies

- **SPEC-003** (`DixonColes`, `MatchPrediction`)
- **SPEC-009** (`GradientBoostModel`)

## Context

Ningún modelo individual es óptimo para todas las situaciones. El ensamble permite:
- **Dixon-Coles**: estructura paramétrica, bueno con pocos datos
- **Mercado de apuestas**: sabiduría colectiva, excelente calibración
- **XGBoost**: captura interacciones no lineales entre features
- **Bayesiano (futuro)**: cuantifica incertidumbre, incorpora priors

El ensamble debe:
1. Aceptar pesos configurables por modelo
2. Combinar las `score_matrix` de cada modelo
3. Retornar un `MatchPrediction` unificado
4. Permitir que ciertos modelos no estén disponibles (peso=0 o ausentes)

## Technical Design

### `src/models/ensemble.py`

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EnsembleConfig:
    dixon_coles_weight: float = 0.35
    market_weight: float = 0.35
    gradient_boost_weight: float = 0.20
    bayesian_weight: float = 0.10
    calibration_samples: int = 1000  # Para estimar pesos óptimos

class ModelEnsemble:
    def __init__(self, config: EnsembleConfig | None = None): ...

    def add_dixon_coles(self, model: DixonColes) -> None: ...
    def add_gradient_boost(self, model: GradientBoostModel) -> None: ...
    def add_bayesian(self, model: Any) -> None: ...  # Futuro

    def set_market_prediction(
        self, score_matrix: np.ndarray, home_dist: np.ndarray, away_dist: np.ndarray
    ) -> None:
        """Setea la predicción derivada del mercado de apuestas."""

    def predict(
        self, team_home: str, team_away: str,
        features: MatchFeatureVector | None = None,
        market_score_matrix: np.ndarray | None = None,
    ) -> MatchPrediction:
        """
        Predicción ensembleada.

        Algoritmo:
        1. Obtener score_matrix de cada modelo disponible
        2. Ponderar por los pesos del EnsembleConfig
        3. Normalizar la matriz resultante
        4. Calcular distribuciones marginales
        5. Retornar MatchPrediction
        """

    def predict_from_individual_predictions(
        self, predictions: dict[str, MatchPrediction]
    ) -> MatchPrediction:
        """
        Alternativa: recibir predicciones ya calculadas y ensamblarlas.

        Args:
            predictions: {"dixon_coles": pred1, "market": pred2, ...}

        Returns:
            MatchPrediction con score_matrix ensembleada
        """

    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val_home: np.ndarray,
        y_val_away: np.ndarray,
        model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> EnsembleConfig:
        """
        Encuentra pesos óptimos minimizando Log Loss en conjunto de validación.
        Usa scipy.optimize.minimize con constraints (pesos >= 0, suman 1).
        """

    def get_weights_dict(self) -> dict[str, float]:
        """Retorna pesos actuales para logging/reporte."""

    def _validate_weights(self) -> None:
        """Verifica que los pesos sumen ~1.0"""
```

### Algoritmo de ensamble

```python
def _ensemble_score_matrices(
    matrices: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Weighted average of score matrices.
    Solo incluye modelos con weight > 0.
    """
    total_weight = 0.0
    result = np.zeros_like(list(matrices.values())[0])

    for name, matrix in matrices.items():
        w = weights.get(name, 0.0)
        if w > 0:
            result += w * matrix
            total_weight += w

    if total_weight > 0:
        result /= total_weight

    return result
```

### Uso típico

```python
ensemble = ModelEnsemble()

# Agregar modelos
ensemble.add_dixon_coles(dixon_coles_model)
ensemble.add_gradient_boost(gb_model)

# Mercado viene de SPEC-006
market_matrix = build_market_score_matrix(correct_score_odds, max_goals=7)

# Predecir
prediction = ensemble.predict(
    "Brasil", "Argentina",
    features=feature_vector,
    market_score_matrix=market_matrix,
)

# El MatchPrediction ya está listo para el EP calculator (SPEC-004)
optimal = ep_calculator.find_optimal_prediction(prediction)
```

## Acceptance Criteria

- [x] `predict()` retorna `MatchPrediction` con score_matrix que suma 1.0
- [x] Con un solo modelo, la predicción es idéntica al modelo individual
- [x] Con dos modelos y pesos (0.5, 0.5), la score_matrix es el promedio
- [x] `predict_from_individual_predictions()` acepta cualquier combinación de modelos
- [x] `optimize_weights()` minimiza Log Loss en validación
- [x] Los pesos optimizados están en [0, 1] y suman 1.0
- [x] `set_market_prediction()` acepta una score_matrix generada externamente
- [x] El ensamble funciona si algún modelo no está disponible (peso=0)
- [x] Test: ensamble de Dixon-Coles y mercado mejora Log Loss vs cada uno por separado
- [x] Test: `get_weights_dict()` retorna el estado actual
- [x] Type hints completos

## Files to Create

```
src/models/ensemble.py
tests/test_ensemble.py
```

## Git Workflow

```bash
git checkout -b feature/spec-010-model-ensemble

git add src/models/ensemble.py
git commit -m "feat(SPEC-010): implement weighted model ensemble for score distributions"

git add tests/test_ensemble.py
git commit -m "test(SPEC-010): add ensemble tests for prediction combination"

pytest tests/test_ensemble.py -v
ruff check src/models/ensemble.py

git checkout main
git merge feature/spec-010-model-ensemble
```

## Notes

- La matriz del mercado se construye a partir de las cuotas de correct score (SPEC-006)
- Los pesos por defecto favorecen ligeramente el mercado (mejor calibración)
- El optimizador de pesos usa scipy.optimize.minimize con método SLSQP
- La función objetivo es Log Loss (neg_log_likelihood)
- Si en el futuro se agrega un modelo bayesiano (PyMC), basta con agregar `add_bayesian()`
- El `MatchPrediction` de salida debe ser indistinguible del de `DixonColes.predict_match()`
