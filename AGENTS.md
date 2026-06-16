# AGENTS.md - Instrucciones para Agentes de IA

## Proyecto
BestBetWC - Sistema Inteligente para Optimizar Pronósticos en Polla Mundialista 2026

## Comandos

```bash
# Instalar dependencias
pip install -e .

# Ejecutar tests
pytest

# Ejecutar con coverage
pytest --cov=src

# Linting
ruff check src/

# Type checking
mypy src/

# Formateo
ruff format src/
```

## Convenciones

- Python 3.11+
- Type hints en todas las funciones públicas
- Docstrings en módulos y clases principales
- Tests unitarios para toda lógica de negocio
- Validación temporal (NO cross-validation aleatorio)
- Las reglas de la polla son CONFIGURABLES (no hardcodear)

## Estructura

- `src/` - Código fuente
- `tests/` - Tests unitarios
- `notebooks/` - Análisis exploratorio
- `data/` - Datos (raw, processed, predictions)

## Métrica Principal

**Expected Score (EP)** - NO accuracy. El objetivo es maximizar puntos esperados en la polla, no precisión predictiva.

## Reglas de la Polla (CONFIGURABLES)

Ver `SPEC.md` sección 1.2 para detalles completos.

## APIs

- The Odds API: cuotas de apuestas
- API-Football: datos deportivos
- FBref/Understat: xG y stats avanzadas

## Importante

- Nunca hardcodear API keys (usar .env)
- Cache de respuestas de API (minimizar requests)
- Validación temporal estricta en modelos
- Las predicciones deben incluir: probabilidad, EP, ownership estimate, contrarian value
