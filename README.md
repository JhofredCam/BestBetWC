# BestBetWC - Sistema Inteligente para Optimizar Pronósticos en Polla Mundialista 2026

Sistema de optimización de pronósticos que maximiza la **Puntuación Esperada (Expected Score)** en lugar de la precisión predictiva.

## Características

- **Modelo Dixon-Coles** para predicción de distribuciones de marcadores
- **Calculadora de Expected Score** basada en reglas configurables de la polla
- **Estrategia adaptativa** según posición en la tabla
- **Teoría de juegos** para diferenciación estratégica
- **Simulación Monte Carlo** para evaluación de estrategias

## Instalación

```bash
pip install -e .
```

## Uso

```bash
# Ver información de las reglas
bestbet info

# Predecir un partido
bestbet predict "Brasil" "Argentina" --home-lambda 1.8 --away-lambda 1.2 --position 3

# Simular un partido
bestbet simulate-match --home-lambda 1.5 --away-lambda 1.0 --simulations 10000
```

## Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest

# Linting
ruff check src/

# Type checking
mypy src/

# Formateo
ruff format src/
```

## Documentación

Ver [SPEC.md](SPEC.md) para especificación completa del sistema.

## Licencia

MIT
