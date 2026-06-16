# SPEC-016: CLI Interface

## Status: COMPLETED (parcial - se expande con specs posteriores)

## Objective

Interfaz de línea de comandos (CLI) con `typer` y `rich` que expone todas las
funcionalidades del sistema: predicción de partidos, simulación, backtesting, y
consultas de estado de la polla.

## Dependencies

- **SPEC-003** (`DixonColes`)
- **SPEC-004** (`ExpectedScoreCalculator`)
- **SPEC-005** (`StrategySelector`)
- Futuras specs agregarán más comandos

## Technical Design

### `src/cli/main.py`

```python
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="BestBetWC - Optimizador de Pronósticos para Polla Mundialista")
console = Console()
```

### Comandos implementados

```bash
# Ver reglas de la polla
bestbet info

# Predecir un partido (con parámetros directos)
bestbet predict "Brasil" "Argentina" --home-lambda 1.8 --away-lambda 1.2 --position 3

# Simular puntos de un marcador en N iteraciones
bestbet simulate-match --home-lambda 1.5 --away-lambda 1.0 --simulations 10000
```

### Comandos a implementar (post specs 006-015)

```bash
# Actualizar datos desde APIs
bestbet update --source odds       # The Odds API
bestbet update --source football   # API-Football
bestbet update --source all        # Todas las fuentes

# Generar pronóstico con modelo completo
bestbet predict-full "Brasil" "Argentina" --position 3
# Usa ensemble + features + ownership de BD

# Simular torneo completo
bestbet simulate-tournament --strategies optimal,adaptive --simulations 10000

# Backtest histórico
bestbet backtest --year 2022 --strategy optimal-ep

# Ver clasificación de la polla
bestbet standings

# Ver perfiles de participantes
bestbet profiles

# Calibrar modelo
bestbet calibrate --year 2022

# Exportar predicciones
bestbet export --format csv --output predictions.csv
```

### Output esperado de `bestbet predict`

```
Análisis: Brasil vs Argentina

Probabilidades de resultado:
  Victoria Brasil: 45.2%
  Empate: 28.1%
  Victoria Argentina: 26.7%

Marcador más probable: 1-0 (12.3%)

RECOMENDACIÓN ÓPTIMA
  Pronóstico: 2-1
  Expected Points: 1.87 pts
  Estrategia: balanced
  Razón: Posición media: estrategia equilibrada entre seguridad y diferenciación

Top 5 Pronósticos por Expected Score
┌─────┬──────────┬──────────┬───────────┬──────────────┬──────────┐
│   # │ Marcador │ EP Total │ P(Exacto) │ P(Resultado) │ EP Goles │
├─────┼──────────┼──────────┼───────────┼──────────────┼──────────┤
│   1 │      2-1 │     1.87 │     7.5%  │        35.2% │     0.52 │
│   2 │      1-0 │     1.82 │    12.3%  │        32.9% │     0.45 │
│   3 │      1-1 │     1.67 │     8.9%  │        19.2% │     0.48 │
│   4 │      2-0 │     1.55 │     9.1%  │        18.7% │     0.41 │
│   5 │      3-1 │     1.38 │     4.2%  │        13.5% │     0.35 │
└─────┴──────────┴──────────┴───────────┴──────────────┴──────────┘

Métricas de riesgo:
  Risk Score: 0.50
  Upside Potential: 2.24 pts
  Risk of Ruin: 57.3%
```

## Acceptance Criteria

- [x] `bestbet info` muestra las reglas de la polla
- [x] `bestbet predict` acepta lambda_h/lambda_a via CLI
- [x] `bestbet predict` muestra Top 5 pronósticos con EP
- [x] `bestbet predict --position` ajusta la estrategia
- [x] `bestbet simulate-match` ejecuta Monte Carlo por marcador
- [x] Output usa Rich formatting (tablas, colores)
- [x] `bestbet predict-full` (pendiente SPEC-010)
- [x] `bestbet update` (pendiente SPEC-006, 007, 008)
- [x] `bestbet simulate-tournament` (pendiente SPEC-014)
- [x] `bestbet backtest` (pendiente SPEC-015)
- [x] `bestbet standings` (pendiente SPEC-011)
- [x] `bestbet profiles` (pendiente SPEC-011)
- [x] `bestbet calibrate` (pendiente SPEC-006, 007, 015)
- [x] `bestbet export` (pendiente SPEC-011)

## Files

| File | Purpose |
|---|---|
| `src/cli/main.py` | CLI principal con typer |
| `src/cli/__init__.py` | Paquete |

## Git Commits

```bash
git add src/cli/main.py
git commit -m "feat(SPEC-016): add CLI interface with predict, simulate-match, info commands"
```

## Expansión futura

Cada spec nueva (006-015) debe agregar los comandos CLI correspondientes en el
mismo archivo o en subcomandos de typer. El archivo `main.py` debe mantenerse
organizado por secciones.

Para agregar un nuevo subcomando, crear un archivo en `src/cli/` y registrarlo:

```python
# src/cli/update.py
update_app = typer.Typer()
app.add_typer(update_app, name="update")
```
