# BestBetWC — Optimizador de Pronósticos para Polla Mundialista 2026

Sistema inteligente que maximiza la **Puntuación Esperada (Expected Score)** en una polla con 15 participantes, usando teoría de juegos, modelos estadísticos y simulación Monte Carlo.

## Cómo funciona

El sistema **NO intenta predecir fútbol con máxima precisión**. En su lugar, para cada partido:

1. **Estima la distribución de marcadores** (`P(goles_local, goles_visitante)`) con un ensemble de modelos (Dixon-Coles + XGBoost + cuotas de mercado)
2. **Calcula el Expected Score** de cada marcador candidato aplicando las reglas exactas de la polla (5 pts exacto, 2 pts resultado, 1 pt goles, +2 pts predicción única)
3. **Aplica teoría de juegos**: estima qué van a predecir los otros 14 participantes (`ownership`) y recomienda el marcador con mayor valor esperado, que puede ser distinto del más probable
4. **Adapta la estrategia** a tu posición actual: conservador si vas líder, agresivo si vas atrás

### Reglas de la polla

| Concepto | Puntos |
|---|---|
| Acertar marcador exacto | 5 pts |
| Acertar resultado (ganador/empate, sin exacto) | 2 pts |
| Acertar goles del local | 1 pt |
| Acertar goles del visitante | 1 pt |
| Predicción única (nadie más la eligió) | +2 pts |
| Bono 16avos (todos los clasificados) | 10 pts |
| Bono 8vos | 8 pts |
| Bono 4tos | 4 pts |
| Bono Semis | 2 pts |
| Bono Final | 5 pts |

### Arquitectura

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  The Odds API │   │ API-Football │   │  FBref/HTML  │
│  (cuotas)     │   │ (resultados) │   │  (xG/stats)  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────┐
│              Feature Pipeline (ETL)               │
│  52 features: mercado + rendimiento + contexto    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│            Model Ensemble                         │
│  Dixon-Coles + XGBoost + Market Odds             │
│  → P(goles_local, goles_visitante)               │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│          Expected Score Calculator                │
│  EP(i,j) = P(exacto)·5 + P(goles)·2 + ...        │
│  + ownership + contrarian value                   │
└──────────────────────┬───────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   CLI    │ │ FastAPI  │ │ Streamlit│
    │ (typer)  │ │  (REST)  │ │  (Web UI)│
    └──────────┘ └──────────┘ └──────────┘
```

## Instalación

```bash
# Clonar
git clone https://github.com/JhofredCam/BestBetWC.git
cd BestBetWC

# Instalar el proyecto
pip install -e .

# Instalar dependencias de desarrollo
pip install -e ".[dev]"
```

### Variables de entorno

Copiá `.env.example` a `.env` y completá las API keys:

```bash
cp .env.example .env
```

```env
THE_ODDS_API_KEY=tu_key     # https://the-odds-api.com (free: 500 req/mes)
API_FOOTBALL_KEY=tu_key     # RapidAPI API-Football (free: 100 req/día)
DATABASE_URL=sqlite:///data/bestbetwc.db
```

> Sin API keys el sistema funciona en modo offline usando datos históricos y el modelo Dixon-Coles con parámetros manuales.

## Uso

### CLI — Línea de comandos

```bash
# Info general y reglas
bestbet info

# Predecir un partido (con lambdas manuales)
bestbet predict "Brasil" "Argentina" --home-lambda 1.8 --away-lambda 1.2 --position 3

# Simular un partido (Monte Carlo)
bestbet simulate-match --home-lambda 1.5 --away-lambda 1.0 --simulations 10000
```

### API REST — FastAPI

```bash
# Iniciar servidor (http://localhost:8000)
uvicorn src.api.app:app --reload

# Documentación interactiva
open http://localhost:8000/docs
```

**Endpoints disponibles:**

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/predictions` | Generar predicción para un partido |
| `GET` | `/api/predictions/match/{id}` | Predicciones para un partido |
| `GET` | `/api/predictions/upcoming` | Partidos próximos |
| `GET` | `/api/matches` | Listar partidos |
| `GET` | `/api/matches/{id}` | Detalle de partido |
| `GET` | `/api/strategies/modes` | Modos de estrategia disponibles |
| `GET` | `/api/strategies/optimal/{position}` | Estrategia óptima según posición |
| `POST` | `/api/simulation/match` | Simular un partido |
| `POST` | `/api/simulation/tournament` | Simular torneo completo |
| `POST` | `/api/backtesting` | Ejecutar backtest |
| `GET` | `/api/backtesting/strategies` | Estrategias disponibles |
| `GET` | `/api/standings` | Clasificación |
| `GET` | `/api/profiles` | Perfiles de participantes |
| `POST` | `/api/data/update` | Actualizar datos desde APIs |

### Web UI — Streamlit

```bash
# Iniciar interfaz web (http://localhost:8501)
streamlit run src/web/app.py

# O con el entry point del proyecto
bestbet-web
```

**Páginas:**

| Página | Descripción |
|---|---|
| Dashboard | Resumen general, próximos partidos, posición actual |
| Predecir Partido | Generar predicción óptima con visualización de score matrix |
| Estrategia | Modo de estrategia según posición, risk/reward |
| Simular | Monte Carlo de partido o torneo completo |
| Clasificación | Tabla de posiciones y puntuaciones |
| Perfiles | Perfiles de los 15 participantes (estilo de juego inferido) |

## Desarrollo

```bash
# Tests (todos)
pytest

# Tests con coverage
pytest --cov=src

# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Formateo
ruff format src/
```

### Estructura del proyecto

```
src/
├── api/             # FastAPI backend (routers, schemas, dependencies)
├── cli/             # CLI con Typer (predict, simulate-match, info)
├── config.py        # Configuración global + reglas de la polla
├── database/        # SQLAlchemy ORM (13 modelos, connection pool)
├── etl/             # Extractores: The Odds API, API-Football, FBref, polla scraper
├── features/        # Feature pipeline: mercado, rendimiento, contexto (52 features)
├── game_theory/     # Perfilado, ownership, opponent modeling
├── models/          # Dixon-Coles, Gradient Boost (XGBoost), Ensemble
├── optimization/    # Expected Score, Strategy Selector, Bracket Optimizer
├── simulation/      # Monte Carlo (torneo + participantes), Backtesting
├── validation/      # Backtesting engine, métricas
└── web/             # Streamlit UI (6 páginas, componentes reutilizables)
```

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| ML/Stats | scikit-learn, XGBoost, scipy |
| Base de datos | SQLite (MVP) vía SQLAlchemy 2.0 |
| API HTTP | FastAPI + uvicorn |
| Web UI | Streamlit |
| CLI | Typer + Rich |
| Scraping | Playwright + BeautifulSoup4 |
| Async HTTP | httpx |
| Testing | pytest + pytest-asyncio |
| Linting | ruff + mypy |

## Métrica principal

**Expected Score (EP)** — NO accuracy. El objetivo es maximizar puntos esperados en la polla, no precisión predictiva. Un marcador menos probable pero poco popular puede tener mayor EP que el favorito del mercado gracias al bono de predicción única (+2 pts).

Ver [SPEC.md](SPEC.md) para la especificación técnica completa del sistema.
