# SPEC-019: UX Simplificada — Enfoque No-Técnico

## Status: PLANNED

## Objective

Transformar la interfaz web de un dashboard técnico (orientado a data scientists) a una
herramienta usable por cualquier persona sin conocimiento de modelos, lambdas, ni teoría
de juegos. El foco es: **partidos de hoy y mañana con sugerencias directas de pronóstico,
y una tabla de posiciones editable**.

## Dependencies

- SPEC-002 (database layer)
- SPEC-006 (The Odds API — cuotas)
- SPEC-007 (API-Football — fixtures, resultados)
- SPEC-010 (Model Ensemble)
- SPEC-012 (Game Theory — ownership)
- SPEC-017 (FastAPI backend)
- SPEC-018 (Streamlit UI — a modificar)

## Context

### Problema actual

El usuario debe ingresar manualmente nombres de equipos, lambdas esperados, posición, etc.
Esto requiere conocimiento técnico que el usuario final no tiene ni debería necesitar.

### Solución

El sistema debe:
1. **Auto-poblar** partidos desde la base de datos (datos reales de API-Football)
2. **Ejecutar modelos automáticamente** al cargar la página
3. **Mostrar sugerencias en lenguaje natural**: "Te sugerimos predecir 2-1. Motivo: este
   marcador tiene buena probabilidad (8.2%) y solo el 12% de participantes lo elegiría,
   dándote +2 pts extra si acertás."
4. **Tabla de posiciones editable**: el usuario puede agregar/quitar participantes,
   editar puntajes manualmente, y el sistema recalcula estrategias en base a eso.

## Technical Design

### Cambios en `src/web/app.py`

- Eliminar el slider de posición y el input de participantes del sidebar
- Sidebar mínimo: solo navegación entre páginas y botón "Actualizar datos"

### Nueva página: `src/web/pages/1_dashboard.py`

Página principal que muestra:

```
┌─────────────────────────────────────────────────────────────┐
│  ⚽ Partidos de Hoy — 15 de junio 2026                       │
├─────────────────────────────────────────────────────────────┤
│  🇧🇷 Brasil vs Argentina 🇦🇷  |  ⏰ 20:00  | Grupo A         │
│  📊 Sugerencia: 2-1                                         │
│  💡 2-1 es probable (8.2%) y poco popular → +2 pts bonus    │
│  [Ver análisis completo]                                    │
├─────────────────────────────────────────────────────────────┤
│  🇩🇪 Alemania vs Francia 🇫🇷  |  ⏰ 17:00  | Grupo B         │
│  📊 Sugerencia: 1-1                                         │
│  💡 Empate técnico, alta probabilidad (12.1%)               │
│  [Ver análisis completo]                                    │
├─────────────────────────────────────────────────────────────┤
│  ⚽ Partidos de Mañana — 16 de junio 2026                    │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

- **Origen de datos**: consulta `GET /api/matches` filtrado por `status=scheduled` y
  `datetime` entre hoy 00:00 y mañana 23:59
- **Sugerencia**: endpoint `POST /api/predictions` con parámetros implícitos (posición
  tomada de la tabla de posiciones)
- **Sin inputs técnicos**: el usuario solo ve resultados, no ingresa lambdas ni
  parámetros de modelo

### Nueva página: `src/web/pages/2_predict.py` (Match Detail)

Cuando el usuario clickea "Ver análisis completo" en un partido del dashboard:

- Muestra nombre de equipos, fecha, grupo, estadio
- **Sugerencia principal** con marcador, explicación en lenguaje natural
- **Top 3 alternativas** con pros/contras
- **Gráfico de probabilidades** (heatmap simple de marcadores)
- **Contexto**: forma reciente, H2H, lesionados clave (si hay datos)

### Refactor: `src/web/pages/3_strategy.py` → Tabla de Posiciones editable

Reemplazar la página actual de estrategia por una tabla editable:

```
┌──────────────────────────────────────────────────────────────┐
│  📊 Tabla de Posiciones                          [Editar]    │
├────┬──────────────────┬────────┬───────┬────────────────────┤
│  # │ Participante     │ Puntos │ Pos.  │ Acciones           │
├────┼──────────────────┼────────┼───────┼────────────────────┤
│  1 │ Juan Pérez       │  45    │   1   │ ✏️ Editar  🗑️      │
│  2 │ María López      │  42    │   2   │ ✏️ Editar  🗑️      │
│  3 │ Tu (vos)         │  38    │   3   │ ✏️ Editar          │
│ ...│ ...              │  ...   │  ...  │                    │
├────┼──────────────────┼────────┼───────┼────────────────────┤
│    │ [+ Agregar participante]                               │
└──────────────────────────────────────────────────────────────┘
```

- **Editable con `st.data_editor()`** de Streamlit
- Persiste en la BD (`participants` + `standings`)
- El participante "actual" (vos) se marca visualmente
- Al editar, el sistema recalcula automáticamente la estrategia para "vos"

### Refactor: `src/web/pages/4_simulate.py` → simplificar

- Eliminar inputs de lambdas
- Pre-poblar con los partidos del día
- Botón "Simular mis pronósticos para hoy"
- Mostrar resultado en lenguaje natural: "Con estos pronósticos, tu posición esperada
  al final del torneo es 3° (rango: 1°-7°)"

### Mantener

- `src/web/pages/5_standings.py` → se fusiona con la tabla editable de `3_strategy`
- `src/web/pages/6_profiles.py` → se mantiene pero con datos reales de BD

### Endpoints a agregar/modificar en `src/api/`

Si no existen, agregar:

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/matches/today` | Partidos de hoy |
| `GET` | `/api/matches/tomorrow` | Partidos de mañana |
| `POST` | `/api/participants` | Crear participante |
| `PUT` | `/api/participants/{id}` | Editar participante |
| `DELETE` | `/api/participants/{id}` | Eliminar participante |
| `PUT` | `/api/standings/{participant_id}` | Actualizar puntaje |
| `GET` | `/api/matches/{id}/analysis` | Análisis completo (predicción + contexto + H2H) |

### Funciones helper en lenguaje natural

Crear `src/web/natural_language.py` con funciones que traduzcan datos técnicos a texto:

```python
def explain_recommendation(result: ExpectedScoreResult, strategy: StrategyMode) -> str:
    """Genera explicación en español simple."""

def format_match_context(match_data: dict) -> str:
    """Forma reciente, H2H, etc. en español."""

def strategy_advice(mode: StrategyMode, position: int) -> str:
    """Consejo según posición: 'Vas primero, jugá seguro'."""
```

## Acceptance Criteria

- [ ] Dashboard muestra partidos de hoy y mañana (de BD o mock si no hay datos)
- [ ] Cada partido muestra sugerencia de pronóstico con explicación en español simple
- [ ] Tabla de posiciones es editable con `st.data_editor`
- [ ] Se puede agregar, editar y eliminar participantes
- [ ] Al editar la tabla, la posición del usuario se actualiza automáticamente
- [ ] La página de predicción (detalle) no pide lambdas ni parámetros técnicos
- [ ] Las explicaciones están en español y no mencionan "EP", "lambda", "contrarian value"
- [ ] El sidebar es mínimo (solo navegación)
- [ ] `ruff check src/web/` pasa limpio
- [ ] `mypy src/web/` sin errores (con --ignore-missing-imports)

## Files to Create/Modify

```
NUEVOS:
  src/web/natural_language.py       # Traducción a español simple
  src/api/routers/participants.py   # CRUD de participantes

MODIFICAR:
  src/web/app.py                    # Sidebar mínimo, sin inputs técnicos
  src/web/pages/1_dashboard.py      # Partidos de hoy/mañana con sugerencias
  src/web/pages/2_predict.py        # Detalle de partido, sin lambdas
  src/web/pages/3_strategy.py       # → Tabla editable de posiciones
  src/web/pages/4_simulate.py       # Simplificado, sin lambdas
  src/web/pages/5_standings.py      # Fusionado con 3_strategy
  src/web/pages/6_profiles.py       # Con datos reales de BD
  src/api/routers/__init__.py       # Registrar nuevo router
  src/api/app.py                    # Incluir nuevo router

ELIMINAR:
  src/web/state.py                  # Ya no necesario (sin estado técnico)
```

## Git Workflow

```bash
git checkout -b feature/spec-019-ux-simplificada
git add <files>
git commit -m "feat(SPEC-019): simplify UX for non-technical users"
git push origin feature/spec-019-ux-simplificada
pytest tests/ -q --tb=short
ruff check src/web/ src/api/
mypy src/web/ src/api/ --ignore-missing-imports
git checkout main; git pull origin main; git merge feature/spec-019-ux-simplificada
git push origin main
```
