# SPEC-018: Web UI (Alternative to CLI)

## Status: PLANNED

## Objective

Interfaz web sencilla servida por FastAPI usando Jinja2 + HTMX como alternativa
al CLI. Permite visualizar predicciones, estrategias y resultados de forma
intuitiva sin necesidad de abrir una terminal.

## Dependencies

- **SPEC-017** (FastAPI backend - sirve la UI)

## Context

El CLI actual (`bestbet predict`) es funcional pero poco amigable para consultas
rápidas pre-partido. Una UI web permite:
- Ver el pronóstico óptimo de un vistazo en el celular antes del partido
- Explorar el Top 10 de marcadores con sus probabilidades
- Ver la clasificación de la polla en tiempo real
- Ejecutar simulaciones con sliders interactivos
- Dashboard con métricas clave de tu posición en la polla

### Stack elegido

| Componente | Tecnología | Justificación |
|---|---|---|
| Templates | **Jinja2** (incluido en FastAPI) | Sin dependencias extra, bien integrado |
| Interactividad | **HTMX 2.0** (CDN, sin build step) | Reemplaza React/Vue para interacciones simples |
| Estilos | **Pico CSS** (CDN, classless) | CSS minimalista, responsive, sin configurar |
| Charts | **Chart.js** (CDN) | Gráficos de probabilidades y EP |
| Empaquetado | Ninguno (CDN + archivos estáticos) | Cero build step, copiar y pegar |

Alternativa considerada y descartada:
- **Streamlit**: agrega mucha dependencia, el estado es complejo, menos control
- **React/Vue + Vite**: overkill para 5-6 páginas, requiere build step

## Technical Design

### `src/web/`

```
src/web/
├── __init__.py
├── routes.py           # Rutas de la UI (Jinja2 templates)
├── static/
│   ├── css/
│   │   └── app.css     # Overrides mínimos sobre Pico CSS
│   └── js/
│       └── app.js      # HTMX extensions, chart init
└── templates/
    ├── base.html       # Layout base (navbar, footer, CDN links)
    ├── index.html      # Dashboard principal
    ├── predict.html    # Formulario de predicción + resultados
    ├── predictions_list.html  # HTMX partial: lista de predicciones
    ├── strategy.html   # Recomendaciones de estrategia
    ├── simulate.html   # Simulador interactivo
    ├── standings.html  # Clasificación de la polla
    ├── profiles.html   # Perfiles de participantes
    ├── metrics.html    # Métricas de calibración y rendimiento
    └── components/
        ├── score_matrix_card.html  # HTMX partial: matriz de marcadores
        ├── ep_chart.html           # HTMX partial: gráfico EP
        ├── match_card.html         # HTMX partial: tarjeta de partido
        └── standing_row.html       # HTMX partial: fila de clasificación
```

### `src/web/routes.py`

```python
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard principal: próximo partido, posición, métricas clave."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "dashboard",
    })

@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Página de predicción de partidos."""
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "page": "predict",
    })

@router.post("/predict/load", response_class=HTMLResponse)
async def predict_load(
    request: Request,
    home_team: str = Form(...),
    away_team: str = Form(...),
    home_lambda: float = Form(1.5),
    away_lambda: float = Form(1.0),
    position: int = Form(1),
):
    """HTMX: carga resultados de predicción via POST al API interna."""
    # Llama internamente a POST /api/predictions
    # Retorna HTML partial con el resultado
    ...

@router.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request):
    """Página de estrategia adaptativa."""
    ...

@router.get("/simulate", response_class=HTMLResponse)
async def simulate_page(request: Request):
    """Simulador Monte Carlo interactivo."""
    ...

@router.get("/standings", response_class=HTMLResponse)
async def standings_page(request: Request):
    """Clasificación de la polla."""
    ...

@router.get("/profiles", response_class=HTMLResponse)
async def profiles_page(request: Request):
    """Perfiles de participantes."""
    ...

@router.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    """Dashboard de métricas de calibración."""
    ...
```

### `src/web/templates/base.html`

```html
<!DOCTYPE html>
<html lang="es" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}BestBetWC{% endblock %}</title>
    <!-- Pico CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <!-- HTMX -->
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <!-- Custom CSS -->
    <link rel="stylesheet" href="/static/css/app.css">
</head>
<body>
    <nav class="container-fluid">
        <ul>
            <li><strong>BestBetWC</strong></li>
        </ul>
        <ul>
            <li><a href="/" {% if page == 'dashboard' %}class="active"{% endif %}>Dashboard</a></li>
            <li><a href="/predict" {% if page == 'predict' %}class="active"{% endif %}>Predecir</a></li>
            <li><a href="/strategy" {% if page == 'strategy' %}class="active"{% endif %}>Estrategia</a></li>
            <li><a href="/simulate" {% if page == 'simulate' %}class="active"{% endif %}>Simular</a></li>
            <li><a href="/standings" {% if page == 'standings' %}class="active"{% endif %}>Clasificación</a></li>
            <li><a href="/profiles" {% if page == 'profiles' %}class="active"{% endif %}>Perfiles</a></li>
        </ul>
    </nav>

    <main class="container">
        {% block content %}{% endblock %}
    </main>

    <footer class="container">
        <small>BestBetWC v0.1.0 - Mundial 2026</small>
    </footer>

    <script src="/static/js/app.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

### `src/web/templates/index.html` - Dashboard

```html
{% extends "base.html" %}
{% block title %}Dashboard - BestBetWC{% endblock %}
{% block content %}

<h1>Dashboard</h1>

<div class="grid">
    <!-- Posición actual -->
    <article>
        <header>Tu Posición</header>
        <h2>#3 de 15</h2>
        <progress value="3" max="15"></progress>
    </article>

    <!-- Puntos totales -->
    <article>
        <header>Puntos Totales</header>
        <h2>142 pts</h2>
        <small>Promedio: 2.2 pts/partido</small>
    </article>

    <!-- Win Probability -->
    <article>
        <header>Win Probability</header>
        <h2>18.5%</h2>
        <small>Probabilidad de ganar la polla</small>
    </article>

    <!-- Próximo partido -->
    <article>
        <header>Próximo Partido</header>
        <h2>Brasil vs Argentina</h2>
        <small>12 Jun 2026 - 16:00</small>
    </article>
</div>

<!-- Próximo partido - predicción rápida -->
<section>
    <h2>Pronóstico Recomendado</h2>
    <div hx-get="/predict/match?home=Brasil&away=Argentina&pos=3"
         hx-trigger="load" hx-target="#quick-prediction">
        <div id="quick-prediction" aria-busy="true">Cargando...</div>
    </div>
</section>

<!-- Gráfico de Expected Score por marcador -->
<section>
    <h2>Top Marcadores por EP</h2>
    <canvas id="epChart" width="600" height="300"></canvas>
</section>

<!-- Clasificación resumida -->
<section>
    <h2>Top 5 Clasificación</h2>
    <div hx-get="/api/standings?limit=5" hx-trigger="load"
         hx-target="#mini-standings">
        <div id="mini-standings" aria-busy="true">Cargando...</div>
    </div>
</section>

{% endblock %}
```

### `src/web/templates/predict.html` - Predicción

```html
{% extends "base.html" %}
{% block title %}Predecir - BestBetWC{% endblock %}
{% block content %}

<h1>Predecir Partido</h1>

<form hx-post="/predict/load" hx-target="#prediction-results" hx-indicator="#spinner">
    <div class="grid">
        <label>
            Equipo Local
            <input type="text" name="home_team" value="Brasil" required>
        </label>
        <label>
            Equipo Visitante
            <input type="text" name="away_team" value="Argentina" required>
        </label>
    </div>

    <div class="grid">
        <label>
            Goles esperados Local (λ)
            <input type="range" name="home_lambda" min="0.1" max="5.0" step="0.1"
                   value="1.5" oninput="this.nextElementSibling.value = this.value">
            <output>1.5</output>
        </label>
        <label>
            Goles esperados Visitante (μ)
            <input type="range" name="away_lambda" min="0.1" max="5.0" step="0.1"
                   value="1.0" oninput="this.nextElementSibling.value = this.value">
            <output>1.0</output>
        </label>
    </div>

    <label>
        Tu Posición en la Polla
        <select name="position">
            <option value="1">1° - Liderando</option>
            <option value="3" selected>3° - Media tabla</option>
            <option value="8">8° - Zona media-baja</option>
            <option value="13">13° - Remontada necesaria</option>
        </select>
    </label>

    <button type="submit" class="contrast">Calcular Pronóstico Óptimo</button>
</form>

<div id="spinner" class="htmx-indicator" aria-busy="true">Calculando...</div>

<div id="prediction-results">
    <!-- HTMX carga aquí los resultados -->
</div>

{% endblock %}
```

### `src/web/templates/components/score_matrix_card.html`

```html
<!-- HTMX partial: Heatmap de probabilidades de marcador -->
<article>
    <header>
        Score Matrix: {{ home_team }} vs {{ away_team }}
    </header>
    <div class="score-grid">
        <table class="score-matrix">
            <thead>
                <tr>
                    <th></th>
                    {% for j in range(8) %}
                    <th>{{ j }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for i in range(8) %}
                <tr>
                    <th>{{ i }}</th>
                    {% for j in range(8) %}
                    <td class="cell-{{ 'highlight' if i == rec_home and j == rec_away else 'normal' }}"
                        style="background-color: rgba(0,150,100, {{ matrix[i][j] * 5 }})">
                        {{ "%.1f%%"|format(matrix[i][j] * 100) }}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</article>
```

### `src/api/app.py` - Integración de UI

```python
# Agregar al crear la app:
from src.web.routes import router as web_router
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
app.include_router(web_router)
```

La UI se sirve en las mismas rutas que la API (mismo puerto 8000), sin CORS necesario.

### `src/web/static/js/app.js`

```javascript
// Init Chart.js para gráfico de EP
function initEPChart(canvasId, labels, values) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Expected Points',
                data: values,
                backgroundColor: 'rgba(0, 150, 100, 0.6)',
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

// HTMX after-swap: inicializar charts en contenido nuevo
document.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail.target.id === 'prediction-results') {
        // El template incluye <script> que llama a initEPChart
    }
});
```

## Screens / Pages

| Ruta | Descripción | HTMX interactividad |
|---|---|---|
| `/` | Dashboard con posición, EP, próximos partidos | Carga lazy de predicción y standings |
| `/predict` | Formulario: equipo A vs B, lambdas, posición | POST → carga top 10 marcadores + heatmap |
| `/strategy` | Explicación de modos, matriz posición→estrategia | Slider de posición actualiza recomendación |
| `/simulate` | Simulador: lambdas, N iteraciones | POST → gráfico de EP con error bars |
| `/standings` | Tabla de clasificación completa | Polling cada 60s via HTMX |
| `/profiles` | Cards de perfil de cada participante | Click expande detalles |
| `/metrics` | Gráficos de calibración, EP acumulado | Selector de torneo para backtesting |

## Acceptance Criteria

### General
- [ ] La UI carga en `http://localhost:8000/` (misma app que la API)
- [ ] Navegación entre páginas funciona sin recarga completa (HTMX boost)
- [ ] Responsive: se ve bien en mobile (Pico CSS maneja esto)
- [ ] Tema oscuro por defecto, respeta `prefers-color-scheme`
- [ ] No hay build step: templates y estáticos se sirven directamente

### Dashboard
- [ ] Muestra posición actual, puntos totales, Win Probability
- [ ] Próximo partido con predicción recomendada cargada via HTMX
- [ ] Gráfico de EP con Chart.js (top 10 marcadores)
- [ ] Mini-clasificación top 5

### Predict Page
- [ ] Formulario con sliders para lambdas y select para posición
- [ ] Sliders muestran valor numérico en tiempo real
- [ ] POST via HTMX carga resultados sin recargar página
- [ ] Heatmap de score matrix con celda destacada (recomendación)
- [ ] Tabla ordenable de top 10 marcadores
- [ ] Indicador de "cargando" mientras se calcula

### Strategy Page
- [ ] Muestra los 4 modos de estrategia con descripciones
- [ ] Selector de posición actualiza recomendación actual

### Simulation Page
- [ ] Formulario: lambdas + número de simulaciones
- [ ] Barra de progreso durante simulación
- [ ] Gráfico comparativo de EP para top 5 marcadores
- [ ] Métricas: EP medio, desviación, hit rates

### Standings Page
- [ ] Tabla completa de los 15 participantes
- [ ] Columnas: posición, nombre, puntos, exactos, resultados
- [ ] Auto-refresh cada 60 segundos (HTMX polling)

### Profiles Page
- [ ] Grid de cards con nombre y arquetipo de cada participante
- [ ] Click en card expande: radar chart de scores, sesgos, accuracy

### No-JS fallback
- [ ] Sin JavaScript, las páginas siguen funcionando (recarga completa)
- [ ] Formularios usan POST normal, no dependen de HTMX para funcionar

### Tests
- [ ] Test: cada ruta retorna HTML 200
- [ ] Test: formulario de predicción con datos válidos retorna HTML con resultados
- [ ] Test: formulario con datos inválidos muestra errores
- [ ] Test: templates contienen los elementos esperados (nav, form, table)

## Files to Create

```
src/web/__init__.py
src/web/routes.py
src/web/static/css/app.css
src/web/static/js/app.js
src/web/templates/base.html
src/web/templates/index.html
src/web/templates/predict.html
src/web/templates/strategy.html
src/web/templates/simulate.html
src/web/templates/standings.html
src/web/templates/profiles.html
src/web/templates/metrics.html
src/web/templates/components/score_matrix_card.html
src/web/templates/components/ep_chart.html
src/web/templates/components/match_card.html
src/web/templates/components/standing_row.html
tests/test_web.py
```

## Git Workflow

```bash
git checkout -b feature/spec-018-web-ui

# Commit 1: base layout + static assets
git add src/web/__init__.py src/web/routes.py
git add src/web/templates/base.html src/web/static/
git commit -m "feat(SPEC-018): add web UI base layout with Pico CSS + HTMX + Chart.js"

# Commit 2: dashboard + predict pages
git add src/web/templates/index.html src/web/templates/predict.html
git add src/web/templates/components/
git commit -m "feat(SPEC-018): add dashboard and prediction pages with HTMX interactivity"

# Commit 3: strategy + simulate + standings + profiles
git add src/web/templates/strategy.html src/web/templates/simulate.html
git add src/web/templates/standings.html src/web/templates/profiles.html src/web/templates/metrics.html
git commit -m "feat(SPEC-018): add strategy, simulation, standings, profiles, and metrics pages"

# Commit 4: integrate with FastAPI app
# Update src/api/app.py to mount static files + include web router
git add src/api/app.py
git commit -m "feat(SPEC-018): integrate web UI into FastAPI app (static mount + router)"

# Commit 5: tests
git add tests/test_web.py
git commit -m "test(SPEC-018): add web UI route and template tests"

pytest tests/test_web.py -v
ruff check src/web/

git checkout main
git merge feature/spec-018-web-ui
```

## Notes

- Pico CSS v2 es classless: el HTML semántico se estiliza automáticamente
- HTMX 2.0 usa `hx-get`, `hx-post`, `hx-trigger`, `hx-target`, `hx-swap`
- Los templates de componentes (partials) se renderizan solos o dentro de layouts
- Chart.js se inicializa desde `<script>` inline en cada partial que use gráficos
- El tema dark/light se controla con `data-theme="dark"` en `<html>`
- Si se necesita autenticación en el futuro, usar FastAPI's OAuth2 + Jinja2 sessions
- NO incluir node_modules, webpack, vite, ni ningún build step
- Todas las dependencias JS/CSS se cargan desde CDN (Pico CSS, HTMX, Chart.js)
- Los archivos estáticos se sirven desde `src/web/static/` montados en `/static/`
- Para development, `uvicorn src.api.app:app --reload` recarga templates automáticamente
