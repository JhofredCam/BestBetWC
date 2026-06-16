# SPEC-018: Web UI (Alternative to CLI)

## Status: PLANNED

## Objective

Interfaz web interactiva con **Streamlit** como alternativa al CLI. Permite visualizar
predicciones, estrategias y resultados de forma intuitiva, bonita y sin necesidad
de abrir una terminal.

## Dependencies

- **SPEC-017** (FastAPI backend) — la UI se comunica con la API via HTTP requests internos
- O bien usa directamente los módulos de Python (`DixonColes`, `EPCalculator`, etc.)

## Context

El CLI actual (`bestbet predict`) es funcional pero poco amigable para consultas
rápidas pre-partido. Streamlit ofrece:
- UI en **puro Python**, sin HTML/CSS/JS manual
- Widgets interactivos nativos (sliders, selects, tabs, progress bars)
- Charts integrados (st.line_chart, st.bar_chart, st.plotly_chart)
- Layout responsive automático
- Hot-reload en desarrollo
- Despliegue trivial: `streamlit run src/web/app.py`

### Alternativas consideradas y descartadas

| Opción | Motivo descarte |
|---|---|
| Jinja2 + HTMX + Pico CSS | Demasiado HTML/CSS manual, no tan bonito sin diseño |
| React + Vite | Build step, JS tooling, overkill para 6-7 vistas |
| Streamlit | Elegido: Python puro, widgets nativos, bonito por defecto |

## Technical Design

### `src/web/`

```
src/web/
├── __init__.py
├── app.py              # Entry point: streamlit run src/web/app.py
├── pages/
│   ├── __init__.py
│   ├── 1_dashboard.py       # Dashboard principal
│   ├── 2_predict.py         # Predicción de partidos
│   ├── 3_strategy.py        # Estrategia adaptativa
│   ├── 4_simulate.py        # Simulador Monte Carlo
│   ├── 5_standings.py       # Clasificación de la polla
│   └── 6_profiles.py        # Perfiles de participantes
├── components/
│   ├── __init__.py
│   ├── score_heatmap.py     # Heatmap de score matrix
│   ├── ep_chart.py          # Gráfico de EP por marcador
│   ├── match_card.py        # Tarjeta de partido
│   ├── strategy_badge.py    # Badge de modo de estrategia
│   └── profile_card.py      # Card de perfil de jugador
├── state.py                 # Session state management
└── api_client.py            # Cliente HTTP para SPEC-017 (o usa módulos directos)
```

### `src/web/app.py` — Entry point

```python
"""
BestBetWC Web UI
Ejecutar: streamlit run src/web/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="BestBetWC",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(
    "https://img.icons8.com/color/48/world-cup.png",
    size="large",
)


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚽ BestBetWC")
    st.caption("Mundial 2026")

    position = st.selectbox(
        "Tu posición actual",
        options=list(range(1, 16)),
        index=2,
        format_func=lambda p: f"{p}° {'🥇' if p == 1 else '🥈' if p == 2 else '🥉' if p == 3 else ''}",
    )

    st.divider()

    total_participants = st.number_input(
        "Participantes", min_value=2, max_value=100, value=15,
    )

    st.divider()
    st.caption("v0.2.0 — Fase 2")


# ── Navegación ─────────────────────────────────────────────────────

pages = {
    "Dashboard": "src/web/pages/1_dashboard.py",
    "Predecir Partido": "src/web/pages/2_predict.py",
    "Estrategia": "src/web/pages/3_strategy.py",
    "Simular": "src/web/pages/4_simulate.py",
    "Clasificación": "src/web/pages/5_standings.py",
    "Perfiles": "src/web/pages/6_profiles.py",
}

pg = st.navigation([st.Page(path, title=title) for title, path in pages.items()])
pg.run()
```

### `src/web/pages/1_dashboard.py` — Dashboard

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.config import POLLA_RULES

st.title("Dashboard")

# ── Métricas principales ───────────────────────────────────────────

position = st.session_state.get("position", 3)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Posición", f"#{position} de {POLLA_RULES.num_participants}",
              delta=None, help="Tu posición actual en la polla")
with col2:
    st.metric("Puntos Totales", "142 pts",
              delta="+8 vs fecha anterior", help="Puntos acumulados")
with col3:
    st.metric("Win Probability", "18.5%",
              delta="+2.1%", help="Probabilidad de ganar la polla")
with col4:
    st.metric("Próximo Partido", "BRA vs ARG",
              delta="12 Jun", help="Fecha del próximo partido")

# ── Próximo partido - predicción rápida ────────────────────────────

st.subheader("Pronóstico Recomendado")

model = DixonColes(max_goals=POLLA_RULES.max_goals)
prediction = model.predict_from_params(lambda_h=1.8, mu_a=1.2)

col1, col2 = st.columns([1, 2])
with col1:
    ep_calc = ExpectedScoreCalculator()
    selector = StrategySelector()
    recommendation = selector.get_recommendation(
        prediction, position, POLLA_RULES.num_participants,
    )

    st.metric(
        label="Marcador Óptimo",
        value=f"{recommendation.prediction.home_goals} - {recommendation.prediction.away_goals}",
        delta=f"EP: {recommendation.prediction.ep_total:.2f} pts",
    )

    st.caption(f"Estrategia: **{recommendation.strategy_mode.value}**")
    st.caption(recommendation.reasoning)

    st.metric("Risk Score", f"{recommendation.risk_score:.0%}")
    st.metric("Upside Potential", f"{recommendation.upside_potential:.2f} pts")

with col2:
    # Gráfico de probabilidades de resultado
    fig = go.Figure(data=[
        go.Bar(
            x=["Victoria\nLocal", "Empate", "Victoria\nVisitante"],
            y=[prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob],
            marker_color=["#2ecc71", "#f1c40f", "#e74c3c"],
            text=[f"{p:.1%}" for p in [
                prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob
            ]],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Probabilidades de Resultado",
        yaxis=dict(title="Probabilidad", tickformat=".0%"),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Top marcadores por EP ──────────────────────────────────────────

st.subheader("Top 5 Marcadores por Expected Score")

ranked = ep_calc.rank_all_predictions(prediction)[:5]

scores = [f"{r.home_goals}-{r.away_goals}" for r in ranked]
eps = [r.ep_total for r in ranked]
exact_probs = [r.prob_exact for r in ranked]
result_probs = [r.prob_result for r in ranked]

fig = go.Figure(data=[
    go.Bar(name="EP Exacto", x=scores, y=[r.ep_exact for r in ranked],
           marker_color="#27ae60"),
    go.Bar(name="EP Resultado", x=scores, y=[r.ep_result for r in ranked],
           marker_color="#2980b9"),
    go.Bar(name="EP Goles", x=scores,
           y=[r.ep_goals_home + r.ep_goals_away for r in ranked],
           marker_color="#8e44ad"),
    go.Bar(name="EP Único", x=scores, y=[r.ep_unique for r in ranked],
           marker_color="#e67e22"),
])
fig.update_layout(
    title="Componentes del Expected Score",
    barmode="stack",
    yaxis=dict(title="Expected Points"),
    height=300,
)
st.plotly_chart(fig, use_container_width=True)

# ── Mini clasificación ─────────────────────────────────────────────

st.subheader("Top 5 Clasificación")

standings_data = [
    {"pos": 1, "nombre": "Carlos", "pts": 156, "exactos": 5, "delta": "—"},
    {"pos": 2, "nombre": "María", "pts": 148, "exactos": 4, "delta": "-8"},
    {"pos": 3, "nombre": "Tú", "pts": 142, "exactos": 3, "delta": "-14"},
    {"pos": 4, "nombre": "Juan", "pts": 138, "exactos": 2, "delta": "-18"},
    {"pos": 5, "nombre": "Ana", "pts": 135, "exactos": 3, "delta": "-21"},
]
st.dataframe(
    standings_data,
    column_config={
        "pos": st.column_config.NumberColumn("#", width="small"),
        "nombre": "Participante",
        "pts": st.column_config.NumberColumn("Puntos", format="%d"),
        "exactos": st.column_config.NumberColumn("Exactos", width="small"),
        "delta": st.column_config.TextColumn("Δ", width="small"),
    },
    hide_index=True,
    use_container_width=True,
)
```

### `src/web/pages/2_predict.py` — Predicción

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.config import POLLA_RULES

st.title("Predecir Partido")

# ── Formulario ─────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.text_input("Equipo Local", "Brasil")
    home_lambda = st.slider("Goles esperados Local (λ)", 0.1, 5.0, 1.8, 0.1)
with col2:
    away_team = st.text_input("Equipo Visitante", "Argentina")
    away_lambda = st.slider("Goles esperados Visitante (μ)", 0.1, 5.0, 1.2, 0.1)
with col3:
    position = st.selectbox(
        "Tu Posición",
        options=list(range(1, 16)),
        index=2,
        format_func=lambda p: f"{p}° {'🥇 Líder' if p==1 else '🥈' if p==2 else ''}",
    )

if st.button("⚽ Calcular Pronóstico Óptimo", type="primary", use_container_width=True):

    model = DixonColes(max_goals=POLLA_RULES.max_goals)
    prediction = model.predict_from_params(home_lambda, away_lambda)
    ep_calc = ExpectedScoreCalculator()
    ranked = ep_calc.rank_all_predictions(prediction)

    # ── Resultados ─────────────────────────────────────────────────

    st.divider()
    st.subheader(f"{home_team} vs {away_team}")

    # Score matrix heatmap
    col1, col2 = st.columns([1, 2])

    with col1:
        # Heatmap con Plotly
        z = prediction.score_matrix * 100
        fig = px.imshow(
            z,
            text_auto=".1f",
            labels=dict(x=f"Goles {away_team}", y=f"Goles {home_team}"),
            color_continuous_scale="Greens",
            zmin=0,
            zmax=float(z.max()),
        )
        fig.update_layout(
            title="Probabilidades de Marcador (%)",
            height=400,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top 10 tabla
        top10 = ranked[:10]
        table_data = []
        for i, r in enumerate(top10):
            icon = "★ " if i == 0 else ""
            table_data.append({
                "#": i + 1,
                "Marcador": f"{icon}{r.home_goals}-{r.away_goals}",
                "EP Total": f"{r.ep_total:.2f}",
                "P(Exacto)": f"{r.prob_exact:.1%}",
                "P(Result)": f"{r.prob_result:.1%}",
                "P(Goles)": f"{r.prob_goals_home + r.prob_goals_away:.1%}",
            })

        st.dataframe(
            table_data,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "Marcador": st.column_config.TextColumn("Marcador"),
                "EP Total": st.column_config.TextColumn("EP Total"),
                "P(Exacto)": st.column_config.TextColumn("P(Exacto)"),
                "P(Result)": st.column_config.TextColumn("P(Result)"),
                "P(Goles)": st.column_config.TextColumn("P(Goles)"),
            },
            hide_index=True,
            use_container_width=True,
        )

    # ── Gráfico de EP apilado ───────────────────────────────────────

    st.subheader("Componentes del Expected Score — Top 10")

    scores = [f"{r.home_goals}-{r.away_goals}" for r in top10]
    fig = go.Figure(data=[
        go.Bar(name="Exacto", x=scores, y=[r.ep_exact for r in top10],
               marker_color="#27ae60"),
        go.Bar(name="Resultado", x=scores, y=[r.ep_result for r in top10],
               marker_color="#2980b9"),
        go.Bar(name="Goles", x=scores,
               y=[r.ep_goals_home + r.ep_goals_away for r in top10],
               marker_color="#8e44ad"),
        go.Bar(name="Único", x=scores, y=[r.ep_unique for r in top10],
               marker_color="#e67e22"),
    ])
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="Expected Points"),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
```

### `src/web/pages/3_strategy.py` — Estrategia

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.optimization.strategy import StrategySelector, StrategyMode

st.title("Estrategia Adaptativa")

st.markdown("""
El sistema ajusta automáticamente tu estrategia según tu posición en la tabla.
La **estrategia óptima** no es la misma si vas liderando que si vas último:
quien lidera debe minimizar riesgo; quien persigue debe diferenciarse.
""")

# ── Simulador de posición ──────────────────────────────────────────

position = st.slider(
    "Simula tu posición en la tabla",
    min_value=1, max_value=15, value=3,
    format="%d°",
)

selector = StrategySelector()
mode = selector.determine_mode(position, 15)

# ── Modos de estrategia ────────────────────────────────────────────

modes_info = {
    StrategyMode.MINIMIZE_RISK: {
        "title": "🛡️ Minimizar Riesgo",
        "description": "Eliges marcadores de alta probabilidad. Evitas diferenciación innecesaria.",
        "color": "#2ecc71",
        "risk": "Bajo",
        "position": "1°",
    },
    StrategyMode.BALANCED: {
        "title": "⚖️ Balanceado",
        "description": "Mezclas predicciones seguras con algunas apuestas diferenciadas.",
        "color": "#3498db",
        "risk": "Medio",
        "position": "2°-5°",
    },
    StrategyMode.DIFFERENTIATION: {
        "title": "🎯 Diferenciación",
        "description": "Buscas marcadores poco populares con alta probabilidad real de ocurrir.",
        "color": "#e67e22",
        "risk": "Medio-Alto",
        "position": "6°-10°",
    },
    StrategyMode.HIGH_RISK: {
        "title": "🚀 Alto Riesgo",
        "description": "Apuestas agresivas: máximo upside, aceptando alta varianza.",
        "color": "#e74c3c",
        "risk": "Alto",
        "position": "11°-15°",
    },
}

cols = st.columns(4)
for i, (mode_enum, info) in enumerate(modes_info.items()):
    with cols[i]:
        is_active = mode_enum == mode
        st.markdown(f"""
        <div style="border: 2px solid {info['color']};
                    border-radius: 10px; padding: 15px;
                    {'background-color: rgba(255,255,255,0.05)' if not is_active
                     else 'background-color: ' + info['color'] + '22'};
                    {'box-shadow: 0 0 15px ' + info['color'] + '44' if is_active else ''}">
            <h4>{info['title']}</h4>
            <p style="font-size: 0.85em; color: #888;">{info['description']}</p>
            <p style="font-size: 0.8em;">Posición: <strong>{info['position']}</strong></p>
            <p style="font-size: 0.8em;">Riesgo: <strong>{info['risk']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ── Visualización de rangos ────────────────────────────────────────

st.subheader("Mapa de Estrategias")

fig = go.Figure()
ranges = [
    (1, 1, "Minimizar Riesgo", "#2ecc71"),
    (2, 5, "Balanceado", "#3498db"),
    (6, 10, "Diferenciación", "#e67e22"),
    (11, 15, "Alto Riesgo", "#e74c3c"),
]
for start, end, label, color in ranges:
    fig.add_trace(go.Bar(
        y=[label], x=[end - start + 1],
        orientation="h", marker_color=color,
        opacity=0.7 if (start <= position <= end) else 0.3,
        text=f"Puestos {start}-{end}",
        textposition="inside",
    ))

fig.update_layout(
    xaxis=dict(title="Número de posiciones en este rango"),
    showlegend=False,
    height=200,
)
st.plotly_chart(fig, use_container_width=True)

# ── Tu modo actual ─────────────────────────────────────────────────

active = modes_info[mode]
st.success(
    f"**Tu estrategia actual ({position}°): {active['title']}**\n\n"
    f"Riesgo: {active['risk']} · {active['description']}"
)
```

### `src/web/pages/4_simulate.py` — Simulación Monte Carlo

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.config import POLLA_RULES

st.title("Simulador Monte Carlo")

col1, col2, col3 = st.columns(3)
with col1:
    home_lambda = st.slider("λ Local", 0.1, 5.0, 1.5, 0.1)
with col2:
    away_lambda = st.slider("μ Visitante", 0.1, 5.0, 1.0, 0.1)
with col3:
    n_simulations = st.select_slider(
        "Simulaciones", options=[100, 1000, 5000, 10000, 50000], value=10000,
    )

if st.button("🎲 Ejecutar Simulación", type="primary", use_container_width=True):
    model = DixonColes(max_goals=POLLA_RULES.max_goals)
    prediction = model.predict_from_params(home_lambda, away_lambda)

    with st.spinner(f"Ejecutando {n_simulations:,} simulaciones..."):
        home_goals_sim = np.random.choice(
            len(prediction.home_goals_dist),
            size=n_simulations, p=prediction.home_goals_dist,
        )
        away_goals_sim = np.random.choice(
            len(prediction.away_goals_dist),
            size=n_simulations, p=prediction.away_goals_dist,
        )

        ep_calc = ExpectedScoreCalculator()
        top5 = ep_calc.rank_all_predictions(prediction)[:5]

        results = []
        for r in top5:
            eps = np.zeros(n_simulations)
            for s in range(n_simulations):
                h, a = home_goals_sim[s], away_goals_sim[s]
                if h == r.home_goals and a == r.away_goals:
                    eps[s] = POLLA_RULES.exact_score_pts
                    eps[s] += POLLA_RULES.goals_home_correct_pts
                    eps[s] += POLLA_RULES.goals_away_correct_pts
                else:
                    if h == r.home_goals:
                        eps[s] += POLLA_RULES.goals_home_correct_pts
                    if a == r.away_goals:
                        eps[s] += POLLA_RULES.goals_away_correct_pts
                    if ((r.home_goals > r.away_goals and h > a)
                        or (r.home_goals == r.away_goals and h == a)
                        or (r.home_goals < r.away_goals and h < a)):
                        eps[s] += POLLA_RULES.result_correct_pts

            results.append({
                "marcador": f"{r.home_goals}-{r.away_goals}",
                "ep_mean": eps.mean(),
                "ep_std": eps.std(),
                "ep_min": eps.min(),
                "ep_max": eps.max(),
            })

    # ── Gráfico ─────────────────────────────────────────────────────

    st.subheader(f"Resultados — {n_simulations:,} iteraciones")

    scores = [r["marcador"] for r in results]
    means = [r["ep_mean"] for r in results]
    stds = [r["ep_std"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=["#27ae60", "#2980b9", "#8e44ad", "#e67e22", "#e74c3c"],
        text=[f"{m:.2f} ± {s:.2f}" for m, s in zip(means, stds)],
        textposition="outside",
    ))
    fig.update_layout(
        title="Expected Score Promedio ± 1σ",
        yaxis=dict(title="Expected Points"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de métricas ───────────────────────────────────────────

    st.dataframe(
        [{
            "Marcador": r["marcador"],
            "EP Medio": f"{r['ep_mean']:.2f}",
            "EP Desv": f"{r['ep_std']:.2f}",
            "EP Mín": f"{r['ep_min']:.0f}",
            "EP Máx": f"{r['ep_max']:.0f}",
        } for r in results],
        hide_index=True,
        use_container_width=True,
    )
```

### `src/web/pages/5_standings.py` — Clasificación

```python
import streamlit as st

st.title("Clasificación de la Polla")

st.dataframe(
    [
        {"#": 1, "Participante": "Carlos", "Pts": 156, "Exactos": 5, "Resultados": 32,
         "Última fecha": "+12 pts"},
        {"#": 2, "Participante": "María", "Pts": 148, "Exactos": 4, "Resultados": 30,
         "Última fecha": "+8 pts"},
        {"#": 3, "Participante": "⭐ Tú", "Pts": 142, "Exactos": 3, "Resultados": 31,
         "Última fecha": "+10 pts"},
        {"#": 4, "Participante": "Juan", "Pts": 138, "Exactos": 2, "Resultados": 29,
         "Última fecha": "+6 pts"},
        {"#": 5, "Participante": "Ana", "Pts": 135, "Exactos": 3, "Resultados": 28,
         "Última fecha": "+14 pts"},
    ],
    column_config={
        "#": st.column_config.NumberColumn("#", width="small"),
        "Participante": st.column_config.TextColumn("Participante"),
        "Pts": st.column_config.NumberColumn("Puntos", format="%d"),
        "Exactos": st.column_config.NumberColumn("Exactos", width="small"),
        "Resultados": st.column_config.NumberColumn("Resultados", width="small"),
        "Última fecha": st.column_config.TextColumn("Última fecha"),
    },
    hide_index=True,
    use_container_width=True,
)
```

### `src/web/pages/6_profiles.py` — Perfiles

```python
import streamlit as st
import plotly.graph_objects as go

st.title("Perfiles de Participantes")

tabs = st.tabs([f"Jugador {i+1}" for i in range(5)])

for i, tab in enumerate(tabs):
    with tab:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"Jugador {i+1}")
            st.metric("Arquetipo", "Conservador")
            st.metric("Result Accuracy", "58%")
            st.metric("Exact Accuracy", "8%")
            st.metric("Avg. Points/Match", "2.1")

        with col2:
            # Radar chart de scores
            categories = ["Conservador", "Agresivo", "Market\nFollower", "Intuición",
                          "Home Bias", "Draw\nAversion"]
            values = [0.7, 0.2, 0.5, 0.3, 0.6, 0.4]

            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                marker=dict(color="#27ae60"),
            ))
            fig.update_layout(height=350, polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
```

### `src/web/api_client.py` — Comunicación con la API

```python
"""
Cliente HTTP para consumir la API de SPEC-017.
Se usa cuando los datos vienen de BD en lugar de parámetros directos.
"""

import httpx
from typing import Optional
from urllib.parse import urljoin

API_BASE = "http://localhost:8000/api"

async def get_match(match_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API_BASE}/matches/{match_id}")
        r.raise_for_status()
        return r.json()

async def predict_match(
    home_team: str, away_team: str,
    home_lambda: float, away_lambda: float, position: int,
) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{API_BASE}/predictions", json={
            "home_team": home_team,
            "away_team": away_team,
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            "current_position": position,
        })
        r.raise_for_status()
        return r.json()

async def get_standings() -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API_BASE}/standings")
        r.raise_for_status()
        return r.json()

async def get_profiles() -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API_BASE}/profiles")
        r.raise_for_status()
        return r.json()
```

## Acceptance Criteria

### General
- [ ] `streamlit run src/web/app.py` levanta la UI en `localhost:8501`
- [ ] Navegación entre páginas funciona (sidebar + st.navigation)
- [ ] Layout responsive (se adapta a mobile)
- [ ] Hot-reload: cambios en código se reflejan sin reiniciar manualmente
- [ ] Sin build step, sin node, sin npm

### Dashboard
- [ ] 4 métricas en cards (posición, puntos, WinProb, próximo partido)
- [ ] Gráfico de barras de probabilidades de resultado
- [ ] Gráfico de barras apiladas de componentes EP (top 5)
- [ ] Tabla mini-clasificación (top 5)
- [ ] Posición se lee de sidebar/st.session_state

### Predict Page
- [ ] 3 columnas: local, visitante, posición
- [ ] Sliders muestran valor numérico
- [ ] Botón "Calcular" dispara predicción
- [ ] Heatmap interactivo de score matrix (Plotly)
- [ ] Tabla con top 10 marcadores y métricas
- [ ] Gráfico de barras apiladas de EP

### Strategy Page
- [ ] 4 cards de modos de estrategia con descripciones
- [ ] Card activo resaltado con glow
- [ ] Slider de posición actualiza modo activo en tiempo real
- [ ] Barras horizontales mostrando rangos

### Simulation Page
- [ ] Sliders para lambdas
- [ ] Select de N simulaciones
- [ ] Spinner durante ejecución
- [ ] Gráfico de barras con error bars (±1σ)
- [ ] Tabla de métricas por marcador

### Standings Page
- [ ] Tabla con todas las columnas
- [ ] Tu fila resaltada

### Profiles Page
- [ ] Tabs por participante
- [ ] Métricas y radar chart por perfil

### Tests
- [ ] Test: `streamlit run` levanta sin errores
- [ ] Test: componentes importables y ejecutables
- [ ] Test: `api_client.py` maneja errores HTTP
- [ ] Test: integración con DixonColes y EP calculator

## Files to Create

```
src/web/__init__.py
src/web/app.py
src/web/state.py
src/web/api_client.py
src/web/pages/__init__.py
src/web/pages/1_dashboard.py
src/web/pages/2_predict.py
src/web/pages/3_strategy.py
src/web/pages/4_simulate.py
src/web/pages/5_standings.py
src/web/pages/6_profiles.py
src/web/components/__init__.py
src/web/components/score_heatmap.py
src/web/components/ep_chart.py
src/web/components/match_card.py
src/web/components/strategy_badge.py
src/web/components/profile_card.py
tests/test_web.py
```

## Pyproject.toml addition

```toml
[project]
dependencies = [
    ...
    "streamlit>=1.35.0",
    "plotly>=5.15.0",  # Ya existe
]

[project.scripts]
bestbet = "src.cli.main:app"
bestbet-web = "streamlit.web.cli:main run src/web/app.py"
```

## Ejecución

```bash
# Desarrollo (hot-reload)
streamlit run src/web/app.py

# Producción
bestbet-web
```

## Git Workflow

```bash
git checkout -b feature/spec-018-web-ui

# Commit 1: estructura base + componentes
git add src/web/__init__.py src/web/app.py src/web/state.py src/web/api_client.py
git add src/web/pages/__init__.py src/web/components/__init__.py
git commit -m "feat(SPEC-018): add Streamlit app structure with navigation and API client"

# Commit 2: dashboard + predict pages
git add src/web/pages/1_dashboard.py src/web/pages/2_predict.py
git add src/web/components/
git commit -m "feat(SPEC-018): add dashboard and predict pages with Plotly charts"

# Commit 3: strategy + simulate pages
git add src/web/pages/3_strategy.py src/web/pages/4_simulate.py
git commit -m "feat(SPEC-018): add strategy and simulation pages"

# Commit 4: standings + profiles pages
git add src/web/pages/5_standings.py src/web/pages/6_profiles.py
git commit -m "feat(SPEC-018): add standings and profiles pages"

# Commit 5: update pyproject.toml + tests
git add pyproject.toml tests/test_web.py
git commit -m "feat(SPEC-018): add streamlit dep, bestbet-web script, and web tests"

# Verify
pytest tests/test_web.py -v
ruff check src/web/
streamlit run src/web/app.py  # Aceptar manualmente

# CI green → merge
git checkout main
git merge feature/spec-018-web-ui

# Push
git push origin main
```

## Notes

- Streamlit es stateful: usar `st.session_state` para posición, configuración
- `st.navigation` + `st.Page` (Streamlit 1.35+) para navegación multi-página
- Plotly para gráficos interactivos (heatmap, barras, radar)
- Los componentes (`src/web/components/`) encapsulan widgets reusables
- `api_client.py` se comunica con SPEC-017 si está corriendo; si no, usa módulos directos
- En desarrollo concurrente con SPEC-017, usar `httpx` para llamadas HTTP internas
- Si SPEC-017 no está implementado, usar directamente DixonColes/EPCalculator en memoria
- NO requiere Node.js, npm, build tools, ni configuraciones complejas
- Peso total de dependencias extra: solo Streamlit (~15MB) + Plotly (ya incluido)
