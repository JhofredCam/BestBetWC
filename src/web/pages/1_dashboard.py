from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.config import POLLA_RULES
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.web.state import get_position

st.title("Dashboard")

position = get_position()

# ── Métricas principales ───────────────────────────────────────────

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

fig2 = go.Figure(data=[
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
fig2.update_layout(
    title="Componentes del Expected Score",
    barmode="stack",
    yaxis=dict(title="Expected Points"),
    height=300,
)
st.plotly_chart(fig2, use_container_width=True)

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
