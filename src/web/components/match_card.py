"""
Tarjeta de partido con info de predicción.
"""

from __future__ import annotations

import streamlit as st

from src.models.dixon_coles import MatchPrediction
from src.optimization.strategy import StrategyRecommendation


def render_match_card(
    home_team: str,
    away_team: str,
    recommendation: StrategyRecommendation,
    match_prediction: MatchPrediction | None = None,
) -> None:
    pred = recommendation.prediction
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label="Marcador Óptimo",
            value=f"{pred.home_goals} - {pred.away_goals}",
            delta=f"EP: {pred.ep_total:.2f} pts",
        )
        st.caption(f"Estrategia: **{recommendation.strategy_mode.value}**")
        st.caption(recommendation.reasoning)
        st.metric("Risk Score", f"{recommendation.risk_score:.0%}")
        st.metric("Upside Potential", f"{recommendation.upside_potential:.2f} pts")

    with col2:
        if match_prediction is not None:
            _render_result_probs(match_prediction)


def _render_result_probs(match_prediction: MatchPrediction) -> None:
    import plotly.graph_objects as go

    probs = [
        match_prediction.home_win_prob,
        match_prediction.draw_prob,
        match_prediction.away_win_prob,
    ]
    fig = go.Figure(data=[
        go.Bar(
            x=["Victoria\nLocal", "Empate", "Victoria\nVisitante"],
            y=probs,
            marker_color=["#2ecc71", "#f1c40f", "#e74c3c"],
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Probabilidades de Resultado",
        yaxis=dict(title="Probabilidad", tickformat=".0%"),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
