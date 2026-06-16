"""
Gráfico de Expected Score por marcador.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from src.optimization.expected_score import ExpectedScoreResult


def render_ep_chart(
    results: Sequence[ExpectedScoreResult],
    title: str = "Componentes del Expected Score",
    height: int = 350,
) -> None:
    scores = [f"{r.home_goals}-{r.away_goals}" for r in results]
    fig = go.Figure(data=[
        go.Bar(name="Exacto", x=scores, y=[r.ep_exact for r in results],
               marker_color="#27ae60"),
        go.Bar(name="Resultado", x=scores, y=[r.ep_result for r in results],
               marker_color="#2980b9"),
        go.Bar(name="Goles", x=scores,
               y=[r.ep_goals_home + r.ep_goals_away for r in results],
               marker_color="#8e44ad"),
        go.Bar(name="Único", x=scores, y=[r.ep_unique for r in results],
               marker_color="#e67e22"),
    ])
    fig.update_layout(
        title=title,
        barmode="stack",
        yaxis=dict(title="Expected Points"),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)
