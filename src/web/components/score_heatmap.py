"""
Heatmap del score matrix usando Plotly.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px


def render_score_heatmap(
    score_matrix: np.ndarray,
    home_label: str = "Goles Local",
    away_label: str = "Goles Visitante",
    title: str = "Probabilidades de Marcador (%)",
    height: int = 400,
) -> None:
    import streamlit as st

    z = score_matrix * 100
    fig = px.imshow(
        z,
        text_auto=".1f",
        labels=dict(x=away_label, y=home_label),
        color_continuous_scale="Greens",
        zmin=0,
        zmax=float(z.max()),
    )
    fig.update_layout(
        title=title,
        height=height,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
