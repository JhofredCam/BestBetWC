"""
Card de perfil de jugador con radar chart.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_profile_card(
    name: str,
    archetype: str,
    result_accuracy: str,
    exact_accuracy: str,
    avg_points: str,
    radar_values: list[float],
    color: str = "#27ae60",
) -> None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(name)
        st.metric("Arquetipo", archetype)
        st.metric("Result Accuracy", result_accuracy)
        st.metric("Exact Accuracy", exact_accuracy)
        st.metric("Avg. Points/Match", avg_points)

    with col2:
        categories = [
            "Conservador", "Agresivo", "Market\nFollower",
            "Intuición", "Home Bias", "Draw\nAversion",
        ]
        values = radar_values + [radar_values[0]]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            marker=dict(color=color),
        ))
        fig.update_layout(height=350, polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)
