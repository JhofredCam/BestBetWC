from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

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
            categories = [
                "Conservador", "Agresivo", "Market\nFollower",
                "Intuición", "Home Bias", "Draw\nAversion",
            ]
            values = [0.7, 0.2, 0.5, 0.3, 0.6, 0.4]

            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                marker=dict(color="#27ae60"),
            ))
            fig.update_layout(height=350, polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True, key=f"profile_radar_{i}")
