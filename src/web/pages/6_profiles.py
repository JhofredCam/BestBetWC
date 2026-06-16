"""
Perfiles de Participantes — datos reales de BD con fallback a demo.
"""

from __future__ import annotations

from typing import cast

import plotly.graph_objects as go
import streamlit as st

from src.database.connection import get_session
from src.game_theory.profiling import PlayerProfiler

st.title("👤 Perfiles de Participantes")

db = get_session()
try:
    profiler = PlayerProfiler(db)

    try:
        profiles = profiler.profile_all()
    except Exception:
        profiles = {}

    if profiles:
        st.success("Datos cargados desde la base de datos.")
        tabs = st.tabs([profile.name for pid, profile in profiles.items()])

        for i, (pid, profile) in enumerate(profiles.items()):
            with tabs[i]:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader(profile.name)
                    st.metric(
                        "Arquetipo",
                        profile.dominant_archetype.value.replace("_", " ").title(),
                    )
                    st.metric(
                        "Precisión de resultado",
                        f"{profile.result_accuracy:.0%}",
                    )
                    st.metric(
                        "Precisión exacta",
                        f"{profile.exact_accuracy:.0%}",
                    )
                    st.metric(
                        "Puntos por partido",
                        f"{profile.avg_points_per_match:.1f}",
                    )
                    st.metric("Total pronósticos", str(profile.total_predictions))

                with col2:
                    categories = [
                        "Conservador", "Agresivo", "Seguidor<br>del mercado",
                        "Intuición", "Sesgo<br>favorito", "Sesgo<br>local",
                    ]
                    values = [
                        profile.conservative_score,
                        profile.aggressive_score,
                        profile.market_follower_score,
                        profile.intuition_score,
                        profile.favorite_bias,
                        profile.home_bias,
                    ]

                    fig = go.Figure(data=go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        marker=dict(color="#27ae60"),
                    ))
                    fig.update_layout(
                        height=350,
                        polar=dict(radialaxis=dict(range=[0, 1])),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"profile_radar_{pid}")

    else:
        st.warning("⚠️ Datos de ejemplo — no hay participantes en la base de datos")

        tabs = st.tabs(["Jugador 1", "Jugador 2", "Jugador 3"])

        demo_data: list[dict[str, object]] = [
            {
                "name": "Carlos", "archetype": "Conservador",
                "result_acc": 0.62, "exact_acc": 0.09, "avg_pts": 2.4,
                "values": [0.7, 0.2, 0.5, 0.3, 0.6, 0.4],
            },
            {
                "name": "María", "archetype": "Agresivo",
                "result_acc": 0.51, "exact_acc": 0.12, "avg_pts": 2.7,
                "values": [0.2, 0.8, 0.3, 0.4, 0.5, 0.3],
            },
            {
                "name": "Esteban", "archetype": "Seguidor del Mercado",
                "result_acc": 0.57, "exact_acc": 0.06, "avg_pts": 2.1,
                "values": [0.4, 0.3, 0.8, 0.2, 0.4, 0.5],
            },
        ]

        for i, data in enumerate(demo_data):
            with tabs[i]:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader(data["name"])
                    st.metric("Arquetipo", str(data["archetype"]))
                    st.metric(
                        "Resultados acertados",
                        f"{float(cast(float, data['result_acc'])):.0%}",
                    )
                    st.metric(
                        "Exactos acertados",
                        f"{float(cast(float, data['exact_acc'])):.0%}",
                    )
                    st.metric(
                        "Promedio pts/partido",
                        f"{float(cast(float, data['avg_pts'])):.1f}",
                    )

                with col2:
                    categories = [
                        "Conservador", "Agresivo", "Seguidor<br>del mercado",
                        "Intuición", "Sesgo<br>favorito", "Sesgo<br>local",
                    ]
                    demo_values_raw = cast(list[float], data["values"])
                    demo_values: list[float] = list(demo_values_raw)

                    fig = go.Figure(data=go.Scatterpolar(
                        r=demo_values + [demo_values[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        marker=dict(color="#27ae60"),
                    ))
                    fig.update_layout(
                        height=350,
                        polar=dict(radialaxis=dict(range=[0, 1])),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"demo_profile_{i}")

finally:
    db.close()
