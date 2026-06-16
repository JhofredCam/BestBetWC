"""
BestBetWC Web UI
Ejecutar: streamlit run src/web/app.py
"""

from __future__ import annotations

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
        format_func=lambda p: (
            f"{p}° {'🥇' if p == 1 else '🥈' if p == 2 else '🥉' if p == 3 else ''}"
        ),
    )
    st.session_state["position"] = position

    st.divider()

    total_participants = st.number_input(
        "Participantes", min_value=2, max_value=100, value=15,
    )
    st.session_state["total_participants"] = total_participants

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
