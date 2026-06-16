"""
BestBetWC Web UI
Ejecutar: streamlit run src/web/app.py
"""

from __future__ import annotations

import streamlit as st

from src.web.state import init_state

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

init_state()

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
    "Dashboard": "pages/1_dashboard.py",
    "Predecir Partido": "pages/2_predict.py",
    "Estrategia": "pages/3_strategy.py",
    "Simular": "pages/4_simulate.py",
    "Clasificación": "pages/5_standings.py",
    "Perfiles": "pages/6_profiles.py",
}

pg = st.navigation([st.Page(path, title=title) for title, path in pages.items()])
pg.run()
