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
    initial_sidebar_state="collapsed",
)

st.logo(
    "https://img.icons8.com/color/48/world-cup.png",
    size="large",
)

# ── Sidebar mínimo ──────────────────────────────────────────────────

with st.sidebar:
    st.title("⚽ BestBetWC")
    st.caption("Mundial 2026")

    st.divider()

    if st.button("🔄 Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("v0.3.0 — UX Simplificada")

# ── Navegación ─────────────────────────────────────────────────────

pages = {
    "Dashboard": "pages/1_dashboard.py",
    "Análisis de Partido": "pages/2_predict.py",
    "Tabla de Posiciones": "pages/3_strategy.py",
    "Simular": "pages/4_simulate.py",
    "Clasificación": "pages/5_standings.py",
    "Perfiles": "pages/6_profiles.py",
}

pg = st.navigation([st.Page(path, title=title) for title, path in pages.items()])
pg.run()
