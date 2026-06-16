from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.optimization.strategy import StrategyMode, StrategySelector

st.title("Estrategia Adaptativa")

st.markdown("""
El sistema ajusta automáticamente tu estrategia según tu posición en la tabla.
La **estrategia óptima** no es la misma si vas liderando que si vas último:
quien lidera debe minimizar riesgo; quien persigue debe diferenciarse.
""")

# ── Simulador de posición ──────────────────────────────────────────

position = st.slider(
    "Simula tu posición en la tabla",
    min_value=1, max_value=15, value=3,
    format="%d°",
)

selector = StrategySelector()
mode = selector.determine_mode(position, 15)

# ── Modos de estrategia ────────────────────────────────────────────

modes_info = {
    StrategyMode.MINIMIZE_RISK: {
        "title": "🛡️ Minimizar Riesgo",
        "description": "Eliges marcadores de alta probabilidad. Evitas diferenciación innecesaria.",
        "color": "#2ecc71",
        "risk": "Bajo",
        "position": "1°",
    },
    StrategyMode.BALANCED: {
        "title": "⚖️ Balanceado",
        "description": "Mezclas predicciones seguras con algunas apuestas diferenciadas.",
        "color": "#3498db",
        "risk": "Medio",
        "position": "2°-5°",
    },
    StrategyMode.DIFFERENTIATION: {
        "title": "🎯 Diferenciación",
        "description": "Buscas marcadores poco populares con alta probabilidad real de ocurrir.",
        "color": "#e67e22",
        "risk": "Medio-Alto",
        "position": "6°-10°",
    },
    StrategyMode.HIGH_RISK: {
        "title": "🚀 Alto Riesgo",
        "description": "Apuestas agresivas: máximo upside, aceptando alta varianza.",
        "color": "#e74c3c",
        "risk": "Alto",
        "position": "11°-15°",
    },
}

cols = st.columns(4)
for i, (mode_enum, info) in enumerate(modes_info.items()):
    with cols[i]:
        is_active = mode_enum == mode
        st.markdown(f"""
        <div style="border: 2px solid {info['color']};
                    border-radius: 10px; padding: 15px;
                    {'background-color: rgba(255,255,255,0.05)' if not is_active
                     else 'background-color: ' + info['color'] + '22'};
                    {'box-shadow: 0 0 15px ' + info['color'] + '44' if is_active else ''}">
            <h4>{info['title']}</h4>
            <p style="font-size: 0.85em; color: #888;">{info['description']}</p>
            <p style="font-size: 0.8em;">Posición: <strong>{info['position']}</strong></p>
            <p style="font-size: 0.8em;">Riesgo: <strong>{info['risk']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ── Visualización de rangos ────────────────────────────────────────

st.subheader("Mapa de Estrategias")

fig = go.Figure()
ranges = [
    (1, 1, "Minimizar Riesgo", "#2ecc71"),
    (2, 5, "Balanceado", "#3498db"),
    (6, 10, "Diferenciación", "#e67e22"),
    (11, 15, "Alto Riesgo", "#e74c3c"),
]
for start, end, label, color in ranges:
    fig.add_trace(go.Bar(
        y=[label], x=[end - start + 1],
        orientation="h", marker_color=color,
        opacity=0.7 if (start <= position <= end) else 0.3,
        text=f"Puestos {start}-{end}",
        textposition="inside",
    ))

fig.update_layout(
    xaxis=dict(title="Número de posiciones en este rango"),
    showlegend=False,
    height=200,
)
st.plotly_chart(fig, use_container_width=True)

# ── Tu modo actual ─────────────────────────────────────────────────

active = modes_info[mode]
st.success(
    f"**Tu estrategia actual ({position}°): {active['title']}**\n\n"
    f"Riesgo: {active['risk']} · {active['description']}"
)
