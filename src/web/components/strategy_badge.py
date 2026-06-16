"""
Badge de modo de estrategia activa.
"""

from __future__ import annotations

import streamlit as st

from src.optimization.strategy import StrategyMode

_MODE_LABELS: dict[StrategyMode, str] = {
    StrategyMode.MINIMIZE_RISK: "🛡️ Minimizar Riesgo",
    StrategyMode.BALANCED: "⚖️ Balanceado",
    StrategyMode.DIFFERENTIATION: "🎯 Diferenciación",
    StrategyMode.HIGH_RISK: "🚀 Alto Riesgo",
}

_MODE_COLORS: dict[StrategyMode, str] = {
    StrategyMode.MINIMIZE_RISK: "#2ecc71",
    StrategyMode.BALANCED: "#3498db",
    StrategyMode.DIFFERENTIATION: "#e67e22",
    StrategyMode.HIGH_RISK: "#e74c3c",
}


def render_strategy_badge(mode: StrategyMode, active: bool = True) -> None:
    label = _MODE_LABELS.get(mode, mode.value)
    color = _MODE_COLORS.get(mode, "#888")
    opacity = "FF" if active else "44"
    st.markdown(
        f'<span style="background-color:{color}{opacity};padding:4px 10px;'
        f'border-radius:6px;font-weight:bold;color:white;">{label}</span>',
        unsafe_allow_html=True,
    )
