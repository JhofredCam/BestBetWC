"""
Match Detail — Predicción completa sin inputs técnicos.
Toma match_id como query param o de session_state.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.config import POLLA_RULES
from src.database.connection import get_session
from src.database.models import Match, Participant, Standing
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.web.natural_language import (
    explain_recommendation,
    format_empty_db_message,
    format_match_context,
    format_match_datetime,
    format_percentage,
    strategy_advice,
)

# ── Obtener match_id ────────────────────────────────────────────────

match_id = None
if "analyze_match_id" in st.session_state:
    match_id = st.session_state.pop("analyze_match_id")
elif "match_id" in st.query_params:
    try:
        match_id = int(st.query_params["match_id"])
    except (ValueError, TypeError):
        pass

if match_id is None:
    st.title("📊 Análisis de Partido")
    st.warning(
        "Esta página recibe un `match_id`. "
        "Volvé al **Dashboard** y hacé clic en **Ver análisis completo** "
        "junto al partido que querés analizar."
    )
    st.info(
        "También podés pasar `?match_id=1` en la URL de esta página (reemplazá `1` por el ID real)."
    )
    st.stop()

# ── Cargar datos del partido ────────────────────────────────────────

db = get_session()
try:
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        st.error(f"Partido #{match_id} no encontrado en la base de datos.")
        st.info(format_empty_db_message())
        st.stop()

    home_name = match.home_team.name if match.home_team else f"Equipo {match.home_team_id}"
    away_name = match.away_team.name if match.away_team else f"Equipo {match.away_team_id}"

    st.title(f"{home_name} vs {away_name}")

    col1, col2 = st.columns(2)
    with col1:
        if match.datetime:
            st.metric("Fecha y Hora", format_match_datetime(match.datetime))
    with col2:
        st.metric("Lugar", match.venue or "Sin definir")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Ronda", match.round or "—")
    with col4:
        if match.group:
            st.metric("Grupo", match.group)

    # ── Contexto del partido ─────────────────────────────────────────

    with st.expander("📋 Contexto del partido", expanded=False):
        context = format_match_context(match)
        st.write(context)

    # ── Ejecutar predicción ──────────────────────────────────────────

    with st.spinner("Calculando predicciones..."):
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        ep_calc = ExpectedScoreCalculator()
        selector = StrategySelector()

        pred = model.predict_match(home_name, away_name)

        standings = (
            db.query(Standing)
            .filter(Standing.round == "overall")
            .order_by(Standing.position.asc())
            .all()
        )
        total_participants = db.query(Participant).count()
        position = standings[0].position if standings else 1
        if total_participants == 0:
            total_participants = 15

        rec = selector.get_recommendation(pred, position, total_participants)
        best = rec.prediction

    # ── Recomendación principal ──────────────────────────────────────

    st.divider()
    st.subheader("🎯 Recomendación Principal")

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.markdown(f"## 📊 Te sugerimos **{best.home_goals}-{best.away_goals}**")
        explanation = explain_recommendation(
            best.home_goals, best.away_goals,
            best.prob_exact, best.ownership_estimate,
        )
        st.info(explanation)
        st.caption(
            f"Estrategia: **{rec.strategy_mode.value.replace('_', ' ').title()}**"
        )
    with col_r2:
        st.metric("Probabilidad exacta", format_percentage(best.prob_exact))
        st.metric("Probabilidad resultado", format_percentage(best.prob_result))
        st.metric("≈ Popularidad", format_percentage(best.ownership_estimate))

    advice = strategy_advice(rec.strategy_mode.value, position)
    st.success(advice)

    # ── Alternativas ─────────────────────────────────────────────────

    st.subheader("🔄 Alternativas a considerar")
    ranked = ep_calc.rank_all_predictions(pred)[:5]

    for i, r in enumerate(ranked[1:4], start=2):
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**Opción {i}: {r.home_goals}-{r.away_goals}**")
                pro = f"Probabilidad: {format_percentage(r.prob_exact)} de resultado exacto"
                con = f"Popularidad estimada: {format_percentage(r.ownership_estimate)}"
                st.caption(f"✅ {pro}  |  👥 {con}")
            with c2:
                st.metric("Score", f"{r.ep_total:.2f}")

    # ── Heatmap de probabilidades ────────────────────────────────────

    st.subheader("🗺️ Mapa de Probabilidades de Marcador")
    st.caption("Verde más intenso = mayor probabilidad de ese marcador exacto")

    annot = [[f"{pred.score_matrix[i, j] * 100:.1f}%"
              for j in range(pred.score_matrix.shape[1])]
             for i in range(pred.score_matrix.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=pred.score_matrix * 100,
        x=[str(j) for j in range(pred.score_matrix.shape[1])],
        y=[str(i) for i in range(pred.score_matrix.shape[0])],
        colorscale="Greens",
        zmin=0,
        zmax=float((pred.score_matrix * 100).max()),
        text=annot,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=(
            f"Goles {home_name}: %{{y}}<br>"
            f"Goles {away_name}: %{{x}}<br>"
            "Probabilidad: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title=f"Goles {away_name}",
        yaxis_title=f"Goles {home_name}",
        height=450,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Probabilidades de resultado ──────────────────────────────────

    st.subheader("📈 Probabilidades de Resultado")
    col_result1, col_result2, col_result3 = st.columns(3)
    with col_result1:
        st.metric(
            f"🏠 Gana {home_name}",
            format_percentage(pred.home_win_prob),
            border=True,
        )
    with col_result2:
        st.metric("🤝 Empate", format_percentage(pred.draw_prob), border=True)
    with col_result3:
        st.metric(
            f"✈️ Gana {away_name}",
            format_percentage(pred.away_win_prob),
            border=True,
        )

finally:
    db.close()
