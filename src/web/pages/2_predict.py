from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import POLLA_RULES
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector

st.title("Predecir Partido")

# ── Formulario ─────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.text_input("Equipo Local", "Brasil")
    home_lambda = st.slider("Goles esperados Local (λ)", 0.1, 5.0, 1.8, 0.1)
with col2:
    away_team = st.text_input("Equipo Visitante", "Argentina")
    away_lambda = st.slider("Goles esperados Visitante (μ)", 0.1, 5.0, 1.2, 0.1)
with col3:
    position = st.selectbox(
        "Tu Posición",
        options=list(range(1, 16)),
        index=2,
        format_func=lambda p: f"{p}° {'🥇 Líder' if p == 1 else '🥈' if p == 2 else ''}",
    )

if st.button("⚽ Calcular Pronóstico Óptimo", type="primary", use_container_width=True):

    model = DixonColes(max_goals=POLLA_RULES.max_goals)
    prediction = model.predict_from_params(home_lambda, away_lambda)
    ep_calc = ExpectedScoreCalculator()
    ranked = ep_calc.rank_all_predictions(prediction)

    selector = StrategySelector()
    recommendation = selector.get_recommendation(
        prediction, position, POLLA_RULES.num_participants,
    )

    # ── Resultados ─────────────────────────────────────────────────

    st.divider()
    st.subheader(f"{home_team} vs {away_team}")

    st.info(
        f"**Estrategia ({position}°): {recommendation.strategy_mode.value}**  \n"
        f"Recomendado: **{recommendation.prediction.home_goals}-"
        f"{recommendation.prediction.away_goals}**  |  "
        f"EP: {recommendation.prediction.ep_total:.2f} pts  |  "
        f"Risk: {recommendation.risk_score:.0%}",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        z = prediction.score_matrix * 100
        fig = px.imshow(
            z,
            text_auto=".1f",
            labels=dict(x=f"Goles {away_team}", y=f"Goles {home_team}"),
            color_continuous_scale="Greens",
            zmin=0,
            zmax=float(z.max()),
        )
        fig.update_layout(
            title="Probabilidades de Marcador (%)",
            height=400,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top10 = ranked[:10]
        table_data = []
        for i, r in enumerate(top10):
            icon = "★ " if i == 0 else ""
            table_data.append({
                "#": i + 1,
                "Marcador": f"{icon}{r.home_goals}-{r.away_goals}",
                "EP Total": f"{r.ep_total:.2f}",
                "P(Exacto)": f"{r.prob_exact:.1%}",
                "P(Result)": f"{r.prob_result:.1%}",
                "P(Goles)": f"{r.prob_goals_home + r.prob_goals_away:.1%}",
            })

        st.dataframe(
            table_data,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "Marcador": st.column_config.TextColumn("Marcador"),
                "EP Total": st.column_config.TextColumn("EP Total"),
                "P(Exacto)": st.column_config.TextColumn("P(Exacto)"),
                "P(Result)": st.column_config.TextColumn("P(Result)"),
                "P(Goles)": st.column_config.TextColumn("P(Goles)"),
            },
            hide_index=True,
            use_container_width=True,
        )

    # ── Gráfico de EP apilado ───────────────────────────────────────

    st.subheader("Componentes del Expected Score — Top 10")

    scores = [f"{r.home_goals}-{r.away_goals}" for r in top10]
    fig = go.Figure(data=[
        go.Bar(name="Exacto", x=scores, y=[r.ep_exact for r in top10],
               marker_color="#27ae60"),
        go.Bar(name="Resultado", x=scores, y=[r.ep_result for r in top10],
               marker_color="#2980b9"),
        go.Bar(name="Goles", x=scores,
               y=[r.ep_goals_home + r.ep_goals_away for r in top10],
               marker_color="#8e44ad"),
        go.Bar(name="Único", x=scores, y=[r.ep_unique for r in top10],
               marker_color="#e67e22"),
    ])
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="Expected Points"),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
