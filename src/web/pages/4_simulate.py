from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.config import POLLA_RULES
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator

st.title("Simulador Monte Carlo")

col1, col2, col3 = st.columns(3)
with col1:
    home_lambda = st.slider("λ Local", 0.1, 5.0, 1.5, 0.1)
with col2:
    away_lambda = st.slider("μ Visitante", 0.1, 5.0, 1.0, 0.1)
with col3:
    n_simulations = st.select_slider(
        "Simulaciones", options=[100, 1000, 5000, 10000, 50000], value=10000,
    )

if st.button("🎲 Ejecutar Simulación", type="primary", use_container_width=True):
    model = DixonColes(max_goals=POLLA_RULES.max_goals)
    prediction = model.predict_from_params(home_lambda, away_lambda)

    with st.spinner(f"Ejecutando {n_simulations:,} simulaciones..."):
        home_goals_sim = np.random.choice(
            len(prediction.home_goals_dist),
            size=n_simulations, p=prediction.home_goals_dist,
        )
        away_goals_sim = np.random.choice(
            len(prediction.away_goals_dist),
            size=n_simulations, p=prediction.away_goals_dist,
        )

        ep_calc = ExpectedScoreCalculator()
        top5 = ep_calc.rank_all_predictions(prediction)[:5]

        results = []
        for r in top5:
            eps = np.zeros(n_simulations)
            for s in range(n_simulations):
                h = int(home_goals_sim[s])
                a = int(away_goals_sim[s])
                if h == r.home_goals and a == r.away_goals:
                    eps[s] = float(POLLA_RULES.exact_score_pts)
                    eps[s] += float(POLLA_RULES.goals_home_correct_pts)
                    eps[s] += float(POLLA_RULES.goals_away_correct_pts)
                else:
                    if h == r.home_goals:
                        eps[s] += float(POLLA_RULES.goals_home_correct_pts)
                    if a == r.away_goals:
                        eps[s] += float(POLLA_RULES.goals_away_correct_pts)
                    if ((r.home_goals > r.away_goals and h > a)
                        or (r.home_goals == r.away_goals and h == a)
                        or (r.home_goals < r.away_goals and h < a)):
                        eps[s] += float(POLLA_RULES.result_correct_pts)

            results.append({
                "marcador": f"{r.home_goals}-{r.away_goals}",
                "ep_mean": float(eps.mean()),
                "ep_std": float(eps.std()),
                "ep_min": float(eps.min()),
                "ep_max": float(eps.max()),
            })

    # ── Gráfico ─────────────────────────────────────────────────────

    st.subheader(f"Resultados — {n_simulations:,} iteraciones")

    scores = [r["marcador"] for r in results]
    means = [r["ep_mean"] for r in results]
    stds = [r["ep_std"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=["#27ae60", "#2980b9", "#8e44ad", "#e67e22", "#e74c3c"],
        text=[f"{m:.2f} ± {s:.2f}" for m, s in zip(means, stds)],
        textposition="outside",
    ))
    fig.update_layout(
        title="Expected Score Promedio ± 1σ",
        yaxis=dict(title="Expected Points"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de métricas ───────────────────────────────────────────

    st.dataframe(
        [{
            "Marcador": r["marcador"],
            "EP Medio": f"{r['ep_mean']:.2f}",
            "EP Desv": f"{r['ep_std']:.2f}",
            "EP Mín": f"{r['ep_min']:.0f}",
            "EP Máx": f"{r['ep_max']:.0f}",
        } for r in results],
        hide_index=True,
        use_container_width=True,
    )
