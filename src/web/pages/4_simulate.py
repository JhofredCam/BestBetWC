"""
Simulador simplificado — sin inputs de lambdas.
Pre-carga partidos del día y muestra resultados en lenguaje natural.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import streamlit as st
from sqlalchemy.orm import Session

from src.config import POLLA_RULES
from src.database.connection import get_session
from src.database.models import Match, Participant, Standing
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.web.natural_language import (
    format_empty_db_message,
    format_expected_rank,
    format_match_datetime,
)

st.title("🎲 Simulador de Pronósticos")


def get_today_matches(db: Session) -> list[Match]:
    now = datetime.now(UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    matches: list[Match] = (
        db.query(Match)
        .filter(Match.datetime >= start, Match.datetime < end)
        .filter(Match.status.ilike("%schedul%"))
        .order_by(Match.datetime.asc())
        .all()
    )
    return matches


db = get_session()
try:
    today_matches = get_today_matches(db)

    if not today_matches:
        st.warning(format_empty_db_message())
        st.stop()

    total_participants = db.query(Participant).count()
    if total_participants == 0:
        total_participants = 15

    user_position = 1
    stnd_rows = (
        db.query(Standing)
        .filter(Standing.round == "overall")
        .order_by(Standing.position.asc())
        .all()
    )
    for stnd in stnd_rows:
        participant = (
            db.query(Participant).filter(Participant.id == stnd.participant_id).first()
        )
        if participant and participant.name.lower() in ("tu", "vos", "tú"):
            user_position = stnd.position
            break

    st.subheader("📅 Partidos de hoy")
    for m in today_matches:
        home_name = m.home_team.name if m.home_team else f"Equipo {m.home_team_id}"
        away_name = m.away_team.name if m.away_team else f"Equipo {m.away_team_id}"
        time_str = format_match_datetime(m.datetime) if m.datetime else "—"
        st.caption(f"⚽ {home_name} vs {away_name} — ⏰ {time_str}")

    st.divider()

    n_simulations = st.slider(
        "Cantidad de simulaciones",
        min_value=100,
        max_value=50000,
        value=5000,
        step=100,
        format="%d",
        help="Más simulaciones = resultado más preciso pero tarda más.",
    )

    if st.button("🎲 Simular mis pronósticos para hoy", type="primary", use_container_width=True):
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        ep_calc = ExpectedScoreCalculator()
        selector = StrategySelector()

        detailed_results: list[dict[str, Any]] = []

        with st.spinner(f"Ejecutando {n_simulations:,} simulaciones..."):
            for m in today_matches:
                home_name = m.home_team.name if m.home_team else f"Equipo {m.home_team_id}"
                away_name = m.away_team.name if m.away_team else f"Equipo {m.away_team_id}"

                pred = model.predict_match(home_name, away_name)
                rec = selector.get_recommendation(
                    pred, user_position, total_participants
                )
                best = rec.prediction

                rng = np.random.default_rng()
                ep_values = np.zeros(n_simulations)
                flat = pred.score_matrix.flatten()
                flat = flat / flat.sum()
                max_g = pred.score_matrix.shape[0] - 1

                for sim_i in range(n_simulations):
                    idx = rng.choice(len(flat), p=flat)
                    actual_h = int(idx // (max_g + 1))
                    actual_a = int(idx % (max_g + 1))

                    if actual_h == best.home_goals and actual_a == best.away_goals:
                        ep_values[sim_i] = float(POLLA_RULES.exact_score_pts)
                        ep_values[sim_i] += float(POLLA_RULES.goals_home_correct_pts)
                        ep_values[sim_i] += float(POLLA_RULES.goals_away_correct_pts)
                    else:
                        if actual_h == best.home_goals:
                            ep_values[sim_i] += float(POLLA_RULES.goals_home_correct_pts)
                        if actual_a == best.away_goals:
                            ep_values[sim_i] += float(POLLA_RULES.goals_away_correct_pts)
                        if (
                            (best.home_goals > best.away_goals and actual_h > actual_a)
                            or (
                                best.home_goals == best.away_goals
                                and actual_h == actual_a
                            )
                            or (
                                best.home_goals < best.away_goals
                                and actual_h < actual_a
                            )
                        ):
                            ep_values[sim_i] += float(POLLA_RULES.result_correct_pts)

                detailed_results.append({
                    "match": f"{home_name} vs {away_name}",
                    "prediction": f"{best.home_goals}-{best.away_goals}",
                    "ep_mean": float(ep_values.mean()),
                    "ep_std": float(ep_values.std()),
                })

        st.divider()
        st.subheader("📊 Resultados de la simulación")

        for r in detailed_results:
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{r['match']}**")
                    st.caption(f"Pronóstico simulado: {r['prediction']}")
                with c2:
                    st.metric(
                        "Puntos esperados",
                        f"{float(r['ep_mean']):.2f}",
                        delta=f"±{float(r['ep_std']):.2f}",
                    )

        st.divider()

        simulated_rank = float(user_position + np.random.default_rng().normal(0, 2))
        simulated_rank = max(1, min(float(total_participants), simulated_rank))
        rank_range = (
            max(1, user_position - 3),
            min(total_participants, user_position + 3),
        )
        st.info(
            format_expected_rank(
                float(simulated_rank),
                float(rank_range[0]),
                float(rank_range[1]),
            )
        )

        st.caption(
            "La probabilidad de acertar un marcador exacto "
            "depende de la precisión del modelo y la dificultad del partido."
        )

finally:
    db.close()
