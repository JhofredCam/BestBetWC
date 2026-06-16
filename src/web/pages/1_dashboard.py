"""
Dashboard — Partidos de hoy y mañana con sugerencias automáticas.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st
from sqlalchemy.orm import Session

from src.config import POLLA_RULES
from src.database.connection import get_session
from src.database.models import Match, Participant, Standing
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.web.natural_language import (
    explain_recommendation,
    format_empty_db_message,
    format_match_datetime,
)

st.title("⚽ Partidos de Hoy y Mañana")


@st.cache_resource
def get_components() -> dict[str, Any]:
    return {
        "model": DixonColes(max_goals=POLLA_RULES.max_goals),
        "ep_calc": ExpectedScoreCalculator(),
        "selector": StrategySelector(),
    }


def get_user_position(db: Session) -> int:
    participants = db.query(Participant).all()
    if not participants:
        return 1
    standings = (
        db.query(Standing)
        .filter(Standing.round == "overall")
        .order_by(Standing.position.asc())
        .all()
    )
    if not standings:
        return 1
    return standings[0].position if standings else 1


def get_matches_for_date(db: Session, date: datetime) -> list[Match]:
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return (
        db.query(Match)
        .filter(Match.datetime >= start, Match.datetime < end)
        .filter(Match.status.ilike("%schedul%"))
        .order_by(Match.datetime.asc())
        .all()
    )


def render_match_card(
    match: Match,
    model: DixonColes,
    ep_calc: ExpectedScoreCalculator,
    selector: StrategySelector,
    position: int,
) -> None:
    home_name = match.home_team.name if match.home_team else f"Equipo {match.home_team_id}"
    away_name = match.away_team.name if match.away_team else f"Equipo {match.away_team_id}"

    time_str = format_match_datetime(match.datetime) if match.datetime else "—"
    venue_str = match.venue or "—"
    round_str = match.round or "—"

    try:
        pred = model.predict_match(home_name, away_name)
        rec = selector.get_recommendation(pred, position, POLLA_RULES.num_participants)
        top_result = rec.prediction
        score = f"{top_result.home_goals}-{top_result.away_goals}"
        explanation = explain_recommendation(
            top_result.home_goals,
            top_result.away_goals,
            top_result.prob_exact,
            top_result.ownership_estimate,
        )
    except Exception:
        score = "—"
        explanation = "No se pudo calcular el pronóstico."

    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader(f"{home_name} vs {away_name}")
            st.caption(f"⏰ {time_str}  |  📍 {venue_str}  |  🏆 {round_str}")

        with cols[1]:
            st.metric("📊 Sugerencia", score)

        st.caption(f"💡 {explanation}")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("📋 Ver análisis completo", key=f"analyze_{match.id}"):
                st.session_state["analyze_match_id"] = match.id
                st.switch_page("pages/2_predict.py")
        with btn_col2:
            if st.button("📝 Simular pronóstico", key=f"sim_{match.id}"):
                st.session_state["sim_match_id"] = match.id
                st.switch_page("pages/4_simulate.py")


# ── Main ────────────────────────────────────────────────────────────

db = get_session()
try:
    now = datetime.now(UTC)
    today_matches = get_matches_for_date(db, now)
    tomorrow = now + timedelta(days=1)
    tomorrow_matches = get_matches_for_date(db, tomorrow)

    all_matches = today_matches + tomorrow_matches

    if not all_matches:
        st.warning(format_empty_db_message())
        st.info(
            "También podés agregar participantes y posiciones en la página "
            "**Tabla de Posiciones** para empezar."
        )
    else:
        comps = get_components()
        position = get_user_position(db)
        total_participants = db.query(Participant).count()

        if total_participants > 0:
            st.caption(
                f"Posición actual: **{position}° de {total_participants}** participantes"
            )

        if today_matches:
            st.subheader(f"📅 Hoy — {now.day} de {now.strftime('%B')}")
            for m in today_matches:
                render_match_card(
                    m, comps["model"], comps["ep_calc"], comps["selector"], position
                )

        if tomorrow_matches:
            st.subheader(f"📅 Mañana — {tomorrow.day} de {tomorrow.strftime('%B')}")
            for m in tomorrow_matches:
                render_match_card(
                    m, comps["model"], comps["ep_calc"], comps["selector"], position
                )

finally:
    db.close()
