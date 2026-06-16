"""
Tabla de Posiciones editable con st.data_editor.
Permite CRUD de participantes y recalcula estrategia automáticamente.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from sqlalchemy.orm import Session

from src.database.connection import get_session
from src.database.models import Participant, Standing
from src.optimization.strategy import StrategySelector
from src.web.natural_language import strategy_advice

st.title("📊 Tabla de Posiciones")

st.markdown(
    "Editá los nombres y puntos de los participantes. "
    "El sistema recalcula automáticamente tu estrategia."
)


def load_standings(db: Session) -> list[dict[str, Any]]:
    participants = db.query(Participant).all()
    rows: list[dict[str, Any]] = []
    for p in participants:
        standing = (
            db.query(Standing)
            .filter(Standing.participant_id == p.id)
            .filter(Standing.round == "overall")
            .first()
        )
        rows.append({
            "Pos": standing.position if standing else 0,
            "id": p.id,
            "Nombre": p.name,
            "Puntos": standing.total_points if standing else 0,
            "SosVos": True,
        })
    rows.sort(key=lambda r: r["Puntos"], reverse=True)
    for i, r in enumerate(rows):
        r["Pos"] = i + 1
    return rows


def apply_changes(
    db: Session,
    new_name: str,
    changed_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
) -> None:
    original_ids = {r["id"] for r in original_rows}
    orig_by_id = {r["id"]: r for r in original_rows}

    for row in changed_rows:
        pid = row["id"]
        if pid not in original_ids:
            new_participant = Participant(name=row["Nombre"])
            db.add(new_participant)
            db.flush()
            new_standing = Standing(
                participant_id=new_participant.id,
                round="overall",
                total_points=row["Puntos"],
                position=row["Pos"],
            )
            db.add(new_standing)
        else:
            orig = orig_by_id[pid]
            if orig["Nombre"] != row["Nombre"] or orig["Puntos"] != row["Puntos"]:
                existing = db.query(Participant).filter(Participant.id == pid).first()
                if existing:
                    existing.name = row["Nombre"]
                standing = (
                    db.query(Standing)
                    .filter(Standing.participant_id == pid)
                    .filter(Standing.round == "overall")
                    .first()
                )
                if standing:
                    standing.total_points = row["Puntos"]
                    standing.position = row["Pos"]
                elif row["Puntos"] > 0:
                    standing = Standing(
                        participant_id=pid,
                        round="overall",
                        total_points=row["Puntos"],
                        position=row["Pos"],
                    )
                    db.add(standing)

    for pid in original_ids - {r["id"] for r in changed_rows}:
        db.query(Standing).filter(Standing.participant_id == pid).delete()
        db.query(Participant).filter(Participant.id == pid).delete()

    if new_name.strip():
        p = Participant(name=new_name.strip())
        db.add(p)
        db.flush()
        new_pos = db.query(Participant).count()
        st = Standing(
            participant_id=p.id,
            round="overall",
            total_points=0,
            position=new_pos,
        )
        db.add(st)

    db.commit()


def render() -> None:
    db = get_session()
    try:
        current_rows = load_standings(db)

        if not current_rows:
            st.info(
                "No hay participantes cargados todavía. "
                "Agregá el primero usando el botón **Agregar participante** abajo."
            )

        edited_df = st.data_editor(
            current_rows,
            column_config={
                "Pos": st.column_config.NumberColumn("#", width="small", disabled=True),
                "id": None,
                "Nombre": st.column_config.TextColumn("Nombre", width="medium"),
                "Puntos": st.column_config.NumberColumn("Puntos", format="%d", width="small"),
                "SosVos": st.column_config.CheckboxColumn(
                    "¿Sos vos?",
                    width="small",
                    help="Marcá esta casilla para identificarte",
                ),
            },
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="standings_editor",
        )

        new_name = st.text_input(
            "Nombre del nuevo participante",
            key="new_participant_name",
            placeholder="Ej: Juan Pérez",
        )
        if st.button("➕ Agregar participante"):
            if new_name.strip():
                apply_changes(db, new_name.strip(), [], [])
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Escribí un nombre para agregar.")

        if st.button("💾 Guardar cambios", type="primary"):
            apply_changes(
                db,
                new_name,
                [row for row in edited_df if row.get("id")],
                current_rows,
            )
            st.cache_data.clear()
            st.success("✅ Cambios guardados.")
            st.rerun()

        st.divider()

        st.subheader("🎯 Tu estrategia actual")

        user_rows = [r for r in current_rows if r.get("SosVos")]
        if user_rows:
            user_row = user_rows[0]
            user_pos = user_row["Pos"]
            total = len(current_rows)
            selector = StrategySelector()
            mode = selector.determine_mode(user_pos, max(total, 15))
            advice = strategy_advice(mode.value, user_pos)
            st.info(advice)
            st.caption(
                f"Tu posición: **{user_pos}° de {max(total, 15)}** — "
                f"Modo: **{mode.value.replace('_', ' ').title()}**"
            )
        else:
            st.info(
                "Marcá la casilla **¿Sos vos?** en un participante para ver "
                "recomendaciones de estrategia personalizadas."
            )

    finally:
        db.close()


render()
