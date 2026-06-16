"""
Clasificación de la Polla — Vista de solo lectura.
Para editar, usá la página Tabla de Posiciones.
"""

from __future__ import annotations

import streamlit as st

from src.database.connection import get_session
from src.database.models import Participant, Standing

st.title("🏆 Clasificación de la Polla")

db = get_session()
try:
    standings = (
        db.query(Standing)
        .filter(Standing.round == "overall")
        .order_by(Standing.position.asc())
        .all()
    )

    if not standings:
        st.info(
            "No hay datos de posiciones cargados. "
            "Andá a **Tabla de Posiciones** para agregar participantes."
        )
    else:
        rows = []
        for s in standings:
            participant = (
                db.query(Participant)
                .filter(Participant.id == s.participant_id)
                .first()
            )
            name = participant.name if participant else f"Participante {s.participant_id}"
            rows.append({
                "#": s.position,
                "Participante": name,
                "Puntos": s.total_points,
            })

        st.dataframe(
            rows,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "Participante": st.column_config.TextColumn("Participante"),
                "Puntos": st.column_config.NumberColumn("Puntos", format="%d"),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Para editar la tabla, andá a la página **Tabla de Posiciones**."
        )

finally:
    db.close()
