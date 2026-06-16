from __future__ import annotations

import streamlit as st

st.title("Clasificación de la Polla")

st.dataframe(
    [
        {"#": 1, "Participante": "Carlos", "Pts": 156, "Exactos": 5, "Resultados": 32,
         "Última fecha": "+12 pts"},
        {"#": 2, "Participante": "María", "Pts": 148, "Exactos": 4, "Resultados": 30,
         "Última fecha": "+8 pts"},
        {"#": 3, "Participante": "⭐ Tú", "Pts": 142, "Exactos": 3, "Resultados": 31,
         "Última fecha": "+10 pts"},
        {"#": 4, "Participante": "Juan", "Pts": 138, "Exactos": 2, "Resultados": 29,
         "Última fecha": "+6 pts"},
        {"#": 5, "Participante": "Ana", "Pts": 135, "Exactos": 3, "Resultados": 28,
         "Última fecha": "+14 pts"},
    ],
    column_config={
        "#": st.column_config.NumberColumn("#", width="small"),
        "Participante": st.column_config.TextColumn("Participante"),
        "Pts": st.column_config.NumberColumn("Puntos", format="%d"),
        "Exactos": st.column_config.NumberColumn("Exactos", width="small"),
        "Resultados": st.column_config.NumberColumn("Resultados", width="small"),
        "Última fecha": st.column_config.TextColumn("Última fecha"),
    },
    hide_index=True,
    use_container_width=True,
)
