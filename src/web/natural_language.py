"""
Plain-language helpers that translate technical model outputs
into simple Spanish text. Never exposes EP, lambda, or contrarian_value.
"""

from __future__ import annotations

from datetime import datetime

from src.database.models import Match, TeamForm


def format_probability(prob: float) -> str:
    """Convert a probability like 0.082 to '8 de cada 100 veces'."""
    per_100 = round(prob * 100)
    if per_100 < 1:
        return "menos de 1 de cada 100 veces"
    if per_100 >= 100:
        return "casi siempre"
    return f"{per_100} de cada 100 veces"


def format_percentage(prob: float) -> str:
    """Convert 0.082 to '8.2%'."""
    return f"{prob * 100:.1f}%"


def explain_recommendation(
    home_goals: int,
    away_goals: int,
    prob_exact: float,
    ownership_estimate: float = 0.0,
) -> str:
    """Generate a simple Spanish explanation of why this score is recommended."""
    prob_str = format_percentage(prob_exact)
    score = f"{home_goals}-{away_goals}"

    parts = [f"Te sugerimos {score} porque tiene buena probabilidad ({prob_str})"]

    if ownership_estimate > 0 and ownership_estimate < 1:
        if ownership_estimate < 0.15:
            parts.append(
                f"y casi nadie lo elegiría (solo el {format_percentage(ownership_estimate)}), "
                f"lo que te daría puntos extra por ser único"
            )
        elif ownership_estimate < 0.30:
            parts.append(
                f"y pocos participantes lo elegirían ({format_percentage(ownership_estimate)}), "
                f"dándote ventaja"
            )
        else:
            parts.append(
                f"y una parte importante ({format_percentage(ownership_estimate)}) "
                f"también lo elegiría"
            )

    return ". ".join(parts) + "."


def format_match_context(match: Match) -> str:
    """Format recent form and context for both teams in Spanish."""
    home_name = match.home_team.name if match.home_team else f"Equipo {match.home_team_id}"
    away_name = match.away_team.name if match.away_team else f"Equipo {match.away_team_id}"

    lines: list[str] = []

    home_forms = [f for f in match.team_form if f.is_home] if match.team_form else []
    away_forms = [f for f in match.team_form if not f.is_home] if match.team_form else []

    if home_forms:
        recent = _format_recent_form(home_name, home_forms)
        if recent:
            lines.append(recent)

    if away_forms:
        recent = _format_recent_form(away_name, away_forms)
        if recent:
            lines.append(recent)

    if match.venue:
        lines.append(f"Se juega en {match.venue}.")
    if match.round:
        lines.append(f"Partido de {_translate_round(match.round)}.")
    if match.group:
        lines.append(f"Grupo {match.group}.")

    return " ".join(lines) if lines else "Sin datos de forma reciente disponibles."


def _format_recent_form(team_name: str, form_entries: list[TeamForm]) -> str:
    """Format a single team's recent form into a sentence."""
    last = form_entries[0]
    if last.result == "W":
        return f"{team_name} viene de ganar {last.goals_scored}-{last.goals_conceded}."
    elif last.result == "L":
        return f"{team_name} viene de perder {last.goals_scored}-{last.goals_conceded}."
    elif last.result == "D":
        return f"{team_name} viene de empatar {last.goals_scored}-{last.goals_conceded}."
    return ""


def _translate_round(round_str: str) -> str:
    """Translate round codes to Spanish."""
    translations: dict[str, str] = {
        "Group Stage": "Fase de Grupos",
        "Round of 16": "Octavos de Final",
        "Quarter-finals": "Cuartos de Final",
        "Semi-finals": "Semifinal",
        "Final": "Final",
    }
    return translations.get(round_str, round_str)


def strategy_advice(mode: str, position: int) -> str:
    """Provide strategic advice based on position in standings."""
    advice: dict[str, dict[str, str]] = {
        "minimize_risk": {
            "title": "Vas primero, mantenete conservador.",
            "detail": (
                "Elegí marcadores seguros y de alta probabilidad. "
                "No necesitás arriesgar: tu ventaja es que los demás tienen que alcanzarte."
            ),
        },
        "balanced": {
            "title": "Estás en la pelea, mantené el equilibrio.",
            "detail": (
                "Combiná predicciones seguras con algunas apuestas diferentes. "
                "No te alejes mucho del pelotón pero buscá pequeñas ventajas."
            ),
        },
        "differentiation": {
            "title": "Necesitás recuperar terreno, diferenciate.",
            "detail": (
                "Buscá marcadores poco populares que tengan chance real de ocurrir. "
                "Arriesgá más: si todos eligen lo mismo, no vas a subir."
            ),
        },
        "high_risk": {
            "title": "Estás lejos, es momento de arriesgar.",
            "detail": (
                "Jugá agresivo. Elegí marcadores inesperados con alto potencial. "
                "Ya no tenés nada que perder y todo por ganar."
            ),
        },
    }

    entry = advice.get(mode, advice["balanced"])
    return f"{entry['title']} {entry['detail']}"


def format_expected_rank(mean_rank: float, min_rank: float, max_rank: float) -> str:
    """Format expected rank in natural language."""
    mean = int(round(mean_rank))
    lo = int(round(min_rank))
    hi = int(round(max_rank))
    return (
        f"Con estos pronósticos, tu posición esperada al final del torneo "
        f"es {mean}° (entre {lo}° y {hi}°)"
    )


def format_empty_db_message() -> str:
    """Message shown when there are no matches in the database."""
    return (
        "No hay partidos cargados. "
        "Ejecutá `bestbet update --source all` para descargar los datos."
    )


def format_match_datetime(dt: datetime) -> str:
    """Format datetime for display."""
    months = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ]
    return f"{dt.day} de {months[dt.month - 1]} — {dt.hour:02d}:{dt.minute:02d}"
