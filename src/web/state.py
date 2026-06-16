"""
Session state management for BestBetWC Streamlit UI.
Provides default values and typed accessors for st.session_state keys.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_POSITION = 3
DEFAULT_PARTICIPANTS = 15
DEFAULT_HOME_LAMBDA = 1.5
DEFAULT_AWAY_LAMBDA = 1.0


def init_state() -> None:
    """Initialize session state with defaults if not already set."""
    defaults: dict[str, Any] = {
        "position": DEFAULT_POSITION,
        "total_participants": DEFAULT_PARTICIPANTS,
        "home_lambda": DEFAULT_HOME_LAMBDA,
        "away_lambda": DEFAULT_AWAY_LAMBDA,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_position() -> int:
    return int(st.session_state.get("position", DEFAULT_POSITION))


def get_total_participants() -> int:
    return int(st.session_state.get("total_participants", DEFAULT_PARTICIPANTS))
