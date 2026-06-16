"""
Cliente HTTP para consumir la API de SPEC-017.
Se usa cuando los datos vienen de BD en lugar de parámetros directos.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx

API_BASE = "http://localhost:8000/api"


async def get_match(match_id: int) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.get(urljoin(API_BASE, f"/matches/{match_id}"))
        r.raise_for_status()
        return dict(r.json())


async def predict_match(
    home_team: str,
    away_team: str,
    home_lambda: float,
    away_lambda: float,
    position: int,
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.post(urljoin(API_BASE, "/predictions/"), json={
            "home_team": home_team,
            "away_team": away_team,
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            "current_position": position,
        })
        r.raise_for_status()
        return dict(r.json())


async def get_standings() -> list[dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        r = await client.get(urljoin(API_BASE, "/standings/"))
        r.raise_for_status()
        return list(r.json())


async def get_profiles() -> list[dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        r = await client.get(urljoin(API_BASE, "/profiles/"))
        r.raise_for_status()
        return list(r.json())
