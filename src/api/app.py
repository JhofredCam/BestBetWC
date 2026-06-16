from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import (
    backtesting,
    data,
    matches,
    participants,
    predictions,
    profiles,
    simulation,
    standings,
    strategies,
)
from src.api.schemas import HealthResponse
from src.database.connection import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="BestBetWC API",
        description=(
            "Sistema Inteligente para Optimizar Pronosticos "
            "en Polla Mundialista 2026"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        return HealthResponse(status="ok", version="0.1.0")

    app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
    app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
    app.include_router(strategies.router, prefix="/api/strategies", tags=["Strategies"])
    app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
    app.include_router(backtesting.router, prefix="/api/backtesting", tags=["Backtesting"])
    app.include_router(standings.router, prefix="/api/standings", tags=["Standings"])
    app.include_router(profiles.router, prefix="/api/profiles", tags=["Profiles"])
    app.include_router(data.router, prefix="/api/data", tags=["Data"])
    app.include_router(participants.router, prefix="/api/participants", tags=["Participants"])

    return app


app = create_app()
initialize_database()
