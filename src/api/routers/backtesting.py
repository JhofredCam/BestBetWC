from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_backtest_engine
from src.api.schemas import BacktestRequest, BacktestResponse, BacktestResultItem
from src.models.dixon_coles import MatchPrediction
from src.validation.backtesting import (
    BacktestEngine,
    always_favorite_strategy,
    make_adaptive_strategy,
    make_optimal_ep_contrarian_strategy,
    make_optimal_ep_strategy,
)

router = APIRouter()

STRATEGY_REGISTRY: dict[str, str] = {
    "optimal_ep": "Maximiza expected score basado en Dixon-Coles",
    "always_favorite": "Siempre predice al favorito 1-0, 0-1, o 1-1",
    "contrarian": "Busca resultados contrarian para maximizar valor diferencial",
    "adaptive_1": "Estrategia adaptativa para posición 1 (lider: minimizar riesgo)",
    "adaptive_5": "Estrategia adaptativa para posición 5 (media: equilibrado)",
    "adaptive_10": "Estrategia adaptativa para posición 10 (atras: diferenciación)",
    "adaptive_15": "Estrategia adaptativa para posición 15 (último: alto riesgo)",
}


def _build_strategy_fn(name: str) -> Callable[[MatchPrediction, str, str], tuple[int, int]]:
    if name == "optimal_ep":
        return make_optimal_ep_strategy()
    elif name == "always_favorite":
        return lambda mp, h, a: always_favorite_strategy(mp, h, a)
    elif name == "contrarian":
        return make_optimal_ep_contrarian_strategy()
    elif name.startswith("adaptive_"):
        pos = int(name.split("_")[1])
        return make_adaptive_strategy(current_position=pos)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {name}")


@router.post("/", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    engine: BacktestEngine = Depends(get_backtest_engine),
) -> BacktestResponse:
    if request.strategy not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

    try:
        strategy_fn = _build_strategy_fn(request.strategy)
        result = engine.run_backtest(
            year=request.year,
            strategy_fn=strategy_fn,
            strategy_name=request.strategy,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}") from e

    match_details: list[BacktestResultItem] = []
    for h in result.prediction_history:
        pred = h["prediction"]
        actual = h["actual"]
        match_details.append(
            BacktestResultItem(
                match_id=h.get("match_id", ""),
                home_team=h.get("home_team", ""),
                away_team=h.get("away_team", ""),
                predicted_score=f"{pred[0]}-{pred[1]}",
                actual_score=f"{actual[0]}-{actual[1]}",
                points=float(h.get("points", 0)),
                correct_result=(
                    (pred[0] > pred[1] and actual[0] > actual[1])
                    or (pred[0] == pred[1] and actual[0] == actual[1])
                    or (pred[0] < pred[1] and actual[0] < actual[1])
                ),
                exact_score=(pred[0] == actual[0] and pred[1] == actual[1]),
            )
        )

    brier = result.brier_score if result.brier_score != float("inf") else 0.0
    log_loss = result.log_loss if result.log_loss != float("inf") else 0.0

    return BacktestResponse(
        year=request.year,
        strategy=request.strategy,
        total_points=result.total_points,
        exact_scores=result.exact_scores,
        correct_results=result.correct_results,
        log_loss=log_loss,
        brier_score=brier,
        match_details=match_details,
    )


@router.get("/strategies")
async def list_backtest_strategies() -> dict[str, str]:
    return STRATEGY_REGISTRY
