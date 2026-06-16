from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_ep_calculator, get_strategy_selector
from src.api.schemas import StrategyModesResponse, StrategyRecommendationResponse
from src.config import STRATEGY_CONFIG
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector

router = APIRouter()


def _build_strategy_modes() -> dict[str, str]:
    c = STRATEGY_CONFIG
    modes: dict[str, str] = {}
    for pos in range(1, 16):
        if pos <= c.leading_threshold:
            mode = "minimize_risk"
        elif c.middle_range[0] <= pos <= c.middle_range[1]:
            mode = "balanced"
        elif c.behind_range[0] <= pos <= c.behind_range[1]:
            mode = "differentiation"
        else:
            mode = "high_risk"
        modes[str(pos)] = mode
    return modes


@router.get("/modes", response_model=StrategyModesResponse)
async def get_strategy_modes() -> StrategyModesResponse:
    return StrategyModesResponse(modes=_build_strategy_modes())


@router.get("/optimal/{position}", response_model=StrategyRecommendationResponse)
async def get_optimal_strategy(
    position: int,
    home_lambda: float = 1.5,
    away_lambda: float = 1.0,
    strategy_selector: StrategySelector = Depends(get_strategy_selector),
) -> StrategyRecommendationResponse:
    dc = DixonColes()
    pred = dc.predict_from_params(home_lambda, away_lambda)
    rec = strategy_selector.get_recommendation(pred, position, 15)
    return StrategyRecommendationResponse(
        prediction=f"{rec.prediction.home_goals}-{rec.prediction.away_goals}",
        ep_total=rec.prediction.ep_total,
        strategy_mode=rec.strategy_mode.value,
        reasoning=rec.reasoning,
        risk_score=rec.risk_score,
        upside_potential=rec.upside_potential,
        risk_of_ruin=rec.risk_of_ruin,
    )
