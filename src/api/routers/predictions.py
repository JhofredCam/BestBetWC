from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_dixon_coles, get_ep_calculator, get_strategy_selector
from src.api.schemas import (
    PredictionResponse,
    PredictRequest,
    ScoreProbability,
    StrategyRecommendationResponse,
)
from src.database.models import Match
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector

router = APIRouter()


def _build_prediction_response(
    home_team: str,
    away_team: str,
    pred: MatchPrediction,
    ep_calculator: ExpectedScoreCalculator,
    strategy_selector: StrategySelector,
    current_position: int = 1,
    total_participants: int = 15,
) -> PredictionResponse:
    all_ranked = ep_calculator.rank_all_predictions(pred)

    top_n = min(10, len(all_ranked))
    top_predictions = []
    for r in all_ranked[:top_n]:
        top_predictions.append(
            ScoreProbability(
                home_goals=r.home_goals,
                away_goals=r.away_goals,
                probability=r.prob_exact,
                ep_total=r.ep_total,
                ep_exact=r.ep_exact,
                ep_result=r.ep_result,
                ep_goals=r.ep_goals_home + r.ep_goals_away,
                ep_unique=r.ep_unique,
            )
        )

    rec = strategy_selector.get_recommendation(pred, current_position, total_participants)
    recommendation = StrategyRecommendationResponse(
        prediction=f"{rec.prediction.home_goals}-{rec.prediction.away_goals}",
        ep_total=rec.prediction.ep_total,
        strategy_mode=rec.strategy_mode.value,
        reasoning=rec.reasoning,
        risk_score=rec.risk_score,
        upside_potential=rec.upside_potential,
        risk_of_ruin=rec.risk_of_ruin,
    )

    score_matrix = pred.score_matrix.tolist()

    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=pred.home_win_prob,
        draw_prob=pred.draw_prob,
        away_win_prob=pred.away_win_prob,
        most_likely_score=f"{pred.most_likely_score[0]}-{pred.most_likely_score[1]}",
        most_likely_score_prob=pred.most_likely_score_prob,
        expected_home_goals=pred.expected_home_goals,
        expected_away_goals=pred.expected_away_goals,
        top_predictions=top_predictions,
        recommendation=recommendation,
        score_matrix=score_matrix,
    )


@router.post("/", response_model=PredictionResponse)
async def predict_match(
    request: PredictRequest,
    dc_model: DixonColes = Depends(get_dixon_coles),
    ep_calculator: ExpectedScoreCalculator = Depends(get_ep_calculator),
    strategy_selector: StrategySelector = Depends(get_strategy_selector),
) -> PredictionResponse:
    pred = dc_model.predict_from_params(request.home_lambda, request.away_lambda)
    return _build_prediction_response(
        home_team=request.home_team,
        away_team=request.away_team,
        pred=pred,
        ep_calculator=ep_calculator,
        strategy_selector=strategy_selector,
        current_position=request.current_position,
    )


@router.get("/match/{match_id}", response_model=PredictionResponse)
async def predict_match_by_id(
    match_id: int,
    position: int = Query(default=1, ge=1, le=15),
    db: Session = Depends(get_db),
    dc_model: DixonColes = Depends(get_dixon_coles),
    ep_calculator: ExpectedScoreCalculator = Depends(get_ep_calculator),
    strategy_selector: StrategySelector = Depends(get_strategy_selector),
) -> PredictionResponse:
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    home_team = match.home_team.name if match.home_team else f"Team{match.home_team_id}"
    away_team = match.away_team.name if match.away_team else f"Team{match.away_team_id}"

    pred = dc_model.predict_match(home_team, away_team)
    return _build_prediction_response(
        home_team=home_team,
        away_team=away_team,
        pred=pred,
        ep_calculator=ep_calculator,
        strategy_selector=strategy_selector,
        current_position=position,
    )


@router.get("/match/{match_id}/analysis")
async def get_match_analysis(
    match_id: int,
    position: int = Query(default=1, ge=1, le=15),
    db: Session = Depends(get_db),
    dc_model: DixonColes = Depends(get_dixon_coles),
    ep_calculator: ExpectedScoreCalculator = Depends(get_ep_calculator),
    strategy_selector: StrategySelector = Depends(get_strategy_selector),
) -> PredictionResponse:
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    home_team = match.home_team.name if match.home_team else f"Team{match.home_team_id}"
    away_team = match.away_team.name if match.away_team else f"Team{match.away_team_id}"
    venue = match.venue
    match_round = match.round
    match_group = match.group

    pred = dc_model.predict_match(home_team, away_team)
    response = _build_prediction_response(
        home_team=home_team,
        away_team=away_team,
        pred=pred,
        ep_calculator=ep_calculator,
        strategy_selector=strategy_selector,
        current_position=position,
    )
    response.venue = venue
    response.round = match_round
    response.group = match_group
    return response


@router.get("/upcoming", response_model=list[PredictionResponse])
async def predict_upcoming_matches(
    position: int = Query(default=1, ge=1, le=15),
    db: Session = Depends(get_db),
    dc_model: DixonColes = Depends(get_dixon_coles),
    ep_calculator: ExpectedScoreCalculator = Depends(get_ep_calculator),
    strategy_selector: StrategySelector = Depends(get_strategy_selector),
) -> list[PredictionResponse]:
    matches = (
        db.query(Match)
        .filter(Match.status.ilike("%schedul%"))
        .order_by(Match.datetime.asc())
        .all()
    )

    results: list[PredictionResponse] = []
    for match in matches:
        home_team = match.home_team.name if match.home_team else f"Team{match.home_team_id}"
        away_team = match.away_team.name if match.away_team else f"Team{match.away_team_id}"
        pred = dc_model.predict_match(home_team, away_team)
        results.append(
            _build_prediction_response(
                home_team=home_team,
                away_team=away_team,
                pred=pred,
                ep_calculator=ep_calculator,
                strategy_selector=strategy_selector,
                current_position=position,
            )
        )

    return results
