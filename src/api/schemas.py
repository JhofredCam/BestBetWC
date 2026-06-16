from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    home_team: str = Field(default="Home", description="Nombre del equipo local")
    away_team: str = Field(default="Away", description="Nombre del equipo visitante")
    home_lambda: float = Field(
        default=1.5, ge=0.1, le=10.0, description="Goles esperados del local"
    )
    away_lambda: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Goles esperados del visitante"
    )
    current_position: int = Field(
        default=1, ge=1, le=15, description="Posición actual en la polla"
    )


class PredictFromOddsRequest(BaseModel):
    match_id: int


class SimulateRequest(BaseModel):
    home_lambda: float = 1.5
    away_lambda: float = 1.0
    simulations: int = Field(default=10000, ge=100, le=100000)


class BacktestRequest(BaseModel):
    year: int = Field(default=2022, ge=2010, le=2026)
    strategy: str = "optimal_ep"


class UpdateDataRequest(BaseModel):
    source: str = Field(default="all", pattern=r"^(odds|football|fbref|all)$")


class ScoreProbability(BaseModel):
    home_goals: int
    away_goals: int
    probability: float
    ep_total: float
    ep_exact: float
    ep_result: float
    ep_goals: float
    ep_unique: float


class StrategyRecommendationResponse(BaseModel):
    prediction: str
    ep_total: float
    strategy_mode: str
    reasoning: str
    risk_score: float
    upside_potential: float
    risk_of_ruin: float


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    most_likely_score: str
    most_likely_score_prob: float
    expected_home_goals: float
    expected_away_goals: float
    top_predictions: list[ScoreProbability]
    recommendation: StrategyRecommendationResponse
    score_matrix: list[list[float]]


class StrategyModesResponse(BaseModel):
    modes: dict[str, str]


class SimulationResult(BaseModel):
    score: str
    ep_mean: float
    ep_min: float
    ep_max: float
    result_hit_rate: float
    exact_hit_rate: float
    std_dev: float


class SimulateResponse(BaseModel):
    home_lambda: float
    away_lambda: float
    num_simulations: int
    results: list[SimulationResult]


class BacktestResultItem(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    predicted_score: str
    actual_score: str
    points: float
    correct_result: bool
    exact_score: bool


class BacktestResponse(BaseModel):
    year: int
    strategy: str
    total_points: float
    exact_scores: int
    correct_results: int
    log_loss: float
    brier_score: float
    match_details: list[BacktestResultItem]


class MatchSummary(BaseModel):
    id: int
    home_team: str
    away_team: str
    datetime: datetime
    round: str
    status: str
    home_score: int | None = None
    away_score: int | None = None


class StandingEntry(BaseModel):
    position: int
    participant_name: str
    total_points: int
    exact_scores: int
    correct_results: int


class ProfileResponse(BaseModel):
    participant_name: str
    archetype: str
    conservative_score: float
    aggressive_score: float
    market_follower_score: float
    intuition_score: float
    favorite_bias: float
    home_bias: float
    result_accuracy: float
    exact_accuracy: float
    avg_points_per_match: float


class ErrorResponse(BaseModel):
    detail: str
    code: str


class HealthResponse(BaseModel):
    status: str
    version: str
