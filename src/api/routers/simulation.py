from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from src.api.schemas import SimulateRequest, SimulateResponse, SimulationResult
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator

router = APIRouter()


def _sample_score(pred: MatchPrediction, rng: np.random.Generator) -> tuple[int, int]:
    flat = pred.score_matrix.flatten()
    flat = flat / flat.sum()
    idx = rng.choice(len(flat), p=flat)
    max_g = pred.score_matrix.shape[0] - 1
    return int(idx // (max_g + 1)), int(idx % (max_g + 1))


def _check_result(predicted: tuple[int, int], actual: tuple[int, int]) -> bool:
    ph, pa = predicted
    ah, aa = actual
    return (
        (ph > pa and ah > aa)
        or (ph == pa and ah == aa)
        or (ph < pa and ah < aa)
    )


def _simulate_match(
    dc: DixonColes,
    ep_calculator: ExpectedScoreCalculator,
    home_lambda: float,
    away_lambda: float,
    n: int,
    rng: np.random.Generator,
) -> list[SimulationResult]:
    pred = dc.predict_from_params(home_lambda, away_lambda)
    max_goals = pred.score_matrix.shape[0]

    results_by_score: dict[tuple[int, int], dict[str, list[float]]] = {}
    for _ in range(n):
        sampled = _sample_score(pred, rng)
        score_key = sampled

        if score_key not in results_by_score:
            results_by_score[score_key] = {"ep": [], "result_hits": [], "exact_hits": []}

        ep_result = ep_calculator.calculate_ep(pred, sampled[0], sampled[1])
        results_by_score[score_key]["ep"].append(ep_result.ep_total)

    for score, buckets in results_by_score.items():
        pred_h, pred_a = score
        for _ in range(min(100, len(buckets["ep"]))):
            actual = _sample_score(pred, rng)
            buckets["result_hits"].append(1.0 if _check_result(score, actual) else 0.0)
            buckets["exact_hits"].append(1.0 if score == actual else 0.0)

    sim_results: list[SimulationResult] = []
    for score, buckets in sorted(results_by_score.items()):
        ep_arr = np.array(buckets["ep"])

        result_hit_rate = (
            float(np.mean(buckets["result_hits"])) if buckets["result_hits"] else 0.0
        )
        exact_hit_rate = (
            float(np.mean(buckets["exact_hits"])) if buckets["exact_hits"] else 0.0
        )

        sim_results.append(
            SimulationResult(
                score=f"{score[0]}-{score[1]}",
                ep_mean=float(np.mean(ep_arr)),
                ep_min=float(np.min(ep_arr)),
                ep_max=float(np.max(ep_arr)),
                result_hit_rate=result_hit_rate,
                exact_hit_rate=exact_hit_rate,
                std_dev=float(np.std(ep_arr)),
            )
        )

    return sim_results


@router.post("/match", response_model=SimulateResponse)
async def simulate_match(request: SimulateRequest) -> SimulateResponse:
    dc = DixonColes()
    ep_calculator = ExpectedScoreCalculator()
    rng = np.random.default_rng(42)
    results = _simulate_match(
        dc, ep_calculator, request.home_lambda, request.away_lambda, request.simulations, rng
    )
    return SimulateResponse(
        home_lambda=request.home_lambda,
        away_lambda=request.away_lambda,
        num_simulations=request.simulations,
        results=results,
    )


@router.post("/tournament")
async def simulate_tournament(
    strategy: str = "optimal_ep",
    simulations: int = 10000,
) -> dict[str, str]:
    return {
        "message": "Tournament simulation requires SPEC-014 MonteCarloEngine and tournament data. "
        "Use POST /api/simulation/match for single match simulations.",
    }
