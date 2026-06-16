from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import ProfileResponse
from src.game_theory.profiling import PlayerProfiler

router = APIRouter()


@router.get("/{participant_id}", response_model=ProfileResponse)
async def get_profile(
    participant_id: int,
    db: Session = Depends(get_db),
) -> ProfileResponse:
    profiler = PlayerProfiler(db)
    try:
        profile = profiler.profile_participant(participant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return ProfileResponse(
        participant_name=profile.name,
        archetype=profile.dominant_archetype.value,
        conservative_score=profile.conservative_score,
        aggressive_score=profile.aggressive_score,
        market_follower_score=profile.market_follower_score,
        intuition_score=profile.intuition_score,
        favorite_bias=profile.favorite_bias,
        home_bias=profile.home_bias,
        result_accuracy=profile.result_accuracy,
        exact_accuracy=profile.exact_accuracy,
        avg_points_per_match=profile.avg_points_per_match,
    )


@router.get("/", response_model=list[ProfileResponse])
async def get_all_profiles(
    db: Session = Depends(get_db),
) -> list[ProfileResponse]:
    profiler = PlayerProfiler(db)
    profiles = profiler.profile_all()

    results: list[ProfileResponse] = []
    for p_id, profile in profiles.items():
        results.append(
            ProfileResponse(
                participant_name=profile.name,
                archetype=profile.dominant_archetype.value,
                conservative_score=profile.conservative_score,
                aggressive_score=profile.aggressive_score,
                market_follower_score=profile.market_follower_score,
                intuition_score=profile.intuition_score,
                favorite_bias=profile.favorite_bias,
                home_bias=profile.home_bias,
                result_accuracy=profile.result_accuracy,
                exact_accuracy=profile.exact_accuracy,
                avg_points_per_match=profile.avg_points_per_match,
            )
        )
    return results
