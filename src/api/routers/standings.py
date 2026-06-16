from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import StandingEntry
from src.database.models import Participant, Score, Standing

router = APIRouter()


@router.get("/", response_model=list[StandingEntry])
async def get_standings(
    round_filter: str | None = Query(default=None, alias="round"),
    db: Session = Depends(get_db),
) -> list[StandingEntry]:
    query = db.query(Standing)
    if round_filter:
        query = query.filter(Standing.round == round_filter)
    standings = query.order_by(Standing.position.asc()).all()

    results: list[StandingEntry] = []
    for s in standings:
        participant = db.query(Participant).filter(Participant.id == s.participant_id).first()
        name = participant.name if participant else f"Participant {s.participant_id}"

        scores = (
            db.query(Score)
            .filter(Score.participant_id == s.participant_id)
            .all()
        )
        exact = sum(sc.exact_pts > 0 for sc in scores)
        correct = sum(
            (sc.result_pts > 0 and sc.exact_pts == 0) for sc in scores
        )

        results.append(
            StandingEntry(
                position=s.position,
                participant_name=name,
                total_points=s.total_points,
                exact_scores=exact,
                correct_results=correct,
            )
        )

    return results


@router.get("/participant/{participant_id}")
async def get_participant_performance(
    participant_id: int,
    db: Session = Depends(get_db),
) -> dict:
    participant = db.query(Participant).filter(Participant.id == participant_id).first()
    if participant is None:
        return {"error": f"Participant {participant_id} not found"}

    scores = (
        db.query(Score)
        .filter(Score.participant_id == participant_id)
        .all()
    )

    total_points = sum(s.total_pts for s in scores)
    exact_scores = sum(1 for s in scores if s.exact_pts > 0)
    correct_results = sum(1 for s in scores if s.result_pts > 0 and s.exact_pts == 0)

    return {
        "participant_id": participant_id,
        "name": participant.name,
        "total_points": total_points,
        "exact_scores": exact_scores,
        "correct_results": correct_results,
        "matches_played": len(scores),
    }
