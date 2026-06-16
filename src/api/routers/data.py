from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import UpdateDataRequest
from src.database.models import Match, Team

router = APIRouter()


@router.post("/update")
async def update_data(
    request: UpdateDataRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return {
        "message": f"Data update triggered for source: {request.source}",
        "status": "queued",
    }


@router.get("/status")
async def data_status(db: Session = Depends(get_db)) -> dict:
    match_count = db.query(Match).count()
    team_count = db.query(Team).count()
    completed_matches = (
        db.query(Match).filter(Match.status == "completed").count()
        if hasattr(db, "query")
        else 0
    )

    return {
        "total_matches": match_count,
        "total_teams": team_count,
        "completed_matches": completed_matches,
        "status": "ok",
    }
