from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import MatchSummary
from src.database.models import Match

router = APIRouter()


def _match_to_summary(m: Match) -> MatchSummary:
    home_team = m.home_team.name if m.home_team else f"Team{m.home_team_id}"
    away_team = m.away_team.name if m.away_team else f"Team{m.away_team_id}"
    return MatchSummary(
        id=m.id,
        home_team=home_team,
        away_team=away_team,
        datetime=m.datetime,
        round=m.round,
        status=m.status,
        home_score=m.home_score,
        away_score=m.away_score,
    )


@router.get("/", response_model=list[MatchSummary])
async def get_matches(
    round_filter: str | None = Query(default=None, alias="round"),
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[MatchSummary]:
    query = db.query(Match)
    if round_filter:
        query = query.filter(Match.round == round_filter)
    if status:
        query = query.filter(Match.status == status)
    matches = query.order_by(Match.datetime.asc()).limit(limit).all()
    return [_match_to_summary(m) for m in matches]


@router.get("/today", response_model=list[MatchSummary])
async def get_today_matches(
    db: Session = Depends(get_db),
) -> list[MatchSummary]:
    now = datetime.now(UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    matches = (
        db.query(Match)
        .filter(Match.datetime >= start, Match.datetime < end)
        .filter(Match.status.ilike("%schedul%"))
        .order_by(Match.datetime.asc())
        .all()
    )
    return [_match_to_summary(m) for m in matches]


@router.get("/tomorrow", response_model=list[MatchSummary])
async def get_tomorrow_matches(
    db: Session = Depends(get_db),
) -> list[MatchSummary]:
    now = datetime.now(UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    end = start + timedelta(days=1)
    matches = (
        db.query(Match)
        .filter(Match.datetime >= start, Match.datetime < end)
        .filter(Match.status.ilike("%schedul%"))
        .order_by(Match.datetime.asc())
        .all()
    )
    return [_match_to_summary(m) for m in matches]


@router.get("/{match_id}", response_model=MatchSummary)
async def get_match(
    match_id: int,
    db: Session = Depends(get_db),
) -> MatchSummary:
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
    return _match_to_summary(match)
