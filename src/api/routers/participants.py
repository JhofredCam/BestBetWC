from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.database.models import Participant, Standing

router = APIRouter()


class ParticipantCreate(BaseModel):
    name: str
    points: int = 0


class ParticipantUpdate(BaseModel):
    name: str | None = None
    points: int | None = None
    position: int | None = None


class ParticipantResponse(BaseModel):
    id: int
    name: str
    points: int
    position: int | None = None


@router.get("/", response_model=list[ParticipantResponse])
async def list_participants(
    db: Session = Depends(get_db),
) -> list[ParticipantResponse]:
    participants = db.query(Participant).all()
    results: list[ParticipantResponse] = []
    for p in participants:
        standing = (
            db.query(Standing)
            .filter(Standing.participant_id == p.id)
            .order_by(Standing.round.desc())
            .first()
        )
        results.append(
            ParticipantResponse(
                id=p.id,
                name=p.name,
                points=standing.total_points if standing else 0,
                position=standing.position if standing else None,
            )
        )
    results.sort(key=lambda r: (r.position is None, r.position or 999))
    return results


@router.post("/", response_model=ParticipantResponse, status_code=201)
async def create_participant(
    data: ParticipantCreate,
    db: Session = Depends(get_db),
) -> ParticipantResponse:
    participant = Participant(name=data.name)
    db.add(participant)
    db.flush()

    count = db.query(Participant).count()
    standing = Standing(
        participant_id=participant.id,
        round="overall",
        total_points=data.points,
        position=count,
    )
    db.add(standing)
    db.commit()
    db.refresh(participant)

    return ParticipantResponse(
        id=participant.id,
        name=participant.name,
        points=data.points,
        position=count,
    )


@router.put("/{participant_id}", response_model=ParticipantResponse)
async def update_participant(
    participant_id: int,
    data: ParticipantUpdate,
    db: Session = Depends(get_db),
) -> ParticipantResponse:
    participant = db.query(Participant).filter(Participant.id == participant_id).first()
    if participant is None:
        raise HTTPException(status_code=404, detail="Participante no encontrado")

    if data.name is not None:
        participant.name = data.name

    standing = (
        db.query(Standing)
        .filter(Standing.participant_id == participant_id)
        .filter(Standing.round == "overall")
        .first()
    )

    if standing and data.points is not None:
        standing.total_points = data.points
    elif data.points is not None:
        standing = Standing(
            participant_id=participant_id,
            round="overall",
            total_points=data.points,
            position=data.position or 1,
        )
        db.add(standing)

    if standing and data.position is not None:
        standing.position = data.position

    db.commit()
    db.refresh(participant)

    return ParticipantResponse(
        id=participant.id,
        name=participant.name,
        points=standing.total_points if standing else 0,
        position=standing.position if standing else None,
    )


@router.delete("/{participant_id}", status_code=204)
async def delete_participant(
    participant_id: int,
    db: Session = Depends(get_db),
) -> None:
    participant = db.query(Participant).filter(Participant.id == participant_id).first()
    if participant is None:
        raise HTTPException(status_code=404, detail="Participante no encontrado")

    db.query(Standing).filter(Standing.participant_id == participant_id).delete()
    db.delete(participant)
    db.commit()
