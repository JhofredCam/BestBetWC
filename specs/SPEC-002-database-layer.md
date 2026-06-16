# SPEC-002: Database Layer

## Status: PLANNED

## Objective

Crear la capa de persistencia con SQLAlchemy ORM y SQLite. Definir todos los modelos
de datos necesarios para el sistema completo (equipos, partidos, cuotas, features,
pronósticos, resultados, puntuaciones, perfiles).

## Dependencies

- **SPEC-001** (config: `DATABASE_URL`)

## Context

El sistema necesita persistir datos de múltiples fuentes (APIs de cuotas, APIs
deportivas, scraping) y generar predicciones. SQLite es suficiente para el MVP
(max 64 partidos por Mundial). PostgreSQL se recomienda para producción.

## Technical Design

### Archivos a crear

```
src/database/
    connection.py      # Engine, Session, Base
    models.py          # Todos los modelos SQLAlchemy
```

### `src/database/connection.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

class Base(DeclarativeBase):
    pass

def get_engine(database_url: str | None = None) -> Engine:
    ...

def get_session() -> Session:
    ...
```

### `src/database/models.py` - Modelos requeridos

```python
class Team(Base):
    """Selección nacional participante en el Mundial"""
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    fifa_code: Mapped[str] = mapped_column(String(3), unique=True)
    confederation: Mapped[str]
    elo_rating: Mapped[float | None]
    fifa_rank: Mapped[int | None]
    group: Mapped[str | None]

class Match(Base):
    """Partido del Mundial"""
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    datetime: Mapped[datetime]
    venue: Mapped[str | None]
    city: Mapped[str | None]
    round: Mapped[str]  # "group", "round_of_16", "quarter", "semi", "final"
    group: Mapped[str | None]
    status: Mapped[str]  # "scheduled", "live", "finished"
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]

class Odds(Base):
    """Snapshot de cuotas de apuestas para un partido"""
    __tablename__ = "odds"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    bookmaker: Mapped[str]
    timestamp: Mapped[datetime]
    home_odds: Mapped[float]
    draw_odds: Mapped[float]
    away_odds: Mapped[float]
    over_15: Mapped[float | None]
    over_25: Mapped[float | None]
    over_35: Mapped[float | None]
    btts_yes: Mapped[float | None]
    btts_no: Mapped[float | None]
    is_closing: Mapped[bool]

class CorrectScoreOdds(Base):
    """Mercado de marcador exacto por casa de apuestas"""
    __tablename__ = "correct_score_odds"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    bookmaker: Mapped[str]
    timestamp: Mapped[datetime]
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    odds: Mapped[float]

class TeamForm(Base):
    """Stats de rendimiento de un equipo en un partido específico"""
    __tablename__ = "team_form"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    goals_scored: Mapped[int]
    goals_conceded: Mapped[int]
    xg: Mapped[float | None]
    xga: Mapped[float | None]
    possession: Mapped[float | None]
    shots: Mapped[int | None]
    shots_on_target: Mapped[int | None]
    result: Mapped[str]  # "W", "D", "L"
    is_home: Mapped[bool]

class HeadToHead(Base):
    """Historial de enfrentamientos directos"""
    __tablename__ = "head_to_head"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_a_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    team_b_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    match_date: Mapped[datetime]
    goals_a: Mapped[int]
    goals_b: Mapped[int]
    competition: Mapped[str | None]

class Injury(Base):
    """Lesiones y suspensiones"""
    __tablename__ = "injuries"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    player_name: Mapped[str]
    injury_type: Mapped[str]
    status: Mapped[str]  # "doubtful", "out", "suspended"
    expected_return: Mapped[datetime | None]

class SystemPrediction(Base):
    """Predicciones generadas por el sistema"""
    __tablename__ = "system_predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    timestamp: Mapped[datetime]
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    ep_score: Mapped[float]
    ownership_estimate: Mapped[float | None]
    contrarian_value: Mapped[float | None]
    confidence: Mapped[float]
    strategy_mode: Mapped[str | None]

class ParticipantPrediction(Base):
    """Pronósticos de otros participantes (scraped)"""
    __tablename__ = "participant_predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    timestamp: Mapped[datetime]

class Participant(Base):
    """Participante de la polla"""
    __tablename__ = "participants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    platform_id: Mapped[str | None]

class ParticipantProfile(Base):
    """Perfil inferido de cada participante (game theory)"""
    __tablename__ = "participant_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    conservative_score: Mapped[float]
    aggressive_score: Mapped[float]
    market_follower: Mapped[float]
    favorite_bias: Mapped[float]
    recency_bias: Mapped[float]
    home_bias: Mapped[float]
    updated_at: Mapped[datetime]

class Score(Base):
    """Puntuación desglosada por participante y partido"""
    __tablename__ = "scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    result_pts: Mapped[int]
    exact_pts: Mapped[int]
    goals_home_pts: Mapped[int]
    goals_away_pts: Mapped[int]
    unique_pts: Mapped[int]
    round_bonus_pts: Mapped[int]
    total_pts: Mapped[int]

class Standing(Base):
    """Clasificación en cada ronda"""
    __tablename__ = "standings"
    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    round: Mapped[str]
    total_points: Mapped[int]
    position: Mapped[int]
```

## Acceptance Criteria

- [ ] `src/database/connection.py` expone `get_engine()` y `get_session()` usando `DATABASE_URL`
- [ ] `src/database/models.py` define todos los modelos listados con tipos correctos
- [ ] `Base.metadata.create_all()` crea las tablas en SQLite sin errores
- [ ] Test: se puede insertar un `Team` y un `Match` y hacer query
- [ ] Test: relaciones entre `Match` y `Odds` funcionan
- [ ] Test: relaciones entre `Participant`, `ParticipantPrediction`, y `Score` funcionan
- [ ] La base de datos se crea en `data/bestbetwc.db` (gitignored)

## Files to Create

```
src/database/connection.py
src/database/models.py
tests/test_database.py
```

## Git Workflow

```bash
git checkout -b feature/spec-002-database-layer

# Commit 1: connection.py
git add src/database/connection.py
git commit -m "feat(SPEC-002): add database connection module"

# Commit 2: models.py
git add src/database/models.py
git commit -m "feat(SPEC-002): add all SQLAlchemy models"

# Commit 3: tests
git add tests/test_database.py
git commit -m "test(SPEC-002): add database model tests"

# Verify
pytest tests/test_database.py -v
ruff check src/database/

# Merge (when approved)
git checkout main
git merge feature/spec-002-database-layer
```

## Verification

```bash
pytest tests/test_database.py -v
python -c "from src.database.connection import get_engine; e = get_engine(); print('OK')"
```
