from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from src.database.connection import get_session
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.validation.backtesting import BacktestEngine


def get_db() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_dixon_coles() -> DixonColes:
    return DixonColes()


def get_ep_calculator() -> ExpectedScoreCalculator:
    return ExpectedScoreCalculator()


def get_strategy_selector() -> StrategySelector:
    return StrategySelector()


def get_backtest_engine() -> BacktestEngine:
    return BacktestEngine()
