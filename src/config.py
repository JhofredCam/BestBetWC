import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/bestbetwc.db")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
THE_ODDS_SPORT_KEY = os.getenv("THE_ODDS_SPORT_KEY", "")


@dataclass
class PollaRules:
    result_correct_pts: int = 2
    result_incorrect_pts: int = 0
    exact_score_pts: int = 5
    goals_home_correct_pts: int = 1
    goals_away_correct_pts: int = 1
    unique_prediction_bonus: int = 2
    round_bonus_16: int = 10
    round_bonus_8: int = 8
    round_bonus_4: int = 4
    round_bonus_semi: int = 2
    round_bonus_final: int = 5
    num_participants: int = 15
    max_goals: int = 7


@dataclass
class ModelConfig:
    dixon_coles_rho: float = -0.13
    ensemble_weights: dict[str, float] | None = None
    calibration_method: str = "isotonic"
    validation_folds: int = 5

    def __post_init__(self) -> None:
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "dixon_coles": 0.4,
                "market": 0.4,
                "gradient_boost": 0.2,
            }


@dataclass
class StrategyConfig:
    leading_threshold: int = 1
    middle_range: tuple[int, int] = (2, 5)
    behind_range: tuple[int, int] = (6, 10)
    trailing_range: tuple[int, int] = (11, 15)
    risk_aversion_leading: float = 0.8
    risk_aversion_trailing: float = 0.2
    contrarian_weight: float = 0.3


POLLA_RULES = PollaRules()
MODEL_CONFIG = ModelConfig()
STRATEGY_CONFIG = StrategyConfig()
