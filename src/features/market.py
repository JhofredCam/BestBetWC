"""Market features extracted from betting odds data."""

import numpy as np
from sqlalchemy.orm import Session

from src.database.models import CorrectScoreOdds, Odds

FEATURE_NAMES = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "market_btts_yes_prob",
    "market_over_15_prob",
    "market_over_25_prob",
    "market_over_35_prob",
    "market_margin",
    "market_num_bookmakers",
    "market_disagreement",
    "odds_movement_home",
    "odds_movement_draw",
    "odds_movement_away",
    "correct_score_entropy",
]


def _implied_probabilities(
    home_odds: float, draw_odds: float, away_odds: float,
) -> tuple[float, float, float, float]:
    margin = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds
    home_prob = (1.0 / home_odds) / margin
    draw_prob = (1.0 / draw_odds) / margin
    away_prob = (1.0 / away_odds) / margin
    return home_prob, draw_prob, away_prob, margin


def _shannon_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def get_market_features(session: Session, match_id: int) -> dict[str, float]:
    closing_odds = (
        session.query(Odds)
        .filter(Odds.match_id == match_id)
        .order_by(Odds.timestamp.desc())
        .all()
    )

    result: dict[str, float] = {
        "market_home_prob": 0.333,
        "market_draw_prob": 0.333,
        "market_away_prob": 0.333,
        "market_btts_yes_prob": 0.0,
        "market_over_15_prob": 0.0,
        "market_over_25_prob": 0.0,
        "market_over_35_prob": 0.0,
        "market_margin": 0.0,
        "market_num_bookmakers": 0,
        "market_disagreement": 0.0,
        "odds_movement_home": 0.0,
        "odds_movement_draw": 0.0,
        "odds_movement_away": 0.0,
        "correct_score_entropy": 0.0,
    }

    if closing_odds:
        latest_ts = closing_odds[0].timestamp
        latest_odds = [o for o in closing_odds if o.timestamp == latest_ts]

        bookmakers: set[str] = set()
        home_probs: list[float] = []
        draw_probs: list[float] = []
        away_probs: list[float] = []

        btts_probs: list[float] = []
        over_15_probs: list[float] = []
        over_25_probs: list[float] = []
        over_35_probs: list[float] = []
        margins: list[float] = []

        for o in latest_odds:
            home_p, draw_p, away_p, margin = _implied_probabilities(
                o.home_odds, o.draw_odds, o.away_odds,
            )
            bookmakers.add(o.bookmaker)
            home_probs.append(home_p)
            draw_probs.append(draw_p)
            away_probs.append(away_p)
            margins.append(margin)

            if o.btts_yes is not None:
                btts_probs.append(o.btts_yes)
            if o.over_15 is not None:
                over_15_probs.append(o.over_15)
            if o.over_25 is not None:
                over_25_probs.append(o.over_25)
            if o.over_35 is not None:
                over_35_probs.append(o.over_35)

        if home_probs:
            result["market_home_prob"] = float(np.mean(home_probs))
            result["market_draw_prob"] = float(np.mean(draw_probs))
            result["market_away_prob"] = float(np.mean(away_probs))
            result["market_margin"] = float(np.mean(margins))
            result["market_num_bookmakers"] = len(bookmakers)
            result["market_disagreement"] = (
                float(np.std(home_probs)) if len(home_probs) > 1 else 0.0
            )

        if btts_probs:
            result["market_btts_yes_prob"] = float(np.mean(btts_probs))
        if over_15_probs:
            result["market_over_15_prob"] = float(np.mean(over_15_probs))
        if over_25_probs:
            result["market_over_25_prob"] = float(np.mean(over_25_probs))
        if over_35_probs:
            result["market_over_35_prob"] = float(np.mean(over_35_probs))

        if len(closing_odds) >= 2:
            earliest_odds = closing_odds[-1]
            home_p_e, draw_p_e, away_p_e, _ = _implied_probabilities(
                earliest_odds.home_odds, earliest_odds.draw_odds, earliest_odds.away_odds,
            )
            home_p_l = home_probs[0] if home_probs else home_p_e
            draw_p_l = draw_probs[0] if draw_probs else draw_p_e
            away_p_l = away_probs[0] if away_probs else away_p_e

            result["odds_movement_home"] = home_p_l - home_p_e
            result["odds_movement_draw"] = draw_p_l - draw_p_e
            result["odds_movement_away"] = away_p_l - away_p_e

    correct_scores = (
        session.query(CorrectScoreOdds)
        .filter(CorrectScoreOdds.match_id == match_id)
        .all()
    )
    if correct_scores:
        probs = np.array([1.0 / cs.odds for cs in correct_scores if cs.odds > 0])
        if len(probs) > 0:
            probs = probs / probs.sum()
            result["correct_score_entropy"] = _shannon_entropy(probs)

    return result
