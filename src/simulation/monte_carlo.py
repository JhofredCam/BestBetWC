from __future__ import annotations

from typing import Any, Callable

from src.models.dixon_coles import MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator
from src.simulation.participants import (
    ParticipantSimulator,
    SimulatedParticipant,
    StrategyMode,
)
from src.simulation.tournament import (
    SimulationConfig,
    SimulationReport,
    TournamentResult,
    TournamentSimulator,
)


class MonteCarloEngine:
    def __init__(
        self,
        tournament_sim: TournamentSimulator,
        participant_sim: ParticipantSimulator | None = None,
        ep_calculator: ExpectedScoreCalculator | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self._tournament_sim = tournament_sim
        self._participant_sim = participant_sim or ParticipantSimulator()
        self._ep_calculator = ep_calculator or ExpectedScoreCalculator()
        self._config = config or SimulationConfig()

    def run_full_simulation(
        self,
        match_predictions: dict[str, MatchPrediction],
        my_strategies: dict[str, Callable[[str, MatchPrediction | None], tuple[int, int]]],
        opponent_profiles: list[SimulatedParticipant] | None = None,
        n_simulations: int = 10000,
    ) -> dict[str, SimulationReport]:
        self._tournament_sim.set_match_predictions(match_predictions)
        tournament_results = self._tournament_sim.simulate_n_tournaments(n_simulations)

        reports: dict[str, SimulationReport] = {}

        opp_fns: list[Callable[[str, MatchPrediction | None], tuple[int, int]]] = []
        if opponent_profiles:
            opp_preds_map = self._participant_sim.simulate_all(
                opponent_profiles, match_predictions
            )
            for p in opponent_profiles:
                preds_for_p = opp_preds_map[p.name]

                def make_opp_fn(preds: dict[str, tuple[int, int]]) -> Callable[
                    [str, MatchPrediction | None], tuple[int, int]
                ]:
                    def fn(match_id: str, pred: MatchPrediction | None = None) -> tuple[int, int]:
                        return preds.get(match_id, (0, 0))
                    return fn

                opp_fns.append(make_opp_fn(preds_for_p))

        for name, strat_fn in my_strategies.items():
            report = self._tournament_sim.evaluate_strategy(
                strategy_name=name,
                strategy_fn=strat_fn,
                tournament_results=tournament_results,
                opponent_strategies=opp_fns if opp_fns else None,
            )
            reports[name] = report

        return reports

    def run_what_if(
        self,
        scenario: str,
        match_predictions: dict[str, MatchPrediction],
        n_simulations: int = 5000,
    ) -> SimulationReport:
        self._tournament_sim.set_match_predictions(match_predictions)

        if scenario == "favorites":
            def favorites_fn(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
                if pred is not None:
                    return pred.most_likely_score
                return (2, 1)
            strat_fn = favorites_fn
        elif scenario == "contrarian_50":
            def contrarian_fn(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
                if pred is None:
                    return (1, 1)
                alt = pred.most_likely_score
                rng = __import__('numpy').random.default_rng(hash(match_id) % (2**31))
                if rng.random() < 0.5:
                    alt = (pred.most_likely_score[1], pred.most_likely_score[0])
                return alt
            strat_fn = contrarian_fn
        elif scenario == "underdogs":
            def underdogs_fn(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
                if pred is None:
                    return (0, 1)
                return pred.most_likely_score
            strat_fn = underdogs_fn
        else:
            def default_fn(match_id: str, pred: MatchPrediction | None) -> tuple[int, int]:
                if pred is not None:
                    return pred.most_likely_score
                return (1, 0)
            strat_fn = default_fn

        tournament_results = self._tournament_sim.simulate_n_tournaments(n_simulations)
        return self._tournament_sim.evaluate_strategy(
            strategy_name=scenario,
            strategy_fn=strat_fn,
            tournament_results=tournament_results,
        )
