from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.config import POLLA_RULES, PollaRules
from src.models.dixon_coles import MatchPrediction


@dataclass
class SimulationConfig:
    num_simulations: int = 10000
    max_goals: int = 7
    seed: int | None = 42
    track_progress: bool = True
    n_groups: int = 12
    teams_per_group: int = 4
    n_third_place_qualify: int = 8


@dataclass
class TournamentResult:
    match_results: list[tuple[int, int]]
    group_standings: dict[str, list[str]]
    knockout_results: list[tuple[str, str, int, int]]
    tournament_id: int = 0


@dataclass
class StrategyResult:
    strategy_name: str
    total_points: float
    match_points: list[float]
    bracket_bonus: float
    position: int
    exact_scores: int
    correct_results: int


@dataclass
class SimulationReport:
    strategy_name: str
    mean_points: float
    std_points: float
    median_points: float
    min_points: float
    max_points: float
    win_probability: float
    top3_probability: float
    last_probability: float
    expected_rank: float
    rank_distribution: dict[int, float]
    points_percentiles: dict[float, float]
    risk_of_ruin: float
    n_simulations: int = 0


_GROUP_MATCHES: list[tuple[int, int]] = [
    (0, 1), (2, 3), (0, 2), (1, 3), (1, 2), (0, 3),
]


def _default_group_teams(config: SimulationConfig) -> list[list[str]]:
    groups: list[list[str]] = []
    for g in range(config.n_groups):
        teams = [f"T{g}-{t}" for t in range(config.teams_per_group)]
        groups.append(teams)
    return groups


def _calculate_points(
    prediction: tuple[int, int],
    result: tuple[int, int],
    rules: PollaRules,
    other_predictions: list[tuple[int, int]] | None = None,
) -> float:
    pred_h, pred_a = prediction
    act_h, act_a = result

    exact = (pred_h == act_h and pred_a == act_a)
    correct_result = (pred_h > pred_a and act_h > act_a) or \
                     (pred_h == pred_a and act_h == act_a) or \
                     (pred_h < pred_a and act_h < act_a)
    goals_home = (pred_h == act_h)
    goals_away = (pred_a == act_a)

    points = 0.0
    if exact:
        points += float(rules.exact_score_pts)
    elif correct_result:
        points += float(rules.result_correct_pts)

    if goals_home:
        points += float(rules.goals_home_correct_pts)
    if goals_away:
        points += float(rules.goals_away_correct_pts)

    if exact and other_predictions is not None and rules.unique_prediction_bonus > 0:
        is_unique = not any(
            (p[0] == pred_h and p[1] == pred_a) for p in other_predictions
        )
        if is_unique:
            points += float(rules.unique_prediction_bonus)

    return points


class TournamentSimulator:
    def __init__(
        self,
        config: SimulationConfig | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self._predictions: dict[str, MatchPrediction] = {}
        self._rng = np.random.default_rng(self.config.seed)
        self._group_teams: list[list[str]] = _default_group_teams(self.config)
        self._match_ids: list[str] = []
        self._match_id_to_teams: dict[str, tuple[str, str]] = {}
        self._group_match_order: dict[int, list[tuple[str, str, tuple[str, str]]]] = {}
        self._knockout_bracket: list[tuple[str, str, str, str, str, str]] = []
        self._knockout_rounds: list[list[tuple[str, str, str, str]]] = []
        self._rules: PollaRules = POLLA_RULES
        self._build_schedule()

    def _build_schedule(self) -> None:
        for g_idx in range(self.config.n_groups):
            teams = self._group_teams[g_idx]
            group_matches: list[tuple[str, str, tuple[str, str]]] = []
            for h_idx, a_idx in _GROUP_MATCHES:
                home = teams[h_idx]
                away = teams[a_idx]
                match_id = f"G{g_idx}-{home}-{away}"
                self._match_ids.append(match_id)
                self._match_id_to_teams[match_id] = (home, away)
                group_matches.append((match_id, home, (home, away)))
            self._group_match_order[g_idx] = group_matches

        total_group_matches = self.config.n_groups * len(_GROUP_MATCHES)
        knock_start = total_group_matches

        round_labels = ["R32", "R16", "QF", "SF", "3P", "F"]
        round_sizes = [16, 8, 4, 2, 1, 1]

        match_counter = knock_start
        for r_label, r_size in zip(round_labels, round_sizes):
            round_matches: list[tuple[str, str, str, str]] = []
            for i in range(r_size):
                m_id = f"KO-{r_label}-{i}"
                round_matches.append((m_id, "TBD", "TBD", r_label))
                self._match_ids.append(m_id)
                self._match_id_to_teams[m_id] = ("TBD", "TBD")
            self._knockout_rounds.append(round_matches)
            match_counter += r_size

        self._total_matches = len(self._match_ids)

    def set_group_teams(self, groups: list[list[str]]) -> None:
        if len(groups) != self.config.n_groups:
            raise ValueError(
                f"Expected {self.config.n_groups} groups, got {len(groups)}"
            )
        self._group_teams = groups
        self._match_ids.clear()
        self._match_id_to_teams.clear()
        self._group_match_order.clear()
        self._knockout_rounds.clear()
        self._build_schedule()

    def set_match_predictions(
        self, predictions: dict[str, MatchPrediction]
    ) -> None:
        self._predictions = predictions

    def _sample_score(self, prediction: MatchPrediction) -> tuple[int, int]:
        flat = prediction.score_matrix.flatten()
        flat = flat / flat.sum()
        idx = self._rng.choice(len(flat), p=flat)
        max_g = prediction.score_matrix.shape[0] - 1
        home_goals = idx // (max_g + 1)
        away_goals = idx % (max_g + 1)
        return int(home_goals), int(away_goals)

    def _simulate_group_stage(self) -> dict[str, list[str]]:
        group_standings: dict[str, list[str]] = {}
        all_third_place: list[tuple[str, int, int, int, int]] = []

        for g_idx in range(self.config.n_groups):
            teams = self._group_teams[g_idx]
            records: dict[str, dict[str, int]] = {
                t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams
            }

            for match_id, home, (h_team, a_team) in self._group_match_order[g_idx]:
                pred = self._predictions.get(match_id)
                if pred is None:
                    h_goals, a_goals = int(self._rng.integers(0, 3)), int(self._rng.integers(0, 3))
                else:
                    h_goals, a_goals = self._sample_score(pred)

                records[h_team]["gf"] += h_goals
                records[h_team]["ga"] += a_goals
                records[a_team]["gf"] += a_goals
                records[a_team]["ga"] += h_goals

                if h_goals > a_goals:
                    records[h_team]["pts"] += 3
                elif h_goals < a_goals:
                    records[a_team]["pts"] += 3
                else:
                    records[h_team]["pts"] += 1
                    records[a_team]["pts"] += 1

            for t in teams:
                records[t]["gd"] = records[t]["gf"] - records[t]["ga"]

            sorted_teams = sorted(
                teams,
                key=lambda t: (records[t]["pts"], records[t]["gd"], records[t]["gf"]),
                reverse=True,
            )
            group_standings[f"G{g_idx}"] = sorted_teams

            if len(sorted_teams) >= 3:
                third = sorted_teams[2]
                r = records[third]
                all_third_place.append((third, r["pts"], r["gd"], r["gf"], g_idx))

        all_third_place.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        qualified_thirds = [t[0] for t in all_third_place[:self.config.n_third_place_qualify]]
        third_place_set = set(qualified_thirds)

        for g_idx in range(self.config.n_groups):
            sorted_teams = group_standings[f"G{g_idx}"]
            qualified = []
            for i, team in enumerate(sorted_teams):
                if i < 2 or (i == 2 and team in third_place_set):
                    qualified.append(team)
            group_standings[f"G{g_idx}"] = qualified

        return group_standings

    def _seed_knockout_bracket(
        self, group_standings: dict[str, list[str]]
    ) -> list[str]:
        all_qualified: list[tuple[int, str]] = []
        group_winners: list[str] = []
        group_runners_up: list[str] = []
        group_thirds: list[str] = []

        for g_idx in range(self.config.n_groups):
            key = f"G{g_idx}"
            qualified = group_standings.get(key, [])
            if len(qualified) >= 1:
                group_winners.append(qualified[0])
                all_qualified.append((0, qualified[0]))
            if len(qualified) >= 2:
                group_runners_up.append(qualified[1])
                all_qualified.append((1, qualified[1]))
            if len(qualified) >= 3:
                group_thirds.append(qualified[2])
                all_qualified.append((2, qualified[2]))

        self._rng.shuffle(group_winners)
        self._rng.shuffle(group_runners_up)
        self._rng.shuffle(group_thirds)

        bracket: list[str] = []
        for i in range(0, len(group_winners), 2):
            if i + 1 < len(group_winners):
                bracket.append(group_winners[i])
                bracket.append(group_winners[i + 1])
        return bracket

    def _simulate_knockout_stage(
        self, qualified_teams: list[str]
    ) -> list[tuple[str, str, int, int]]:
        results: list[tuple[str, str, int, int]] = []
        current_round_teams = list(qualified_teams)
        n_rounds = len(self._knockout_rounds)

        semi_winners: list[str] = []
        semi_losers: list[str] = []

        for round_idx, round_matches in enumerate(self._knockout_rounds):
            is_semi_final = (round_idx == n_rounds - 3)
            is_third_place = (round_idx == n_rounds - 2)

            if is_third_place and semi_losers:
                current_round_teams = list(semi_losers)
            elif round_idx > 0 and not is_third_place:
                pass

            next_round: list[str] = []
            round_losers: list[str] = []

            for match_idx in range(len(round_matches)):
                t_idx_h = match_idx * 2
                t_idx_a = match_idx * 2 + 1

                if t_idx_h >= len(current_round_teams) or t_idx_a >= len(current_round_teams):
                    continue

                home = current_round_teams[t_idx_h]
                away = current_round_teams[t_idx_a]

                match_id = round_matches[match_idx][0]
                pred = self._predictions.get(match_id)
                if pred is None:
                    h_goals = int(self._rng.integers(0, 3))
                    a_goals = int(self._rng.integers(0, 3))
                else:
                    h_goals, a_goals = self._sample_score(pred)

                results.append((home, away, h_goals, a_goals))

                if h_goals > a_goals:
                    next_round.append(home)
                    round_losers.append(away)
                elif a_goals > h_goals:
                    next_round.append(away)
                    round_losers.append(home)
                else:
                    if self._rng.random() < 0.5:
                        next_round.append(home)
                        round_losers.append(away)
                    else:
                        next_round.append(away)
                        round_losers.append(home)

            if is_semi_final:
                semi_winners = list(next_round)
                semi_losers = list(round_losers)
                current_round_teams = semi_winners
            elif is_third_place:
                current_round_teams = semi_winners
            else:
                current_round_teams = next_round

        return results

    def simulate_tournament(self) -> TournamentResult:
        self._rng = np.random.default_rng(
            self._rng.bit_generator.random_raw() % (2**32)
        )

        group_standings_raw = self._simulate_group_stage()

        all_qualified: list[str] = []
        for g_idx in range(self.config.n_groups):
            key = f"G{g_idx}"
            all_qualified.extend(group_standings_raw.get(key, []))

        knockout_results = self._simulate_knockout_stage(all_qualified)

        match_results: list[tuple[int, int]] = []
        for g_idx in range(self.config.n_groups):
            for match_id, home, (h_team, a_team) in self._group_match_order[g_idx]:
                pred = self._predictions.get(match_id)
                if pred is None:
                    h_goals, a_goals = int(self._rng.integers(0, 3)), int(self._rng.integers(0, 3))
                else:
                    h_goals, a_goals = self._sample_score(pred)
                match_results.append((h_goals, a_goals))

        for _, _, h_goals, a_goals in knockout_results:
            match_results.append((h_goals, a_goals))

        return TournamentResult(
            match_results=match_results,
            group_standings={
                f"G{g}": group_standings_raw[f"G{g}"] for g in range(self.config.n_groups)
            },
            knockout_results=knockout_results,
        )

    def simulate_n_tournaments(
        self, n: int | None = None,
    ) -> list[TournamentResult]:
        n_sims = n or self.config.num_simulations
        results: list[TournamentResult] = []

        if self.config.track_progress:
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]
                iterator = tqdm(range(n_sims), desc="Simulando torneos")
            except ImportError:
                iterator = range(n_sims)
        else:
            iterator = range(n_sims)

        for i in iterator:
            result = self.simulate_tournament()
            result.tournament_id = i
            results.append(result)

        return results

    def evaluate_strategy(
        self,
        strategy_name: str,
        strategy_fn: Callable[[str, MatchPrediction | None], tuple[int, int]],
        tournament_results: list[TournamentResult],
        opponent_strategies: list[
            Callable[[str, MatchPrediction | None], tuple[int, int]]
        ] | None = None,
    ) -> SimulationReport:
        all_points: list[float] = []
        rank_counts: dict[int, int] = {}
        n_sims = len(tournament_results)
        n_participants = 1 + (len(opponent_strategies) if opponent_strategies else 0)
        n_participants = max(n_participants, 2)
        win_count = 0
        top3_count = 0
        last_count = 0
        rank_sum = 0.0
        all_exact: list[int] = []
        all_correct: list[int] = []

        for sim_idx, tournament in enumerate(tournament_results):
            my_points_total = 0.0
            my_match_points: list[float] = []
            my_exact = 0
            my_correct = 0
            match_idx = 0

            for g_idx in range(self.config.n_groups):
                for mid, home_team, (h_team, a_team) in self._group_match_order[g_idx]:
                    pred = self._predictions.get(mid)
                    my_pred = strategy_fn(mid, pred)
                    if match_idx < len(tournament.match_results):
                        result = tournament.match_results[match_idx]
                    else:
                        result = (0, 0)

                    opp_preds: list[tuple[int, int]] = []
                    if opponent_strategies:
                        for opp_fn in opponent_strategies:
                            opp_preds.append(opp_fn(mid, pred))

                    pts = _calculate_points(my_pred, result, self._rules, opp_preds)
                    my_points_total += pts
                    my_match_points.append(pts)

                    if my_pred[0] == result[0] and my_pred[1] == result[1]:
                        my_exact += 1
                    elif (my_pred[0] > my_pred[1] and result[0] > result[1]) or \
                         (my_pred[0] == my_pred[1] and result[0] == result[1]) or \
                         (my_pred[0] < my_pred[1] and result[0] < result[1]):
                        my_correct += 1

                    match_idx += 1

            for _, _, hg, ag in tournament.knockout_results:
                pass

            all_points.append(my_points_total)
            all_exact.append(my_exact)
            all_correct.append(my_correct)

            if opponent_strategies:
                opp_points: list[float] = []
                for opp_idx, opp_fn in enumerate(opponent_strategies):
                    opp_total = 0.0
                    match_idx = 0
                    for g_idx in range(self.config.n_groups):
                        for mid, home_team, (h_team, a_team) in self._group_match_order[g_idx]:
                            pred = self._predictions.get(mid)
                            opp_pred = opp_fn(mid, pred)
                            if match_idx < len(tournament.match_results):
                                result = tournament.match_results[match_idx]
                            else:
                                result = (0, 0)

                            other_opp_preds: list[tuple[int, int]] = []
                            if opponent_strategies:
                                for o_idx2, o_fn2 in enumerate(opponent_strategies):
                                    if o_idx2 != opp_idx:
                                        other_opp_preds.append(o_fn2(mid, pred))
                            my_pred = strategy_fn(mid, pred)
                            other_opp_preds.append(my_pred)

                            pts = _calculate_points(opp_pred, result, self._rules, other_opp_preds)
                            opp_total += pts
                            match_idx += 1
                    opp_points.append(opp_total)

                all_scores = [my_points_total] + opp_points
                sorted_scores = sorted(all_scores, reverse=True)
                my_position = sorted_scores.index(my_points_total) + 1
            else:
                my_position = 1

            position = max(1, min(my_position, 15))
            rank_counts[position] = rank_counts.get(position, 0) + 1
            rank_sum += position

            if position == 1:
                win_count += 1
            if position <= 3:
                top3_count += 1
            if position >= 15:
                last_count += 1

        points_arr = np.array(all_points)
        mean_points = float(np.mean(points_arr))
        std_points = float(np.std(points_arr))
        median_points = float(np.median(points_arr))
        min_points = float(np.min(points_arr))
        max_points = float(np.max(points_arr))
        win_prob = win_count / n_sims
        top3_prob = top3_count / n_sims
        last_prob = last_count / n_sims
        expected_rank = rank_sum / n_sims

        rank_dist: dict[int, float] = {}
        for pos in range(1, 16):
            rank_dist[pos] = rank_counts.get(pos, 0) / n_sims

        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        points_percentiles: dict[float, float] = {}
        for pct in percentiles:
            points_percentiles[pct] = float(np.percentile(points_arr, pct * 100))

        risk_of_ruin = sum(
            rank_counts.get(p, 0) for p in range(8, 16)
        ) / n_sims

        return SimulationReport(
            strategy_name=strategy_name,
            mean_points=mean_points,
            std_points=std_points,
            median_points=median_points,
            min_points=min_points,
            max_points=max_points,
            win_probability=win_prob,
            top3_probability=top3_prob,
            last_probability=last_prob,
            expected_rank=expected_rank,
            rank_distribution=rank_dist,
            points_percentiles=points_percentiles,
            risk_of_ruin=risk_of_ruin,
            n_simulations=n_sims,
        )

    def compare_strategies(
        self,
        strategies: dict[str, Callable[[str, MatchPrediction | None], tuple[int, int]]],
        n_simulations: int = 10000,
    ) -> dict[str, SimulationReport]:
        tournament_results = self.simulate_n_tournaments(n_simulations)
        reports: dict[str, SimulationReport] = {}

        strategy_items = list(strategies.items())
        for name, fn in strategy_items:
            opp_fns = [f for n, f in strategy_items if n != name]
            report = self.evaluate_strategy(name, fn, tournament_results, opp_fns)
            reports[name] = report

        return reports

    @staticmethod
    def calculate_win_probability(
        my_points: list[float],
        opponent_points: list[list[float]],
    ) -> float:
        if not my_points or not opponent_points:
            return 0.0

        n = len(my_points)
        wins = 0

        for i in range(n):
            my_p = my_points[i]
            opp_max = max(opp_pts[i] for opp_pts in opponent_points if i < len(opp_pts))
            if my_p > opp_max:
                wins += 1

        return wins / n

    @property
    def match_ids(self) -> list[str]:
        return list(self._match_ids)

    @property
    def total_matches(self) -> int:
        return self._total_matches
