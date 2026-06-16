from dataclasses import dataclass, field

import numpy as np

from src.config import POLLA_RULES, PollaRules
from src.models.dixon_coles import MatchPrediction


@dataclass
class BracketConfig:
    bonus_16: int = 10
    bonus_8: int = 8
    bonus_4: int = 4
    bonus_semi: int = 2
    bonus_final: int = 5
    num_groups: int = 12
    teams_per_group: int = 4
    advancing_per_group: int = 2
    best_thirds: int = 8

    @classmethod
    def from_polla_rules(cls, rules: PollaRules | None = None) -> "BracketConfig":
        r = rules or POLLA_RULES
        return cls(
            bonus_16=r.round_bonus_16,
            bonus_8=r.round_bonus_8,
            bonus_4=r.round_bonus_4,
            bonus_semi=r.round_bonus_semi,
            bonus_final=r.round_bonus_final,
        )


@dataclass
class BracketPrediction:
    round_of_32: list[str]
    round_of_16: list[str]
    quarter_finalists: list[str]
    semi_finalists: list[str]
    finalists: list[str]
    champion: str | None
    prob_perfect: float
    expected_bonus: float
    round_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    advancing_probs: dict[str, float] = field(default_factory=dict)


@dataclass
class GroupPrediction:
    group_name: str
    standings: list[tuple[str, float]]
    winner_prob: dict[str, float]
    runner_up_prob: dict[str, float]


class BracketOptimizer:
    def __init__(
        self,
        config: BracketConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or BracketConfig.from_polla_rules()
        self._rng = np.random.default_rng(seed if seed is not None else 42)

    def predict_group_standings(
        self,
        group_matches: list[dict[str, str]],
        match_predictions: dict[tuple[str, str], MatchPrediction],
        num_simulations: int = 10000,
    ) -> GroupPrediction:
        group_name = _extract_group_name(group_matches)
        all_teams = sorted(set(t for m in group_matches for t in (m["home"], m["away"])))
        advancing_count = {team: 0 for team in all_teams}
        winner_count = {team: 0 for team in all_teams}
        runner_up_count = {team: 0 for team in all_teams}

        for _ in range(num_simulations):
            table = {team: {"pts": 0, "gf": 0, "gc": 0} for team in all_teams}

            for m in group_matches:
                home = m["home"]
                away = m["away"]
                pred = match_predictions.get((home, away))
                if pred is None:
                    continue
                h_goals, a_goals = _sample_score(pred.score_matrix, self._rng)
                table[home]["gf"] += h_goals
                table[home]["gc"] += a_goals
                table[away]["gf"] += a_goals
                table[away]["gc"] += h_goals

                if h_goals > a_goals:
                    table[home]["pts"] += 3
                elif h_goals == a_goals:
                    table[home]["pts"] += 1
                    table[away]["pts"] += 1
                else:
                    table[away]["pts"] += 3

            sorted_table = sorted(
                table.items(),
                key=lambda x: (
                    x[1]["pts"],
                    x[1]["gf"] - x[1]["gc"],
                    x[1]["gf"],
                ),
                reverse=True,
            )

            winner_count[sorted_table[0][0]] += 1
            runner_up_count[sorted_table[1][0]] += 1
            advancing_count[sorted_table[0][0]] += 1
            advancing_count[sorted_table[1][0]] += 1

        n = max(num_simulations, 1)
        standings = [
            (team, advancing_count[team] / n) for team, _ in advancing_count.items()
        ]
        standings.sort(key=lambda x: x[1], reverse=True)

        return GroupPrediction(
            group_name=group_name,
            standings=standings,
            winner_prob={t: c / n for t, c in winner_count.items()},
            runner_up_prob={t: c / n for t, c in runner_up_count.items()},
        )

    def predict_bracket(
        self,
        all_matches: list[dict[str, str]],
        match_predictions: dict[tuple[str, str], MatchPrediction],
        num_simulations: int = 10000,
    ) -> BracketPrediction:
        groups = _build_groups(all_matches)

        r32_count: dict[str, int] = {}
        r16_count: dict[str, int] = {}
        qf_count: dict[str, int] = {}
        sf_count: dict[str, int] = {}
        f_count: dict[str, int] = {}
        champ_count: dict[str, int] = {}

        for _ in range(num_simulations):
            advancing = _simulate_group_stage(
                groups, match_predictions, self.config, self._rng
            )
            bracket_advancing = advancing[:32]

            r32_teams = bracket_advancing[:32]
            for t in r32_teams:
                r32_count[t] = r32_count.get(t, 0) + 1

            r16_teams = _simulate_knockout_round(
                r32_teams, match_predictions, self._rng
            )
            for t in r16_teams:
                r16_count[t] = r16_count.get(t, 0) + 1

            qf_teams = _simulate_knockout_round(
                r16_teams, match_predictions, self._rng
            )
            for t in qf_teams:
                qf_count[t] = qf_count.get(t, 0) + 1

            sf_teams = _simulate_knockout_round(
                qf_teams, match_predictions, self._rng
            )
            for t in sf_teams:
                sf_count[t] = sf_count.get(t, 0) + 1

            final_teams = _simulate_knockout_round(
                sf_teams, match_predictions, self._rng
            )
            for t in final_teams:
                f_count[t] = f_count.get(t, 0) + 1

            champion = _simulate_knockout_round(
                final_teams, match_predictions, self._rng
            )
            if champion:
                champ_count[champion[0]] = champ_count.get(champion[0], 0) + 1

        ns = max(num_simulations, 1)
        most_likely_r32 = sorted(r32_count, key=lambda t: r32_count[t], reverse=True)[:32]
        most_likely_r16 = sorted(r16_count, key=lambda t: r16_count[t], reverse=True)[:16]
        most_likely_qf = sorted(qf_count, key=lambda t: qf_count[t], reverse=True)[:8]
        most_likely_sf = sorted(sf_count, key=lambda t: sf_count[t], reverse=True)[:4]
        most_likely_f = sorted(f_count, key=lambda t: f_count[t], reverse=True)[:2]
        most_likely_champ = max(champ_count, key=lambda t: champ_count[t]) if champ_count else None

        r32_probs = {t: r32_count.get(t, 0) / ns for t in most_likely_r32}
        prob_perfect_r32 = float(np.prod([r32_probs.get(t, 0) for t in most_likely_r32]))
        prob_perfect_r16 = float(np.prod([
            r16_count.get(t, 0) / ns for t in most_likely_r16
        ]))
        prob_perfect_qf = float(np.prod([
            qf_count.get(t, 0) / ns for t in most_likely_qf
        ]))
        prob_perfect_sf = float(np.prod([
            sf_count.get(t, 0) / ns for t in most_likely_sf
        ]))

        if most_likely_champ and len(most_likely_f) >= 2:
            pf_f0 = f_count.get(most_likely_f[0], 0) / ns
            pf_f1 = f_count.get(most_likely_f[1], 0) / ns
            prob_perfect_f = pf_f0 * pf_f1
            if prob_perfect_f > 0:
                prob_perfect_f *= champ_count.get(most_likely_champ, 0) / ns
        else:
            prob_perfect_f = 0.0

        prob_perfect = float(min(prob_perfect_r32 * prob_perfect_r16 * prob_perfect_qf, 1.0))

        expected_bonus = (
            prob_perfect_r32 * self.config.bonus_16
            + prob_perfect_r16 * self.config.bonus_8
            + prob_perfect_qf * self.config.bonus_4
            + prob_perfect_sf * self.config.bonus_semi
            + prob_perfect_f * self.config.bonus_final
        )

        advancing_probs: dict[str, float] = {}
        all_teams_set = set(t for m in all_matches for t in (m["home"], m["away"]))
        for t in all_teams_set:
            advancing_probs[t] = r16_count.get(t, 0) / ns

        return BracketPrediction(
            round_of_32=most_likely_r32,
            round_of_16=most_likely_r16,
            quarter_finalists=most_likely_qf,
            semi_finalists=most_likely_sf,
            finalists=most_likely_f,
            champion=most_likely_champ,
            prob_perfect=prob_perfect,
            expected_bonus=expected_bonus,
            round_probs={
                "r32": r32_probs,
                "r16": {t: r16_count.get(t, 0) / ns for t in most_likely_r16},
                "qf": {t: qf_count.get(t, 0) / ns for t in most_likely_qf},
                "sf": {t: sf_count.get(t, 0) / ns for t in most_likely_sf},
                "final": {t: f_count.get(t, 0) / ns for t in most_likely_f},
            },
            advancing_probs=advancing_probs,
        )

    def calculate_expected_bracket_bonus(
        self, bracket_pred: BracketPrediction
    ) -> float:
        return bracket_pred.expected_bonus

    def optimize_bracket_predictions(
        self,
        bracket_pred: BracketPrediction,
        ownership_estimates: dict[str, np.ndarray],
    ) -> BracketPrediction:
        return bracket_pred

    def bracket_contrarian_value(
        self,
        my_bracket: BracketPrediction,
        opponent_brackets: list[BracketPrediction],
    ) -> float:
        if not opponent_brackets:
            return 0.0

        overlap_scores: list[float] = []
        rounds = ["r32", "r16", "qf", "sf", "final"]
        my_teams = _flatten_bracket_teams(my_bracket, rounds)

        for opp in opponent_brackets:
            opp_teams = _flatten_bracket_teams(opp, rounds)
            overlap = len(my_teams & opp_teams) / max(len(my_teams | opp_teams), 1)
            overlap_scores.append(overlap)

        avg_overlap = float(np.mean(overlap_scores))
        return float(1.0 - avg_overlap)


class KnockoutSimulator:
    def __init__(
        self,
        match_predictions: dict[tuple[str, str], MatchPrediction],
        seed: int | None = None,
    ) -> None:
        self._predictions = match_predictions
        self._rng = np.random.default_rng(seed if seed is not None else 42)

    def simulate_round(
        self,
        teams: list[str],
    ) -> list[str]:
        return _simulate_knockout_round(teams, self._predictions, self._rng)

    def simulate_full_bracket(
        self,
        round_of_32: list[str],
    ) -> BracketPrediction:
        r16 = _simulate_knockout_round(round_of_32, self._predictions, self._rng)
        qf = _simulate_knockout_round(r16, self._predictions, self._rng)
        sf = _simulate_knockout_round(qf, self._predictions, self._rng)
        final = _simulate_knockout_round(sf, self._predictions, self._rng)
        champion_teams = _simulate_knockout_round(final, self._predictions, self._rng)
        champion = champion_teams[0] if champion_teams else None

        return BracketPrediction(
            round_of_32=round_of_32,
            round_of_16=r16,
            quarter_finalists=qf,
            semi_finalists=sf,
            finalists=final,
            champion=champion,
            prob_perfect=0.0,
            expected_bonus=0.0,
        )


def _sample_score(score_matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = score_matrix.ravel()
    flat = flat / flat.sum()
    idx = rng.choice(len(flat), p=flat)
    n = score_matrix.shape[0]
    return idx // n, idx % n


def _extract_group_name(matches: list[dict[str, str]]) -> str:
    for m in matches:
        if "group" in m:
            return m["group"]
    return "UNKNOWN"


def _build_groups(
    all_matches: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for m in all_matches:
        grp = m.get("group", "A")
        groups.setdefault(grp, []).append(m)
    return groups


def _simulate_group_stage(
    groups: dict[str, list[dict[str, str]]],
    match_predictions: dict[tuple[str, str], MatchPrediction],
    config: BracketConfig,
    rng: np.random.Generator,
) -> list[str]:
    all_winners: list[tuple[str, float, float]] = []
    all_runners_up: list[tuple[str, float, float]] = []
    all_thirds: list[tuple[str, float, float, str]] = []

    for grp_name, grp_matches in sorted(groups.items()):
        teams = sorted(set(
            t for m in grp_matches for t in (m["home"], m["away"])
        ))
        table = {team: {"pts": 0, "gf": 0, "gc": 0} for team in teams}

        for m in grp_matches:
            home = m["home"]
            away = m["away"]
            pred = match_predictions.get((home, away))
            if pred is None:
                continue
            h_goals, a_goals = _sample_score(pred.score_matrix, rng)
            table[home]["gf"] += h_goals
            table[home]["gc"] += a_goals
            table[away]["gf"] += a_goals
            table[away]["gc"] += h_goals

            if h_goals > a_goals:
                table[home]["pts"] += 3
            elif h_goals == a_goals:
                table[home]["pts"] += 1
                table[away]["pts"] += 1
            else:
                table[away]["pts"] += 3

        sorted_table = sorted(
            table.items(),
            key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["gc"], x[1]["gf"]),
            reverse=True,
        )

        all_winners.append((sorted_table[0][0], float(sorted_table[0][1]["pts"]),
                           float(sorted_table[0][1]["gf"] - sorted_table[0][1]["gc"])))
        all_runners_up.append((sorted_table[1][0], float(sorted_table[1][1]["pts"]),
                              float(sorted_table[1][1]["gf"] - sorted_table[1][1]["gc"])))
        all_thirds.append((sorted_table[2][0], float(sorted_table[2][1]["pts"]),
                          float(sorted_table[2][1]["gf"] - sorted_table[2][1]["gc"]), grp_name))

    best_thirds = sorted(all_thirds, key=lambda x: (x[1], x[2]), reverse=True)[:config.best_thirds]

    advancing: list[str] = []
    advancing.extend(t[0] for t in all_winners)
    advancing.extend(t[0] for t in all_runners_up)
    advancing.extend(t[0] for t in best_thirds)

    return advancing


def _simulate_knockout_round(
    teams: list[str],
    match_predictions: dict[tuple[str, str], MatchPrediction],
    rng: np.random.Generator,
) -> list[str]:
    if len(teams) < 2:
        return teams[:]

    winners: list[str] = []
    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            winners.append(teams[i])
            continue
        home = teams[i]
        away = teams[i + 1]
        pred = match_predictions.get((home, away))
        if pred is None:
            pred = match_predictions.get((away, home))
        if pred is None:
            winners.append(home)
            continue

        h_win = pred.home_win_prob
        a_win = pred.away_win_prob
        draw = pred.draw_prob

        if draw > 0:
            denom = h_win + a_win
            prob_h = h_win / denom if denom > 0 else 0.5
        else:
            denom = h_win + a_win
            prob_h = h_win / denom if denom > 0 else 0.5

        winner = home if rng.random() < prob_h else away
        winners.append(winner)

    return winners


def _flatten_bracket_teams(
    bracket: BracketPrediction, rounds: list[str]
) -> set[str]:
    teams: set[str] = set()
    teams.update(bracket.round_of_32)
    teams.update(bracket.round_of_16)
    teams.update(bracket.quarter_finalists)
    teams.update(bracket.semi_finalists)
    teams.update(bracket.finalists)
    if bracket.champion:
        teams.add(bracket.champion)
    return teams
