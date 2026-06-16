import numpy as np
import pytest

from src.config import PollaRules
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.bracket import (
    BracketConfig,
    BracketOptimizer,
    BracketPrediction,
    GroupPrediction,
    KnockoutSimulator,
)


def _make_prediction(lambda_h: float, mu_a: float) -> MatchPrediction:
    model = DixonColes(max_goals=7)
    return model.predict_from_params(lambda_h=lambda_h, mu_a=mu_a)


def _build_group_matches(
    teams: list[str], group_name: str = "A"
) -> list[dict[str, str]]:
    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            matches.append({"home": teams[i], "away": teams[j], "group": group_name})
    return matches


def _build_group_predictions(
    teams: list[str], strengths: list[float]
) -> dict[tuple[str, str], MatchPrediction]:
    preds = {}
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i == j:
                continue
            pair = (teams[i], teams[j])
            if pair not in preds:
                lambda_h = strengths[i]
                mu_a = strengths[j]
                preds[pair] = _make_prediction(lambda_h, mu_a)
    return preds


class TestBracketConfig:
    def test_default_values(self) -> None:
        config = BracketConfig()
        assert config.bonus_16 == 10
        assert config.bonus_8 == 8
        assert config.bonus_4 == 4
        assert config.bonus_semi == 2
        assert config.bonus_final == 5
        assert config.num_groups == 12
        assert config.teams_per_group == 4
        assert config.best_thirds == 8

    def test_from_polla_rules(self) -> None:
        rules = PollaRules(
            round_bonus_16=12,
            round_bonus_8=10,
            round_bonus_4=6,
            round_bonus_semi=3,
            round_bonus_final=7,
        )
        config = BracketConfig.from_polla_rules(rules)
        assert config.bonus_16 == 12
        assert config.bonus_8 == 10
        assert config.bonus_4 == 6
        assert config.bonus_semi == 3
        assert config.bonus_final == 7

    def test_custom_bonuses_preserved(self) -> None:
        config = BracketConfig(
            bonus_16=20, bonus_8=15, bonus_4=10, bonus_semi=5, bonus_final=8
        )
        assert config.bonus_16 == 20
        assert config.bonus_8 == 15
        assert config.bonus_4 == 10
        assert config.bonus_semi == 5
        assert config.bonus_final == 8


class TestGroupPrediction:
    def test_standings_structure(self) -> None:
        gp = GroupPrediction(
            group_name="A",
            standings=[("BRA", 0.95), ("ARG", 0.88), ("CHI", 0.15), ("PER", 0.02)],
            winner_prob={"BRA": 0.55, "ARG": 0.45},
            runner_up_prob={"BRA": 0.40, "ARG": 0.43, "CHI": 0.17},
        )
        assert gp.group_name == "A"
        assert len(gp.standings) == 4
        assert gp.standings[0][1] > gp.standings[-1][1]


class TestBracketPrediction:
    def test_bracket_creation(self) -> None:
        bp = BracketPrediction(
            round_of_32=["BRA", "ARG", "FRA", "ENG", "ESP", "GER", "POR", "NED",
                         "BEL", "CRO", "URU", "MEX", "USA", "JPN", "KOR", "SEN",
                         "MAR", "DEN", "SUI", "SRB", "POL", "WAL", "GHA", "CMR",
                         "ECU", "QAT", "IRN", "AUS", "CRC", "NZL", "TUN", "KSA"],
            round_of_16=["BRA", "ARG", "FRA", "ENG", "ESP", "GER", "POR", "NED",
                        "BEL", "CRO", "URU", "MEX", "USA", "JPN", "KOR", "SEN"],
            quarter_finalists=["BRA", "ARG", "FRA", "ENG", "ESP", "GER", "POR", "NED"],
            semi_finalists=["BRA", "ARG", "FRA", "ENG"],
            finalists=["BRA", "FRA"],
            champion="BRA",
            prob_perfect=0.01,
            expected_bonus=0.5,
        )
        assert len(bp.round_of_32) == 32
        assert len(bp.round_of_16) == 16
        assert len(bp.quarter_finalists) == 8
        assert len(bp.semi_finalists) == 4
        assert len(bp.finalists) == 2
        assert bp.champion == "BRA"
        assert 0 <= bp.prob_perfect <= 1.0
        assert bp.expected_bonus >= 0


class TestBracketOptimizer:
    @pytest.fixture
    def optimizer(self) -> BracketOptimizer:
        return BracketOptimizer(seed=42)

    def test_predict_group_standings_basic(self, optimizer: BracketOptimizer) -> None:
        teams = ["BRA", "ARG", "CHI", "PER"]
        strengths = [2.5, 2.0, 1.0, 0.5]
        matches = _build_group_matches(teams, "A")
        preds = _build_group_predictions(teams, strengths)

        result = optimizer.predict_group_standings(matches, preds, num_simulations=5000)

        assert result.group_name == "A"
        assert len(result.standings) == 4

        prob_sum = sum(p for _, p in result.standings)
        assert abs(prob_sum - 2.0) < 0.05

        assert result.standings[0][0] in teams
        assert result.standings[0][1] > 0

    def test_clear_favorite_high_advancing_prob(
        self, optimizer: BracketOptimizer
    ) -> None:
        teams = ["FAV", "MED1", "MED2", "WEAK"]
        strengths = [4.0, 1.5, 1.0, 0.3]
        matches = _build_group_matches(teams, "A")
        preds = _build_group_predictions(teams, strengths)

        result = optimizer.predict_group_standings(matches, preds, num_simulations=5000)

        fav_prob = next(p for t, p in result.standings if t == "FAV")
        weak_prob = next(p for t, p in result.standings if t == "WEAK")
        assert fav_prob > 0.90
        assert weak_prob < fav_prob

    def test_winner_and_runner_up_probs(self, optimizer: BracketOptimizer) -> None:
        teams = ["BRA", "ARG", "CHI", "PER"]
        strengths = [3.0, 2.5, 1.0, 0.4]
        matches = _build_group_matches(teams, "A")
        preds = _build_group_predictions(teams, strengths)

        result = optimizer.predict_group_standings(matches, preds, num_simulations=5000)

        winner_top = max(result.winner_prob, key=lambda t: result.winner_prob[t])
        assert winner_top == "BRA"

        wp_sum = sum(result.winner_prob.values())
        assert abs(wp_sum - 1.0) < 0.05

        rp_sum = sum(result.runner_up_prob.values())
        assert abs(rp_sum - 1.0) < 0.05

    def test_deterministic_with_seed(self) -> None:
        teams = ["BRA", "ARG", "CHI", "PER"]
        strengths = [2.5, 2.0, 1.0, 0.5]
        matches = _build_group_matches(teams, "A")
        preds = _build_group_predictions(teams, strengths)

        opt1 = BracketOptimizer(seed=42)
        result1 = opt1.predict_group_standings(matches, preds, num_simulations=5000)

        opt2 = BracketOptimizer(seed=42)
        result2 = opt2.predict_group_standings(matches, preds, num_simulations=5000)

        for (t1, p1), (t2, p2) in zip(result1.standings, result2.standings):
            assert t1 == t2
            assert abs(p1 - p2) < 1e-10

    def test_predict_bracket_12_groups(self, optimizer: BracketOptimizer) -> None:
        teams_per_group = 4
        strengths = [2.5, 1.8, 1.0, 0.5]
        all_matches: list[dict[str, str]] = []
        all_preds: dict[tuple[str, str], MatchPrediction] = {}

        for g in range(12):
            grp_teams = [f"T{g}_{t}" for t in range(teams_per_group)]
            grp_matches = _build_group_matches(grp_teams, f"Group{g}")
            all_matches.extend(grp_matches)
            grp_preds = _build_group_predictions(grp_teams, strengths)
            all_preds.update(grp_preds)

        result = optimizer.predict_bracket(all_matches, all_preds, num_simulations=2000)

        assert len(result.round_of_32) == 32
        assert len(result.round_of_16) == 16
        assert len(result.quarter_finalists) == 8
        assert len(result.semi_finalists) == 4
        assert len(result.finalists) == 2
        assert result.champion is not None
        assert result.expected_bonus >= 0

    def test_calculate_expected_bracket_bonus(
        self, optimizer: BracketOptimizer
    ) -> None:
        bp = BracketPrediction(
            round_of_32=["T"] * 32,
            round_of_16=["T"] * 16,
            quarter_finalists=["T"] * 8,
            semi_finalists=["T"] * 4,
            finalists=["T"] * 2,
            champion="T",
            prob_perfect=0.0,
            expected_bonus=3.5,
        )
        result = optimizer.calculate_expected_bracket_bonus(bp)
        assert result == 3.5

    def test_ebb_never_exceeds_total_bonus(self, optimizer: BracketOptimizer) -> None:
        max_bonus = (
            optimizer.config.bonus_16
            + optimizer.config.bonus_8
            + optimizer.config.bonus_4
            + optimizer.config.bonus_semi
            + optimizer.config.bonus_final
        )

        teams_per_group = 4
        strengths = [2.0, 1.5, 1.0, 0.5]
        all_matches: list[dict[str, str]] = []
        all_preds: dict[tuple[str, str], MatchPrediction] = {}

        for g in range(12):
            grp_teams = [f"T{g}_{t}" for t in range(teams_per_group)]
            grp_matches = _build_group_matches(grp_teams, f"Group{g}")
            all_matches.extend(grp_matches)
            grp_preds = _build_group_predictions(grp_teams, strengths)
            all_preds.update(grp_preds)

        result = optimizer.predict_bracket(all_matches, all_preds, num_simulations=2000)
        assert result.expected_bonus <= max_bonus

    def test_bracket_contrarian_value_self_similarity(
        self, optimizer: BracketOptimizer
    ) -> None:
        bp = BracketPrediction(
            round_of_32=["T0", "T1"] * 16,
            round_of_16=["T0"] * 16,
            quarter_finalists=["T0"] * 8,
            semi_finalists=["T0"] * 4,
            finalists=["T0"] * 2,
            champion="T0",
            prob_perfect=0.0,
            expected_bonus=0.0,
        )
        value = optimizer.bracket_contrarian_value(bp, [bp])
        assert value == 0.0

    def test_bracket_contrarian_value_different_brackets(
        self, optimizer: BracketOptimizer
    ) -> None:
        bp1 = BracketPrediction(
            round_of_32=[f"A{i}" for i in range(32)],
            round_of_16=[f"A{i}" for i in range(16)],
            quarter_finalists=[f"A{i}" for i in range(8)],
            semi_finalists=[f"A{i}" for i in range(4)],
            finalists=["A0", "A1"],
            champion="A0",
            prob_perfect=0.0,
            expected_bonus=0.0,
        )
        bp2 = BracketPrediction(
            round_of_32=[f"B{i}" for i in range(32)],
            round_of_16=[f"B{i}" for i in range(16)],
            quarter_finalists=[f"B{i}" for i in range(8)],
            semi_finalists=[f"B{i}" for i in range(4)],
            finalists=["B0", "B1"],
            champion="B0",
            prob_perfect=0.0,
            expected_bonus=0.0,
        )
        value = optimizer.bracket_contrarian_value(bp1, [bp2])
        assert value == 1.0

    def test_bracket_contrarian_value_partial_overlap(
        self, optimizer: BracketOptimizer
    ) -> None:
        bp1 = BracketPrediction(
            round_of_32=[f"T{i}" for i in range(32)],
            round_of_16=[f"T{i}" for i in range(16)],
            quarter_finalists=[f"T{i}" for i in range(8)],
            semi_finalists=[f"T{i}" for i in range(4)],
            finalists=["T0", "T1"],
            champion="T0",
            prob_perfect=0.0,
            expected_bonus=0.0,
        )
        bp2 = BracketPrediction(
            round_of_32=[f"T{i}" for i in range(16)] + [f"O{i}" for i in range(16)],
            round_of_16=[f"T{i}" for i in range(8)] + [f"O{i}" for i in range(8)],
            quarter_finalists=[f"T{i}" for i in range(4)] + [f"O{i}" for i in range(4)],
            semi_finalists=[f"T{i}" for i in range(2)] + [f"O{i}" for i in range(2)],
            finalists=["T0", "O0"],
            champion="T0",
            prob_perfect=0.0,
            expected_bonus=0.0,
        )
        value = optimizer.bracket_contrarian_value(bp1, [bp2])
        assert 0.0 < value < 1.0

    def test_bracket_contrarian_value_empty_opponents(
        self, optimizer: BracketOptimizer
    ) -> None:
        bp = BracketPrediction(
            round_of_32=["T0"] * 32,
            round_of_16=["T0"] * 16,
            quarter_finalists=["T0"] * 8,
            semi_finalists=["T0"] * 4,
            finalists=["T0"] * 2,
            champion="T0",
            prob_perfect=0.0,
            expected_bonus=0.0,
        )
        value = optimizer.bracket_contrarian_value(bp, [])
        assert value == 0.0

    def test_optimize_bracket_predictions_passthrough(
        self, optimizer: BracketOptimizer
    ) -> None:
        bp = BracketPrediction(
            round_of_32=[f"T{i}" for i in range(32)],
            round_of_16=[f"T{i}" for i in range(16)],
            quarter_finalists=[f"T{i}" for i in range(8)],
            semi_finalists=[f"T{i}" for i in range(4)],
            finalists=["T0", "T1"],
            champion="T0",
            prob_perfect=0.1,
            expected_bonus=2.0,
        )
        ownership = np.array([[0.1]])
        result = optimizer.optimize_bracket_predictions(bp, {"match_1": ownership})
        assert result.round_of_32 == bp.round_of_32
        assert result.expected_bonus == bp.expected_bonus

    def test_performance_10000_simulations(self, optimizer: BracketOptimizer) -> None:
        import time
        teams = ["BRA", "ARG", "CHI", "PER"]
        strengths = [2.5, 2.0, 1.0, 0.5]
        matches = _build_group_matches(teams, "A")
        preds = _build_group_predictions(teams, strengths)

        start = time.perf_counter()
        result = optimizer.predict_group_standings(matches, preds, num_simulations=10000)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0
        assert result.standings[0][1] > 0


class TestKnockoutSimulator:
    def test_simulate_round_basic(self) -> None:
        teams = ["BRA", "ARG", "FRA", "ENG"]
        preds = {}
        for h, a in [("BRA", "ARG"), ("FRA", "ENG")]:
            pred = _make_prediction(2.5, 1.5)
            preds[(h, a)] = pred

        sim = KnockoutSimulator(preds, seed=42)
        winners = sim.simulate_round(teams)
        assert len(winners) == 2
        assert all(w in teams for w in winners)

    def test_simulate_full_bracket(self) -> None:
        r32 = [f"T{i}" for i in range(32)]
        preds = {}
        for i in range(0, 32, 2):
            h = r32[i]
            a = r32[i + 1]
            pred = _make_prediction(2.0, 1.5)
            preds[(h, a)] = pred

        sim = KnockoutSimulator(preds, seed=42)
        result = sim.simulate_full_bracket(r32)

        assert len(result.round_of_32) == 32
        assert len(result.round_of_16) == 16
        assert len(result.quarter_finalists) == 8
        assert len(result.semi_finalists) == 4
        assert len(result.finalists) == 2
        assert result.champion is not None

    def test_simulate_round_empty(self) -> None:
        sim = KnockoutSimulator({})
        winners = sim.simulate_round([])
        assert winners == []

    def test_simulate_round_single_team(self) -> None:
        sim = KnockoutSimulator({})
        winners = sim.simulate_round(["BRA"])
        assert winners == ["BRA"]

    def test_simulate_round_missing_prediction(self) -> None:
        sim = KnockoutSimulator({})
        winners = sim.simulate_round(["BRA", "ARG"])
        assert len(winners) == 1
        assert winners[0] in ("BRA", "ARG")
