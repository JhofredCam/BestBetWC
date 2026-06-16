from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import POLLA_RULES, PollaRules
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector


@dataclass
class BacktestConfig:
    validation_years: list[int] = field(default_factory=lambda: [2014, 2018, 2022])
    max_goals: int = 7
    temporal_split: bool = True
    min_train_matches: int = 20
    data_dir: Path | None = None
    fit_maxiter: int | None = 200
    csv_map: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw"

    def get_csv_path(self, year: int) -> Path:
        if year in self.csv_map:
            if self.data_dir is not None:
                return self.data_dir / self.csv_map[year]
            return Path(self.csv_map[year])
        if self.data_dir is not None:
            return self.data_dir / f"world_cup_{year}.csv"
        return Path(f"world_cup_{year}.csv")


@dataclass
class BacktestMatch:
    match_id: str
    date: datetime
    home_team: str
    away_team: str
    round: str
    home_goals: int
    away_goals: int
    features: dict[str, float] | None = None


@dataclass
class BacktestResult:
    strategy_name: str
    tournament: str
    total_points: float
    points_per_match: list[float]
    exact_scores: int
    correct_results: int
    prediction_history: list[dict]
    log_loss: float
    brier_score: float
    calibration_error: float
    ranked_probability_score: float
    closings_line_value: float | None = None


@dataclass
class BacktestReport:
    tournament: str
    strategies: dict[str, BacktestResult]
    baseline_result: BacktestResult
    relative_performance: dict[str, float]
    summary: str


def _calculate_points(
    prediction: tuple[int, int],
    result: tuple[int, int],
    rules: PollaRules,
) -> float:
    pred_h, pred_a = prediction
    act_h, act_a = result

    exact = pred_h == act_h and pred_a == act_a
    correct_result = (
        (pred_h > pred_a and act_h > act_a)
        or (pred_h == pred_a and act_h == act_a)
        or (pred_h < pred_a and act_h < act_a)
    )
    goals_home = pred_h == act_h
    goals_away = pred_a == act_a

    points = 0.0
    if exact:
        points += float(rules.exact_score_pts)
    elif correct_result:
        points += float(rules.result_correct_pts)

    if goals_home:
        points += float(rules.goals_home_correct_pts)
    if goals_away:
        points += float(rules.goals_away_correct_pts)

    return points


def calculate_ece(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = float(actuals[mask].mean())
            bin_conf = float(probs[mask].mean())
            ece += float(mask.sum()) * abs(bin_acc - bin_conf) / len(probs)
    return float(ece)


def calculate_score_matrix_log_loss(
    score_matrices: list[np.ndarray],
    actual_scores: list[tuple[int, int]],
) -> float:
    if not score_matrices:
        return float("inf")
    ll = 0.0
    for matrix, (h, a) in zip(score_matrices, actual_scores):
        p = max(float(matrix[h, a]), 1e-15)
        ll += np.log(p)
    return -ll / len(score_matrices)


def _calculate_brier_score(
    prob_matrices: list[np.ndarray],
    actual_scores: list[tuple[int, int]],
) -> float:
    if not prob_matrices:
        return float("inf")
    total = 0.0
    n_bins = 0
    for matrix, (h, a) in zip(prob_matrices, actual_scores):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                target = 1.0 if (i == h and j == a) else 0.0
                total += (matrix[i, j] - target) ** 2
                n_bins += 1
    return total / n_bins


def _calculate_rps(
    prob_matrices: list[np.ndarray],
    actual_scores: list[tuple[int, int]],
) -> float:
    if not prob_matrices:
        return float("inf")
    total = 0.0
    for matrix, (h, a) in zip(prob_matrices, actual_scores):
        max_g = matrix.shape[0]
        flat_pred = matrix.flatten()
        flat_obs = np.zeros_like(flat_pred)
        obs_idx = h * max_g + a
        flat_obs[obs_idx] = 1.0
        cum_pred = np.cumsum(flat_pred)
        cum_obs = np.cumsum(flat_obs)
        total += float(np.sum((cum_pred - cum_obs) ** 2))
    return total / len(prob_matrices)


def always_favorite_strategy(
    match_pred: MatchPrediction,
    team_home: str,
    team_away: str,
) -> tuple[int, int]:
    if match_pred.home_win_prob > match_pred.away_win_prob:
        return (1, 0)
    elif match_pred.away_win_prob > match_pred.home_win_prob:
        return (0, 1)
    else:
        return (1, 1)


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self._rules = POLLA_RULES
        self._ep_calculator = ExpectedScoreCalculator()
        self._max_goals = self.config.max_goals

    def load_tournament_data(self, year: int, data_file: Path | None = None) -> list[BacktestMatch]:
        if data_file is not None:
            csv_path = data_file
        else:
            csv_path = self.config.get_csv_path(year)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {csv_path}. "
                f"Place world_cup_{year}.csv in {self.config.data_dir}"
            )

        matches: list[BacktestMatch] = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                matches.append(
                    BacktestMatch(
                        match_id=row.get("match_id", ""),
                        date=datetime.fromisoformat(row["date"]),
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        round=row.get("round", "Unknown"),
                        home_goals=int(row["home_goals"]),
                        away_goals=int(row["away_goals"]),
                    )
                )

        matches.sort(key=lambda m: m.date)
        return matches

    def _matches_to_dicts(self, matches: list[BacktestMatch]) -> list[dict[str, int | str]]:
        return [
            {
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
            }
            for m in matches
        ]

    def run_backtest(
        self,
        year: int,
        strategy_fn: Callable[[MatchPrediction, str, str], tuple[int, int]],
        strategy_name: str,
        data_file: Path | None = None,
    ) -> BacktestResult:
        tournament = f"World Cup {year}"
        matches = self.load_tournament_data(year, data_file=data_file)

        if len(matches) == 0:
            return BacktestResult(
                strategy_name=strategy_name,
                tournament=tournament,
                total_points=0.0,
                points_per_match=[],
                exact_scores=0,
                correct_results=0,
                prediction_history=[],
                log_loss=float("inf"),
                brier_score=float("inf"),
                calibration_error=float("inf"),
                ranked_probability_score=float("inf"),
            )

        total_points = 0.0
        points_per_match: list[float] = []
        exact_scores = 0
        correct_results = 0
        prediction_history: list[dict] = []

        score_matrices: list[np.ndarray] = []
        actual_score_list: list[tuple[int, int]] = []

        home_probs: list[float] = []
        home_actuals: list[float] = []

        model = DixonColes(max_goals=self._max_goals)
        fit_opts = {"maxiter": self.config.fit_maxiter} if self.config.fit_maxiter else None

        for i, match in enumerate(matches):
            train_matches = matches[:i]
            train_dicts = self._matches_to_dicts(train_matches)

            model = DixonColes(max_goals=self._max_goals)
            if len(train_dicts) >= self.config.min_train_matches:
                model.fit(train_dicts, fit_options=fit_opts)

            pred = model.predict_match(match.home_team, match.away_team)
            my_prediction = strategy_fn(pred, match.home_team, match.away_team)
            actual = (match.home_goals, match.away_goals)

            pts = _calculate_points(my_prediction, actual, self._rules)
            total_points += pts
            points_per_match.append(pts)

            if my_prediction == actual:
                exact_scores += 1
            elif (
                (my_prediction[0] > my_prediction[1] and actual[0] > actual[1])
                or (my_prediction[0] == my_prediction[1] and actual[0] == actual[1])
                or (my_prediction[0] < my_prediction[1] and actual[0] < actual[1])
            ):
                correct_results += 1

            score_matrices.append(pred.score_matrix.copy())
            actual_score_list.append(actual)

            home_probs.append(pred.home_win_prob)
            home_actuals.append(1.0 if actual[0] > actual[1] else 0.0)

            prediction_history.append({
                "match_id": match.match_id,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "prediction": my_prediction,
                "actual": actual,
                "points": pts,
                "home_win_prob": pred.home_win_prob,
                "draw_prob": pred.draw_prob,
                "away_win_prob": pred.away_win_prob,
            })

        log_loss = calculate_score_matrix_log_loss(score_matrices, actual_score_list)
        brier = _calculate_brier_score(score_matrices, actual_score_list)
        rps = _calculate_rps(score_matrices, actual_score_list)

        home_probs_arr = np.array(home_probs)
        home_actuals_arr = np.array(home_actuals)
        ece = (
            calculate_ece(home_probs_arr, home_actuals_arr)
            if len(home_probs) > 0
            else float("inf")
        )

        return BacktestResult(
            strategy_name=strategy_name,
            tournament=tournament,
            total_points=total_points,
            points_per_match=points_per_match,
            exact_scores=exact_scores,
            correct_results=correct_results,
            prediction_history=prediction_history,
            log_loss=log_loss,
            brier_score=brier,
            calibration_error=ece,
            ranked_probability_score=rps,
        )

    def run_backtest_expanding_window(
        self,
        year: int,
        strategy_fn: Callable[[MatchPrediction, str, str], tuple[int, int]],
        strategy_name: str,
        window_size: int = 50,
        data_file: Path | None = None,
    ) -> BacktestResult:
        tournament = f"World Cup {year}"
        matches = self.load_tournament_data(year, data_file=data_file)

        if len(matches) == 0:
            return BacktestResult(
                strategy_name=strategy_name,
                tournament=tournament,
                total_points=0.0,
                points_per_match=[],
                exact_scores=0,
                correct_results=0,
                prediction_history=[],
                log_loss=float("inf"),
                brier_score=float("inf"),
                calibration_error=float("inf"),
                ranked_probability_score=float("inf"),
            )

        total_points = 0.0
        points_per_match: list[float] = []
        exact_scores = 0
        correct_results = 0
        prediction_history: list[dict] = []

        score_matrices: list[np.ndarray] = []
        actual_score_list: list[tuple[int, int]] = []
        home_probs: list[float] = []
        home_actuals: list[float] = []

        fit_opts = {"maxiter": self.config.fit_maxiter} if self.config.fit_maxiter else None

        for i, match in enumerate(matches):
            window_start = max(0, i - window_size)
            train_matches = matches[window_start:i]
            train_dicts = self._matches_to_dicts(train_matches)

            model = DixonColes(max_goals=self._max_goals)
            if len(train_dicts) >= self.config.min_train_matches:
                model.fit(train_dicts, fit_options=fit_opts)

            pred = model.predict_match(match.home_team, match.away_team)
            my_prediction = strategy_fn(pred, match.home_team, match.away_team)
            actual = (match.home_goals, match.away_goals)

            pts = _calculate_points(my_prediction, actual, self._rules)
            total_points += pts
            points_per_match.append(pts)

            if my_prediction == actual:
                exact_scores += 1
            elif (
                (my_prediction[0] > my_prediction[1] and actual[0] > actual[1])
                or (my_prediction[0] == my_prediction[1] and actual[0] == actual[1])
                or (my_prediction[0] < my_prediction[1] and actual[0] < actual[1])
            ):
                correct_results += 1

            score_matrices.append(pred.score_matrix.copy())
            actual_score_list.append(actual)

            home_probs.append(pred.home_win_prob)
            home_actuals.append(1.0 if actual[0] > actual[1] else 0.0)

            prediction_history.append({
                "match_id": match.match_id,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "prediction": my_prediction,
                "actual": actual,
                "points": pts,
            })

        log_loss = calculate_score_matrix_log_loss(score_matrices, actual_score_list)
        brier = _calculate_brier_score(score_matrices, actual_score_list)
        rps = _calculate_rps(score_matrices, actual_score_list)

        home_probs_arr = np.array(home_probs)
        home_actuals_arr = np.array(home_actuals)
        ece = (
            calculate_ece(home_probs_arr, home_actuals_arr)
            if len(home_probs) > 0
            else float("inf")
        )

        return BacktestResult(
            strategy_name=strategy_name,
            tournament=tournament,
            total_points=total_points,
            points_per_match=points_per_match,
            exact_scores=exact_scores,
            correct_results=correct_results,
            prediction_history=prediction_history,
            log_loss=log_loss,
            brier_score=brier,
            calibration_error=ece,
            ranked_probability_score=rps,
        )

    def run_all_tournaments(
        self,
        strategy_fn: Callable[[MatchPrediction, str, str], tuple[int, int]],
        strategy_name: str,
        data_files: dict[int, Path] | None = None,
    ) -> dict[int, BacktestResult]:
        results: dict[int, BacktestResult] = {}
        for year in self.config.validation_years:
            df = data_files.get(year) if data_files else None
            results[year] = self.run_backtest(year, strategy_fn, strategy_name, data_file=df)
        return results

    def compare_strategies(
        self,
        strategies: dict[str, Callable[[MatchPrediction, str, str], tuple[int, int]]],
        year: int = 2022,
        data_file: Path | None = None,
    ) -> BacktestReport:
        results: dict[str, BacktestResult] = {}
        for name, fn in strategies.items():
            results[name] = self.run_backtest(year, fn, name, data_file=data_file)

        baseline_fn = _always_favorite_closure()
        baseline_result = self.run_backtest(
            year, baseline_fn, "always_favorite", data_file=data_file
        )
        baseline_points = baseline_result.total_points

        relative_performance: dict[str, float] = {}
        for name, result in results.items():
            if baseline_points > 0:
                relative_performance[name] = (
                    (result.total_points - baseline_points) / baseline_points * 100
                )
            else:
                relative_performance[name] = float("inf") if result.total_points > 0 else 0.0

        lines: list[str] = [
            f"Backtest Report - {baseline_result.tournament}",
            "-" * 50,
        ]
        for name, result in results.items():
            rp = relative_performance.get(name, 0.0)
            lines.append(
                f"{name}: {result.total_points:.1f} pts "
                f"({rp:+.1f}% vs baseline), "
                f"Exact: {result.exact_scores}, "
                f"Correct: {result.correct_results}, "
                f"LL: {result.log_loss:.3f}, "
                f"Brier: {result.brier_score:.4f}"
            )

        lines.append("-" * 50)
        lines.append(
            f"Baseline (always_favorite): {baseline_points:.1f} pts, "
            f"Exact: {baseline_result.exact_scores}, "
            f"Correct: {baseline_result.correct_results}"
        )

        return BacktestReport(
            tournament=baseline_result.tournament,
            strategies=results,
            baseline_result=baseline_result,
            relative_performance=relative_performance,
            summary="\n".join(lines),
        )

    def calculate_metrics(
        self,
        predictions: list[np.ndarray],
        actuals: list[tuple[int, int]],
    ) -> dict[str, float]:
        score_matrices = predictions
        actual_score_list = actuals

        ll = calculate_score_matrix_log_loss(score_matrices, actual_score_list)
        brier = _calculate_brier_score(score_matrices, actual_score_list)
        rps = _calculate_rps(score_matrices, actual_score_list)

        home_probs = [float(m.sum(axis=0)[0]) for m in score_matrices]
        home_actuals = [1.0 if h > a else 0.0 for h, a in actual_score_list]
        ece = calculate_ece(np.array(home_probs), np.array(home_actuals))

        return {
            "log_loss": ll,
            "brier_score": brier,
            "ranked_probability_score": rps,
            "calibration_error": ece,
        }

    def calculate_calibration_error(
        self,
        probs: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        return calculate_ece(probs, actuals, n_bins)


def _always_favorite_closure() -> Callable[[MatchPrediction, str, str], tuple[int, int]]:
    def fn(match_pred: MatchPrediction, team_home: str, team_away: str) -> tuple[int, int]:
        return always_favorite_strategy(match_pred, team_home, team_away)
    return fn


def make_optimal_ep_strategy(
    ep_calculator: ExpectedScoreCalculator | None = None,
) -> Callable[[MatchPrediction, str, str], tuple[int, int]]:
    calc = ep_calculator or ExpectedScoreCalculator()

    def fn(match_pred: MatchPrediction, team_home: str, team_away: str) -> tuple[int, int]:
        optimal = calc.find_optimal_prediction(match_pred)
        return (optimal.home_goals, optimal.away_goals)

    return fn


def make_optimal_ep_contrarian_strategy(
    ep_calculator: ExpectedScoreCalculator | None = None,
    ownership_matrix: np.ndarray | None = None,
) -> Callable[[MatchPrediction, str, str], tuple[int, int]]:
    calc = ep_calculator or ExpectedScoreCalculator()

    def fn(match_pred: MatchPrediction, team_home: str, team_away: str) -> tuple[int, int]:
        ow = ownership_matrix
        if ow is None:
            ow = np.zeros((match_pred.score_matrix.shape[0], match_pred.score_matrix.shape[1]))
        optimal = calc.find_optimal_prediction(match_pred, ow)
        return (optimal.home_goals, optimal.away_goals)

    return fn


def make_adaptive_strategy(
    strategy_selector: StrategySelector | None = None,
    current_position: int = 1,
    ownership_matrix: np.ndarray | None = None,
) -> Callable[[MatchPrediction, str, str], tuple[int, int]]:
    selector = strategy_selector or StrategySelector()

    def fn(match_pred: MatchPrediction, team_home: str, team_away: str) -> tuple[int, int]:
        ow = ownership_matrix
        if ow is None:
            ow = np.zeros((match_pred.score_matrix.shape[0], match_pred.score_matrix.shape[1]))
        rec = selector.get_recommendation(match_pred, current_position, 15, ow)
        return (rec.prediction.home_goals, rec.prediction.away_goals)

    return fn
