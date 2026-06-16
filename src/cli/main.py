import asyncio
import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import API_FOOTBALL_KEY, POLLA_RULES, THE_ODDS_API_KEY
from src.database.connection import get_session
from src.database.models import (
    Participant,
    ParticipantProfile,
    Standing,
    SystemPrediction,
)
from src.etl.api_football import APIFootballClient, extract_world_cup_data
from src.etl.fbref import FBrefScraper
from src.etl.odds_api import CachedOddsClient
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.models.ensemble import ModelEnsemble
from src.models.gradient_boost import GradientBoostModel
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector
from src.simulation.monte_carlo import MonteCarloEngine
from src.simulation.participants import ParticipantSimulator
from src.simulation.tournament import SimulationConfig, TournamentSimulator
from src.validation.backtesting import (
    BacktestConfig,
    BacktestEngine,
    always_favorite_strategy,
    make_adaptive_strategy,
    make_optimal_ep_strategy,
)

app = typer.Typer(help="BestBetWC - Optimizador de Pronósticos para Polla Mundialista")
console = Console()


# =============================================================================
# EXISTING COMMANDS
# =============================================================================


@app.command()
def predict(
    home_team: str = typer.Argument(..., help="Nombre del equipo local"),
    away_team: str = typer.Argument(..., help="Nombre del equipo visitante"),
    home_lambda: float = typer.Option(1.5, help="Goles esperados del local"),
    away_lambda: float = typer.Option(1.0, help="Goles esperados del visitante"),
    position: int = typer.Option(1, help="Posición actual en la polla (1-15)"),
) -> None:
    console.print(f"\n[bold cyan]Análisis: {home_team} vs {away_team}[/bold cyan]\n")

    model = DixonColes(max_goals=POLLA_RULES.max_goals)
    prediction = model.predict_from_params(home_lambda, away_lambda)

    console.print("[bold]Probabilidades de resultado:[/bold]")
    console.print(f"  Victoria {home_team}: {prediction.home_win_prob:.1%}")
    console.print(f"  Empate: {prediction.draw_prob:.1%}")
    console.print(f"  Victoria {away_team}: {prediction.away_win_prob:.1%}")
    most_likely = (
        f"{prediction.most_likely_score[0]}-{prediction.most_likely_score[1]}"
        f" ({prediction.most_likely_score_prob:.1%})"
    )
    console.print(f"\n[bold]Marcador más probable:[/bold] {most_likely}")

    calculator = ExpectedScoreCalculator()
    selector = StrategySelector()

    recommendation = selector.get_recommendation(
        prediction=prediction,
        current_position=position,
        total_participants=POLLA_RULES.num_participants,
    )

    console.print("\n[bold green]RECOMENDACIÓN ÓPTIMA[/bold green]")
    pred_str = (
        f"[bold]{recommendation.prediction.home_goals}"
        f"-{recommendation.prediction.away_goals}[/bold]"
    )
    console.print(f"  Pronóstico: {pred_str}")
    console.print(f"  Expected Points: [bold]{recommendation.prediction.ep_total:.2f} pts[/bold]")
    console.print(f"  Estrategia: {recommendation.strategy_mode.value}")
    console.print(f"  Razón: {recommendation.reasoning}")

    table = Table(title="\nTop 5 Pronósticos por Expected Score")
    table.add_column("#", style="dim")
    table.add_column("Marcador", style="cyan")
    table.add_column("EP Total", justify="right", style="green")
    table.add_column("P(Exacto)", justify="right")
    table.add_column("P(Resultado)", justify="right")
    table.add_column("EP Goles", justify="right")

    all_predictions = calculator.rank_all_predictions(prediction)
    for i, pred in enumerate(all_predictions[:5], 1):
        ep_goals = pred.ep_goals_home + pred.ep_goals_away
        table.add_row(
            str(i),
            f"{pred.home_goals}-{pred.away_goals}",
            f"{pred.ep_total:.2f}",
            f"{pred.prob_exact:.1%}",
            f"{pred.prob_result:.1%}",
            f"{ep_goals:.2f}",
        )

    console.print(table)

    console.print("\n[bold]Métricas de riesgo:[/bold]")
    console.print(f"  Risk Score: {recommendation.risk_score:.2f}")
    console.print(f"  Upside Potential: {recommendation.upside_potential:.2f} pts")
    console.print(f"  Risk of Ruin: {recommendation.risk_of_ruin:.1%}")


@app.command()
def simulate_match(
    home_lambda: float = typer.Option(1.5, help="Goles esperados del local"),
    away_lambda: float = typer.Option(1.0, help="Goles esperados del visitante"),
    simulations: int = typer.Option(10000, help="Número de simulaciones"),
) -> None:
    console.print(
        f"\n[bold cyan]Simulación Monte Carlo ({simulations:,} iteraciones)[/bold cyan]\n"
    )

    model = DixonColes(max_goals=POLLA_RULES.max_goals)
    prediction = model.predict_from_params(home_lambda, away_lambda)

    home_goals_sim = np.random.choice(
        len(prediction.home_goals_dist),
        size=simulations,
        p=prediction.home_goals_dist,
    )
    away_goals_sim = np.random.choice(
        len(prediction.away_goals_dist),
        size=simulations,
        p=prediction.away_goals_dist,
    )

    calculator = ExpectedScoreCalculator()
    top_predictions = calculator.rank_all_predictions(prediction)[:5]

    table = Table(title="Expected Score por Simulación")
    table.add_column("Marcador", style="cyan")
    table.add_column("EP Promedio", justify="right", style="green")
    table.add_column("EP Mínimo", justify="right")
    table.add_column("EP Máximo", justify="right")
    table.add_column("% Acierto Resultado", justify="right")
    table.add_column("% Acierto Exacto", justify="right")

    for pred in top_predictions:
        eps = []
        result_hits = 0
        exact_hits = 0

        for h, a in zip(home_goals_sim, away_goals_sim):
            ep = 0.0
            if h == pred.home_goals and a == pred.away_goals:
                ep += POLLA_RULES.exact_score_pts
                ep += POLLA_RULES.goals_home_correct_pts
                ep += POLLA_RULES.goals_away_correct_pts
                exact_hits += 1
                result_hits += 1
            else:
                if h == pred.home_goals:
                    ep += POLLA_RULES.goals_home_correct_pts
                if a == pred.away_goals:
                    ep += POLLA_RULES.goals_away_correct_pts

                if (pred.home_goals > pred.away_goals and h > a) or \
                   (pred.home_goals == pred.away_goals and h == a) or \
                   (pred.home_goals < pred.away_goals and h < a):
                    ep += POLLA_RULES.result_correct_pts
                    result_hits += 1

            eps.append(ep)

        eps_array = np.array(eps)
        table.add_row(
            f"{pred.home_goals}-{pred.away_goals}",
            f"{eps_array.mean():.2f}",
            f"{eps_array.min():.0f}",
            f"{eps_array.max():.0f}",
            f"{result_hits / simulations:.1%}",
            f"{exact_hits / simulations:.1%}",
        )

    console.print(table)


@app.command()
def info() -> None:
    console.print("\n[bold cyan]BestBetWC - Sistema de Optimización de Pronósticos[/bold cyan]\n")
    console.print("[bold]Reglas de la Polla:[/bold]")
    console.print(f"  Resultado correcto: {POLLA_RULES.result_correct_pts} pts")
    console.print(f"  Marcador exacto: {POLLA_RULES.exact_score_pts} pts (reemplaza resultado)")
    console.print(f"  Goles local correcto: {POLLA_RULES.goals_home_correct_pts} pt")
    console.print(f"  Goles visitante correcto: {POLLA_RULES.goals_away_correct_pts} pt")
    console.print(f"  Bono predicción única: {POLLA_RULES.unique_prediction_bonus} pts")
    console.print("\n[bold]Bonos por Ronda (bracket completo):[/bold]")
    console.print(f"  16avos: {POLLA_RULES.round_bonus_16} pts")
    console.print(f"  8vos: {POLLA_RULES.round_bonus_8} pts")
    console.print(f"  4tos: {POLLA_RULES.round_bonus_4} pts")
    console.print(f"  Semis: {POLLA_RULES.round_bonus_semi} pts")
    console.print(f"  Final: {POLLA_RULES.round_bonus_final} pts")
    console.print(f"\n[bold]Participantes:[/bold] {POLLA_RULES.num_participants}")


# =============================================================================
# NEW COMMANDS (SPEC-016)
# =============================================================================


@app.command()
def predict_full(
    home_team: str = typer.Argument(..., help="Nombre del equipo local"),
    away_team: str = typer.Argument(..., help="Nombre del equipo visitante"),
    position: int = typer.Option(1, help="Posición actual en la polla (1-15)"),
) -> None:
    """Full prediction using ModelEnsemble + feature pipeline + ownership."""
    console.print(
        Panel.fit(
            f"[bold cyan]Predicción Completa: {home_team} vs {away_team}[/bold cyan]",
            border_style="cyan",
        )
    )

    used_ensemble = False
    try:
        ensemble = ModelEnsemble()
        if ensemble._dixon_coles is None and ensemble._gradient_boost is None:
            raise RuntimeError("Ensemble not fitted")

        ensemble_pred = ensemble.predict(home_team, away_team)
        used_ensemble = True
    except Exception:
        console.print(
            "[yellow]Advertencia: Ensemble no disponible, degradando a DixonColes.[/yellow]"
        )
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        ensemble_pred = model.predict_match(home_team, away_team)

    prediction = ensemble_pred

    console.print("\n[bold]Probabilidades de resultado:[/bold]")
    console.print(f"  Victoria {home_team}: {prediction.home_win_prob:.1%}")
    console.print(f"  Empate: {prediction.draw_prob:.1%}")
    console.print(f"  Victoria {away_team}: {prediction.away_win_prob:.1%}")

    most_likely = (
        f"{prediction.most_likely_score[0]}-{prediction.most_likely_score[1]}"
        f" ({prediction.most_likely_score_prob:.1%})"
    )
    console.print(f"\n[bold]Marcador más probable:[/bold] {most_likely}")

    modelo_label = "ModelEnsemble" if used_ensemble else "DixonColes (fallback)"
    console.print(f"\n[bold cyan]Modelo:[/bold] {modelo_label}")
    if used_ensemble:
        weights = ensemble.get_weights_dict()
        w_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items() if v > 0)
        console.print(f"  Pesos: {w_str}")

    calculator = ExpectedScoreCalculator()
    selector = StrategySelector()

    recommendation = selector.get_recommendation(
        prediction=prediction,
        current_position=position,
        total_participants=POLLA_RULES.num_participants,
    )

    console.print("\n[bold green]RECOMENDACIÓN ÓPTIMA[/bold green]")
    pred_str = (
        f"[bold]{recommendation.prediction.home_goals}"
        f"-{recommendation.prediction.away_goals}[/bold]"
    )
    console.print(f"  Pronóstico: {pred_str}")
    console.print(f"  Expected Points: [bold]{recommendation.prediction.ep_total:.2f} pts[/bold]")
    console.print(f"  Estrategia: {recommendation.strategy_mode.value}")
    console.print(f"  Razón: {recommendation.reasoning}")
    console.print(f"  Contrarian Value: {recommendation.prediction.contrarian_value:.3f}")
    console.print(f"  Ownership Est.: {recommendation.prediction.ownership_estimate:.1%}")

    table = Table(title="\nTop 5 Pronósticos por Expected Score")
    table.add_column("#", style="dim")
    table.add_column("Marcador", style="cyan")
    table.add_column("EP Total", justify="right", style="green")
    table.add_column("P(Exacto)", justify="right")
    table.add_column("P(Resultado)", justify="right")
    table.add_column("Contrarian", justify="right")

    all_predictions = calculator.rank_all_predictions(prediction)
    for i, ep_result in enumerate(all_predictions[:5], 1):
        table.add_row(
            str(i),
            f"{ep_result.home_goals}-{ep_result.away_goals}",
            f"{ep_result.ep_total:.2f}",
            f"{ep_result.prob_exact:.1%}",
            f"{ep_result.prob_result:.1%}",
            f"{ep_result.contrarian_value:.3f}",
        )

    console.print(table)

    console.print("\n[bold]Métricas de riesgo:[/bold]")
    console.print(f"  Risk Score: {recommendation.risk_score:.2f}")
    console.print(f"  Upside Potential: {recommendation.upside_potential:.2f} pts")
    console.print(f"  Risk of Ruin: {recommendation.risk_of_ruin:.1%}")


@app.command()
def update(
    source: str = typer.Option(
        "all", "--source", "-s",
        help="Data source to update (odds, football, fbref, all)",
    ),
) -> None:
    """Fetch data from APIs and update the database."""
    valid_sources = {"odds", "football", "fbref", "all"}
    if source not in valid_sources:
        console.print(f"[red]Error: fuente inválida '{source}'. Opciones: {valid_sources}[/red]")
        raise typer.Exit(code=1)

    async def _run_update() -> None:
        results: list[str] = []

        if source in ("odds", "all"):
            try:
                if not THE_ODDS_API_KEY:
                    console.print(
                        "[yellow]THE_ODDS_API_KEY no configurada. Saltando odds.[/yellow]"
                    )
                else:
                    console.print("[cyan]Obteniendo cuotas de The Odds API...[/cyan]")
                    odds_client = CachedOddsClient(THE_ODDS_API_KEY)
                    odds_list, score_list = await odds_client.get_all_odds()
                    await odds_client.close()
                    if odds_list:
                        console.print(
                            f"  [green]Obtenidos {len(odds_list)} odds y "
                            f"{len(score_list)} correct score odds[/green]"
                        )
                    else:
                        console.print(
                            "  [yellow]No se encontraron cuotas disponibles[/yellow]"
                        )
                    results.append(f"odds: {len(odds_list)} snapshots")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    console.print(
                        "  [yellow]Mundial 2026 no disponible aún en The Odds API."
                        " Se activará al acercarse la fecha.[/yellow]"
                    )
                elif e.response.status_code == 401:
                    console.print(
                        "  [red]API key inválida. Revisá THE_ODDS_API_KEY en .env[/red]"
                    )
                else:
                    console.print(f"  [red]Error HTTP {e.response.status_code}[/red]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        if source in ("football", "all"):
            try:
                if not API_FOOTBALL_KEY:
                    console.print(
                        "[yellow]API_FOOTBALL_KEY no configurada."
                        " Saltando football.[/yellow]"
                    )
                else:
                    console.print("[cyan]Obteniendo datos de API-Football...[/cyan]")
                    session = get_session()
                    football_client = APIFootballClient(API_FOOTBALL_KEY)
                    await extract_world_cup_data(football_client, session)
                    await football_client.close()
                    session.close()
                    console.print(
                        "  [green]Datos de equipos y partidos actualizados[/green]"
                    )
                    results.append("football: actualizado")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    console.print(
                        "  [yellow]Límite de requests excedido (100/día). "
                        "Reintentá mañana.[/yellow]"
                    )
                else:
                    console.print(f"  [red]Error HTTP {e.response.status_code}[/red]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        if source in ("fbref", "all"):
            try:
                console.print("[cyan]Obteniendo datos de FBref...[/cyan]")
                scraper = FBrefScraper()
                match_stats = await scraper.scrape_world_cup_matches(2022)
                await scraper.close()
                console.print(
                    f"  [green]Obtenidas {len(match_stats)} "
                    f"estadísticas de partidos[/green]"
                )
                results.append(f"fbref: {len(match_stats)} partidos")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    console.print(
                        "  [yellow]FBref bloqueó la solicitud (anti-bot). "
                        "Reintentá más tarde o usá --source football.[/yellow]"
                    )
                elif e.response.status_code == 404:
                    console.print(
                        "  [yellow]URL no encontrada en FBref.[/yellow]"
                    )
                else:
                    console.print(f"  [red]Error HTTP {e.response.status_code}[/red]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        if results:
            console.print("\n[bold green]Actualización completada:[/bold green]")
            for r in results:
                console.print(f"  [green]OK[/green] {r}")
        else:
            console.print("\n[yellow]No se actualizó ninguna fuente.[/yellow]")

    asyncio.run(_run_update())


@app.command()
def simulate_tournament(
    simulations: int = typer.Option(
        1000, "--simulations", "-n", help="Número de simulaciones"
    ),
    strategy: str = typer.Option(
        "all", "--strategy", "-s",
        help="Strategy to evaluate (optimal, adaptive, conservative, contrarian, all)",
    ),
) -> None:
    """Full tournament Monte Carlo simulation with strategy comparison."""
    valid_strategies = {"optimal", "adaptive", "conservative", "contrarian", "all"}
    if strategy not in valid_strategies:
        console.print(
            f"[red]Error: estrategia inválida '{strategy}'. "
            f"Opciones: {valid_strategies}[/red]"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Simulación de Torneo Completo"
            f" ({simulations:,} iteraciones)[/bold cyan]",
            border_style="cyan",
        )
    )

    config = SimulationConfig(num_simulations=simulations, seed=42)
    simulator = TournamentSimulator(config=config)
    participant_sim = ParticipantSimulator(seed=42)
    engine = MonteCarloEngine(
        tournament_sim=simulator,
        participant_sim=participant_sim,
        config=config,
    )

    def _optimal_fn(
        match_id: str, pred: MatchPrediction | None
    ) -> tuple[int, int]:
        if pred is not None:
            calc = ExpectedScoreCalculator()
            optimal = calc.find_optimal_prediction(pred)
            return (optimal.home_goals, optimal.away_goals)
        return (1, 0)

    def _adaptive_fn(
        match_id: str, pred: MatchPrediction | None
    ) -> tuple[int, int]:
        if pred is not None:
            selector = StrategySelector()
            mid_pos = POLLA_RULES.num_participants // 2
            rec = selector.get_recommendation(
                pred, mid_pos, POLLA_RULES.num_participants
            )
            return (rec.prediction.home_goals, rec.prediction.away_goals)
        return (1, 0)

    def _conservative_fn(
        match_id: str, pred: MatchPrediction | None
    ) -> tuple[int, int]:
        if pred is not None:
            return participant_sim._conservative_strategy(pred)
        return (1, 0)

    def _contrarian_fn(
        match_id: str, pred: MatchPrediction | None
    ) -> tuple[int, int]:
        if pred is not None:
            return participant_sim._aggressive_strategy(pred)
        return (0, 1)

    strats_map: dict[str, Callable[[str, MatchPrediction | None],
                                    tuple[int, int]]] = {
        "optimal": _optimal_fn,
        "adaptive": _adaptive_fn,
        "conservative": _conservative_fn,
        "contrarian": _contrarian_fn,
    }

    if strategy == "all":
        active = dict(strats_map)
    else:
        active = {strategy: strats_map[strategy]}

    dummy_predictions: dict[str, MatchPrediction] = {}
    rng_dc = np.random.default_rng(42)
    for match_id in simulator.match_ids:
        dc = DixonColes(max_goals=POLLA_RULES.max_goals)
        dummy_predictions[match_id] = dc.predict_from_params(
            float(rng_dc.uniform(1.0, 2.0)),
            float(rng_dc.uniform(0.8, 1.5)),
        )

    reports = engine.run_full_simulation(
        match_predictions=dummy_predictions,
        my_strategies=active,
        n_simulations=simulations,
    )

    console.print("\n[bold green]Resultados de Simulación:[/bold green]\n")

    table = Table(title="Reporte de Estrategias")
    table.add_column("Estrategia", style="cyan")
    table.add_column("Media Pts", justify="right", style="green")
    table.add_column("Std Pts", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Top3 %", justify="right")
    table.add_column("Rank Esp.", justify="right")
    table.add_column("Ruin %", justify="right")

    for name, report in reports.items():
        table.add_row(
            name,
            f"{report.mean_points:.1f}",
            f"{report.std_points:.1f}",
            f"{report.win_probability:.1%}",
            f"{report.top3_probability:.1%}",
            f"{report.expected_rank:.1f}",
            f"{report.risk_of_ruin:.1%}",
        )

    console.print(table)

    console.print("\n[bold]Rank Distribution (top strategy):[/bold]")
    if reports:
        top_name = max(reports, key=lambda n: reports[n].mean_points)
        top_report = reports[top_name]
        for pos in range(1, 16):
            prob = top_report.rank_distribution.get(pos, 0.0)
            bar = "=" * int(prob * 50)
            console.print(f"  #{pos:2d}: {bar} {prob:.1%}")


@app.command()
def backtest(
    year: str = typer.Option(
        "all", "--year", "-y",
        help="World Cup year to backtest (2014, 2018, 2022, all)",
    ),
) -> None:
    """Run historical backtesting across World Cup tournaments."""
    valid_years = {"2014", "2018", "2022", "all"}
    if year not in valid_years:
        console.print(f"[red]Error: año inválido '{year}'. "
                      f"Opciones: {valid_years}[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Backtest Histórico - World Cup {year}[/bold cyan]",
            border_style="cyan",
        )
    )

    config = BacktestConfig(
        validation_years=[2014, 2018, 2022],
    )
    engine = BacktestEngine(config=config)

    strategies: dict[str, Callable[
        [MatchPrediction, str, str], tuple[int, int]
    ]] = {
        "optimal_ep": make_optimal_ep_strategy(),
        "always_favorite": always_favorite_strategy,
        "adaptive": make_adaptive_strategy(current_position=7),
    }

    years_to_run = config.validation_years if year == "all" else [int(year)]

    for y in years_to_run:
        console.print(f"\n[bold cyan]--- World Cup {y} ---[/bold cyan]")

        table = Table(title=f"Backtest Results - {y}")
        table.add_column("Estrategia", style="cyan")
        table.add_column("Total Pts", justify="right", style="green")
        table.add_column("Pts/Match", justify="right")
        table.add_column("Exactos", justify="right")
        table.add_column("Correctos", justify="right")
        table.add_column("Log Loss", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("ECE", justify="right")

        for st_name, strat_fn in strategies.items():
            try:
                result = engine.run_backtest(y, strat_fn, st_name)
                if result.points_per_match:
                    ppm = result.total_points / len(result.points_per_match)
                else:
                    ppm = 0.0
                table.add_row(
                    st_name,
                    f"{result.total_points:.1f}",
                    f"{ppm:.2f}",
                    str(result.exact_scores),
                    str(result.correct_results),
                    f"{result.log_loss:.3f}",
                    f"{result.brier_score:.4f}",
                    f"{result.calibration_error:.4f}",
                )
            except FileNotFoundError as e:
                console.print(f"  [yellow]Archivo no encontrado: {e}[/yellow]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        console.print(table)

    if year == "all":
        console.print("\n[bold green]Resumen comparativo:[/bold green]")
        try:
            report = engine.compare_strategies(strategies, year=2022)
            for st_name, rel in report.relative_performance.items():
                color = "green" if rel > 0 else "red"
                console.print(f"  {st_name}: [{color}]{rel:+.1f}%[/{color}] vs baseline")
        except Exception as e:
            console.print(f"  [yellow]No se pudo calcular resumen: {e}[/yellow]")


@app.command()
def standings() -> None:
    """Show current polla standings from the database."""
    console.print(
        Panel.fit("[bold cyan]Clasificación de la Polla[/bold cyan]", border_style="cyan")
    )

    try:
        session = get_session()
    except Exception as e:
        console.print(f"[red]Error conectando a la base de datos: {e}[/red]")
        console.print(
            "[yellow]Asegúrate de que la DB existe "
            "y DATABASE_URL está configurada.[/yellow]"
        )
        raise typer.Exit(code=1)

    try:
        results = (
            session.query(Standing, Participant)
            .join(Participant, Standing.participant_id == Participant.id)
            .order_by(Standing.position)
            .all()
        )
    except Exception as e:
        console.print(f"[red]Error consultando standings: {e}[/red]")
        session.close()
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No hay datos de clasificación en la base de datos.[/yellow]")
        session.close()
        return

    table = Table(title="Standings")
    table.add_column("Pos", style="dim", justify="right")
    table.add_column("Participante", style="cyan")
    table.add_column("Ronda", style="yellow")
    table.add_column("Puntos", justify="right", style="green")

    for standing, participant in results:
        table.add_row(
            str(standing.position),
            participant.name,
            standing.round,
            str(standing.total_points),
        )

    console.print(table)
    session.close()


@app.command()
def profiles(
    participant: str = typer.Option(
        None, "--participant", "-p",
        help="Nombre del participante (omite para mostrar todos)",
    ),
) -> None:
    """Show participant profiles from the database."""
    console.print(
        Panel.fit("[bold cyan]Perfiles de Participantes[/bold cyan]", border_style="cyan")
    )

    try:
        session = get_session()
    except Exception as e:
        console.print(f"[red]Error conectando a la base de datos: {e}[/red]")
        raise typer.Exit(code=1)

    try:
        query = (
            session.query(ParticipantProfile, Participant)
            .join(Participant, ParticipantProfile.participant_id == Participant.id)
        )
        if participant:
            query = query.filter(Participant.name.ilike(f"%{participant}%"))
        results = query.all()
    except Exception as e:
        console.print(f"[red]Error consultando perfiles: {e}[/red]")
        session.close()
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No se encontraron perfiles de participantes.[/yellow]")
        session.close()
        return

    for profile, part in results:
        console.print(f"\n[bold cyan]{part.name}[/bold cyan]")
        console.print(f"  Conservative Score: {profile.conservative_score:.2f}")
        console.print(f"  Aggressive Score: {profile.aggressive_score:.2f}")
        console.print(f"  Market Follower: {profile.market_follower:.2f}")
        console.print(f"  Favorite Bias: {profile.favorite_bias:.2f}")
        console.print(f"  Recency Bias: {profile.recency_bias:.2f}")
        console.print(f"  Home Bias: {profile.home_bias:.2f}")
        console.print(f"  Updated: {profile.updated_at}")

    session.close()


@app.command()
def calibrate(
    year: int = typer.Option(2022, "--year", "-y", help="World Cup year (2018 or 2022)"),
    model: str = typer.Option(
        "all", "--model", "-m",
        help="Model to calibrate (dixon_coles, gradient_boost, all)",
    ),
) -> None:
    """Calibrate models using historical World Cup data."""
    valid_years = {2018, 2022}
    if year not in valid_years:
        console.print(f"[red]Error: año inválido '{year}'. Opciones: {valid_years}[/red]")
        raise typer.Exit(code=1)

    valid_models = {"dixon_coles", "gradient_boost", "all"}
    if model not in valid_models:
        console.print(f"[red]Error: modelo inválido '{model}'. "
                      f"Opciones: {valid_models}[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Calibración de Modelos - World Cup {year}[/bold cyan]",
            border_style="cyan",
        )
    )

    from src.config import RAW_DATA_DIR

    csv_path = RAW_DATA_DIR / f"world_cup_{year}.csv"
    if not csv_path.exists():
        console.print(f"[red]Error: archivo de datos no encontrado: {csv_path}[/red]")
        raise typer.Exit(code=1)

    matches: list[dict[str, int | str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches.append({
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
            })

    console.print(f"  Datos cargados: {len(matches)} partidos")

    if model in ("dixon_coles", "all"):
        console.print("\n[cyan]Calibrando Dixon-Coles...[/cyan]")
        dc = DixonColes(max_goals=POLLA_RULES.max_goals)
        try:
            dc.fit(matches)
            console.print(
                f"  [green]Calibrado: {len(dc.team_attack)} equipos[/green]"
            )
            console.print(f"  Home Advantage: {dc.home_advantage:.4f}")
            console.print(f"  Rho: {dc.rho}")
        except Exception as e:
            console.print(f"  [red]Error calibrando Dixon-Coles: {e}[/red]")

    if model in ("gradient_boost", "all"):
        console.print("\n[cyan]Calibrando GradientBoost...[/cyan]")
        console.print(
            "  [yellow]Nota: GradientBoost requiere features completos. "
            "Cargando con defaults...[/yellow]"
        )
        try:
            from src.features import FEATURE_COLUMNS, MatchFeatureVector

            gb = GradientBoostModel(max_goals=POLLA_RULES.max_goals)
            _x_list: list[np.ndarray] = []
            _y_home_list: list[int] = []
            _y_away_list: list[int] = []

            for m in matches:
                fv = MatchFeatureVector(
                    match_id=0,
                    timestamp=datetime.now(UTC),
                    market_home_prob=0.4,
                    market_draw_prob=0.3,
                    market_away_prob=0.3,
                )
                arr = fv.to_array()
                _x_list.append(arr)
                _y_home_list.append(int(m["home_goals"]))
                _y_away_list.append(int(m["away_goals"]))

            y_home = np.array(_y_home_list if _y_home_list else np.zeros(1, dtype=int))
            y_away = np.array(_y_away_list if _y_home_list else np.zeros(1, dtype=int))

            if len(y_home) > 10:
                from src.models.gradient_boost import temporal_train_test_split

                train_matches = [
                    MatchFeatureVector(
                        match_id=0,
                        timestamp=datetime.now(UTC),
                        market_home_prob=0.4,
                        market_draw_prob=0.3,
                        market_away_prob=0.3,
                    ) for _ in range(len(matches))
                ]
                train, test = temporal_train_test_split(
                    train_matches, train_ratio=0.8
                )

                x_train = np.array([m.to_array() for m in train])
                y_home_train = y_home[:len(train)]
                y_away_train = y_away[:len(train)]
                x_val = np.array([m.to_array() for m in test])
                y_home_val = y_home[len(train):len(train) + len(test)]
                y_away_val = y_away[len(train):len(train) + len(test)]

                gb.fit(
                    x_train, y_home_train, y_away_train,
                    eval_set=(x_val, y_home_val, y_away_val)
                    if len(test) > 0 else None,
                )
                console.print("[green]GradientBoost calibrado[/green]")
                console.print(f"  Features utilizadas: {len(FEATURE_COLUMNS)}")
            else:
                console.print(
                    "  [yellow]Datos insuficientes para GradientBoost"
                    " (min 10 partidos)[/yellow]"
                )
        except Exception as e:
            console.print(f"  [red]Error calibrando GradientBoost: {e}[/red]")

    console.print("\n[bold green]Calibración completada[/bold green]")


@app.command()
def export(
    format: str = typer.Option("csv", "--format", "-f", help="Output format (csv, json)"),
    output: Path = typer.Option(
        Path("predictions_export.csv"), "--output", "-o",
        help="Output file path",
    ),
) -> None:
    """Export predictions from the database to a file."""
    valid_formats = {"csv", "json"}
    if format not in valid_formats:
        console.print(f"[red]Error: formato inválido '{format}'. "
                      f"Opciones: {valid_formats}[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Exportando predicciones a {output} ({format})[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        session = get_session()
    except Exception as e:
        console.print(f"[red]Error conectando a la base de datos: {e}[/red]")
        raise typer.Exit(code=1)

    try:
        predictions = (
            session.query(SystemPrediction)
            .order_by(SystemPrediction.timestamp.desc())
            .all()
        )
    except Exception as e:
        console.print(f"[red]Error consultando predicciones: {e}[/red]")
        session.close()
        raise typer.Exit(code=1)

    if not predictions:
        console.print("[yellow]No hay predicciones para exportar.[/yellow]")
        session.close()
        return

    data = []
    for p in predictions:
        data.append({
            "match_id": p.match_id,
            "timestamp": p.timestamp.isoformat() if p.timestamp else "",
            "home_goals": p.home_goals,
            "away_goals": p.away_goals,
            "ep_score": round(p.ep_score, 4),
            "ownership_estimate": (
                round(p.ownership_estimate, 4) if p.ownership_estimate else None
            ),
            "contrarian_value": (
                round(p.contrarian_value, 4) if p.contrarian_value else None
            ),
            "confidence": round(p.confidence, 4),
            "strategy_mode": p.strategy_mode,
        })

    if format == "json":
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        if data:
            with open(output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    console.print(f"  [green]Exportadas {len(data)} predicciones a {output}[/green]")
    session.close()


if __name__ == "__main__":
    app()
