import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from src.config import POLLA_RULES
from src.models.dixon_coles import DixonColes
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategySelector

app = typer.Typer(help="BestBetWC - Optimizador de Pronósticos para Polla Mundialista")
console = Console()


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


if __name__ == "__main__":
    app()
