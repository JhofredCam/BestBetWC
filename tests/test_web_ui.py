"""Tests for SPEC-018: Web UI (Streamlit)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import POLLA_RULES
from src.models.dixon_coles import DixonColes, MatchPrediction
from src.optimization.expected_score import ExpectedScoreCalculator
from src.optimization.strategy import StrategyMode, StrategySelector
from src.web import __version__
from src.web.api_client import get_match, get_profiles, get_standings, predict_match
from src.web.components.ep_chart import render_ep_chart
from src.web.components.match_card import render_match_card
from src.web.components.profile_card import render_profile_card
from src.web.components.score_heatmap import render_score_heatmap
from src.web.components.strategy_badge import render_strategy_badge
from src.web.state import get_position, get_total_participants, init_state


class TestPackageInit:
    def test_version(self) -> None:
        assert __version__ == "0.2.0"


class TestStateModule:
    def test_defaults(self) -> None:
        import streamlit as st

        st.session_state.clear()
        init_state()
        assert st.session_state["position"] == 3
        assert st.session_state["total_participants"] == 15
        assert st.session_state["home_lambda"] == 1.5
        assert st.session_state["away_lambda"] == 1.0

    def test_get_position_default(self) -> None:
        import streamlit as st

        st.session_state.clear()
        assert get_position() == 3

    def test_get_total_participants_default(self) -> None:
        import streamlit as st

        st.session_state.clear()
        assert get_total_participants() == 15


class TestComponentImports:
    """Verify all web components are importable."""

    def test_score_heatmap_import(self) -> None:
        assert render_score_heatmap is not None

    def test_ep_chart_import(self) -> None:
        assert render_ep_chart is not None

    def test_match_card_import(self) -> None:
        assert render_match_card is not None

    def test_strategy_badge_import(self) -> None:
        assert render_strategy_badge is not None

    def test_profile_card_import(self) -> None:
        assert render_profile_card is not None


class TestDashboardIntegration:
    """Test the underlying logic used by the dashboard page."""

    def test_dashboard_model_prediction(self) -> None:
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.8, mu_a=1.2)
        assert isinstance(prediction, MatchPrediction)
        assert prediction.score_matrix.shape == (8, 8)
        assert 0 <= prediction.home_win_prob <= 1

    def test_dashboard_ep_ranking(self) -> None:
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.8, mu_a=1.2)
        ep_calc = ExpectedScoreCalculator()
        ranked = ep_calc.rank_all_predictions(prediction)
        assert len(ranked) == 64
        assert ranked[0].ep_total >= ranked[-1].ep_total

    def test_dashboard_strategy_recommendation(self) -> None:
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.8, mu_a=1.2)
        selector = StrategySelector()
        recommendation = selector.get_recommendation(prediction, 3, 15)
        assert recommendation.prediction.ep_total >= 0
        assert recommendation.strategy_mode == StrategyMode.BALANCED
        assert recommendation.risk_score > 0
        assert recommendation.upside_potential >= 0


class TestPredictPageIntegration:
    """Test the underlying logic used by the predict page."""

    def test_predict_top10_ranking(self) -> None:
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.8, mu_a=1.2)
        ep_calc = ExpectedScoreCalculator()
        ranked = ep_calc.rank_all_predictions(prediction)
        top10 = ranked[:10]
        assert len(top10) == 10
        assert top10[0].ep_total >= top10[-1].ep_total
        for r in top10:
            assert r.prob_exact >= 0
            assert r.prob_result >= 0
            assert r.prob_goals_home >= 0
            assert r.prob_goals_away >= 0

    def test_predict_score_matrix_sum_is_one(self) -> None:
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.8, mu_a=1.2)
        assert abs(prediction.score_matrix.sum() - 1.0) < 1e-6


class TestStrategyPageIntegration:
    """Test the underlying logic used by the strategy page."""

    def test_strategy_mode_minimize_risk(self) -> None:
        selector = StrategySelector()
        mode = selector.determine_mode(1, 15)
        assert mode == StrategyMode.MINIMIZE_RISK

    def test_strategy_mode_balanced(self) -> None:
        selector = StrategySelector()
        mode = selector.determine_mode(3, 15)
        assert mode == StrategyMode.BALANCED

    def test_strategy_mode_differentiation(self) -> None:
        selector = StrategySelector()
        mode = selector.determine_mode(8, 15)
        assert mode == StrategyMode.DIFFERENTIATION

    def test_strategy_mode_high_risk(self) -> None:
        selector = StrategySelector()
        mode = selector.determine_mode(15, 15)
        assert mode == StrategyMode.HIGH_RISK

    def test_all_modes_have_descriptions(self) -> None:
        modes_info = {
            StrategyMode.MINIMIZE_RISK: "Minimizar Riesgo",
            StrategyMode.BALANCED: "Balanceado",
            StrategyMode.DIFFERENTIATION: "Diferenciación",
            StrategyMode.HIGH_RISK: "Alto Riesgo",
        }
        for mode in StrategyMode:
            assert mode in modes_info


class TestSimulationPageIntegration:
    """Test the underlying logic used by the simulation page."""

    def test_simulation_output_shape(self) -> None:
        n_sim = 1000
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.5, mu_a=1.0)

        home_goals_sim = np.random.choice(
            len(prediction.home_goals_dist),
            size=n_sim, p=prediction.home_goals_dist,
        )
        away_goals_sim = np.random.choice(
            len(prediction.away_goals_dist),
            size=n_sim, p=prediction.away_goals_dist,
        )

        assert len(home_goals_sim) == n_sim
        assert len(away_goals_sim) == n_sim

    def test_simulation_ep_calculation(self) -> None:
        n_sim = 500
        model = DixonColes(max_goals=POLLA_RULES.max_goals)
        prediction = model.predict_from_params(lambda_h=1.5, mu_a=1.0)

        home_goals_sim = np.random.choice(
            len(prediction.home_goals_dist),
            size=n_sim, p=prediction.home_goals_dist,
        )
        away_goals_sim = np.random.choice(
            len(prediction.away_goals_dist),
            size=n_sim, p=prediction.away_goals_dist,
        )

        ep_calc = ExpectedScoreCalculator()
        top5 = ep_calc.rank_all_predictions(prediction)[:5]

        for r in top5:
            eps = np.zeros(n_sim)
            for s in range(n_sim):
                h = int(home_goals_sim[s])
                a = int(away_goals_sim[s])
                if h == r.home_goals and a == r.away_goals:
                    eps[s] = float(POLLA_RULES.exact_score_pts)
                    eps[s] += float(POLLA_RULES.goals_home_correct_pts)
                    eps[s] += float(POLLA_RULES.goals_away_correct_pts)
                else:
                    if h == r.home_goals:
                        eps[s] += float(POLLA_RULES.goals_home_correct_pts)
                    if a == r.away_goals:
                        eps[s] += float(POLLA_RULES.goals_away_correct_pts)
                    if ((r.home_goals > r.away_goals and h > a)
                        or (r.home_goals == r.away_goals and h == a)
                        or (r.home_goals < r.away_goals and h < a)):
                        eps[s] += float(POLLA_RULES.result_correct_pts)

            assert eps.mean() >= 0
            assert eps.std() >= 0


@pytest.mark.asyncio
class TestApiClient:
    """Test the API client for SPEC-017 communication."""

    async def test_get_match_200(self) -> None:
        mock_response = {
            "id": 1, "home_team": "Brazil", "away_team": "Argentina",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            result = await get_match(1)
            assert result["id"] == 1

    async def test_get_match_404(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 404")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            with pytest.raises(Exception):
                await get_match(99999)

    async def test_predict_match_200(self) -> None:
        mock_response = {
            "home_team": "Brasil",
            "away_team": "Argentina",
            "home_win_prob": 0.45,
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await predict_match("Brasil", "Argentina", 1.5, 1.0, 3)
            assert result["home_team"] == "Brasil"

    async def test_get_standings_empty(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            result = await get_standings()
            assert isinstance(result, list)

    async def test_get_profiles_empty(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            result = await get_profiles()
            assert isinstance(result, list)


class TestPageModulesImportable:
    """Verify all page modules exist and can be compiled."""

    def test_page_files_exist_and_compile(self) -> None:
        import py_compile
        from pathlib import Path

        pages_dir = Path(__file__).parent.parent / "src" / "web" / "pages"
        page_files = sorted(pages_dir.glob("[0-9]_*.py"))
        assert len(page_files) == 6, f"Expected 6 page files, got {len(page_files)}"

        for pf in page_files:
            py_compile.compile(str(pf), doraise=True)
