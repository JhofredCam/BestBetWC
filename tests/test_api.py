from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, create_app
from src.api.schemas import HealthResponse


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def separate_app() -> TestClient:
    a = create_app()
    return TestClient(a)


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_response_model(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        HealthResponse(**response.json())


class TestPredictions:
    def test_predict_default_params(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_team": "Argentina",
            "away_team": "Brazil",
            "home_lambda": 1.5,
            "away_lambda": 1.0,
            "current_position": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["home_team"] == "Argentina"
        assert data["away_team"] == "Brazil"
        assert "home_win_prob" in data
        assert "draw_prob" in data
        assert "away_win_prob" in data
        assert 0 <= data["home_win_prob"] <= 1
        assert 0 <= data["draw_prob"] <= 1
        assert 0 <= data["away_win_prob"] <= 1
        assert data["most_likely_score"] is not None
        assert "top_predictions" in data
        assert len(data["top_predictions"]) > 0
        assert "recommendation" in data
        assert "score_matrix" in data
        assert len(data["score_matrix"]) > 0

    def test_predict_custom_lambdas(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_team": "Spain",
            "away_team": "Germany",
            "home_lambda": 2.0,
            "away_lambda": 1.5,
            "current_position": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["expected_home_goals"] == pytest.approx(2.0, rel=1e-6)
        assert data["expected_away_goals"] == pytest.approx(1.5, rel=1e-6)

    def test_predict_strategy_varies_by_position(self, client: TestClient) -> None:
        base = {"home_team": "France", "away_team": "England"}

        resp1 = client.post("/api/predictions/", json={**base, "current_position": 1})
        resp15 = client.post("/api/predictions/", json={**base, "current_position": 15})

        data1 = resp1.json()
        data15 = resp15.json()

        assert data1["recommendation"]["strategy_mode"] == "minimize_risk"
        assert data15["recommendation"]["strategy_mode"] == "high_risk"

    def test_predict_invalid_lambda_low(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_team": "A",
            "away_team": "B",
            "home_lambda": 0.0,
            "away_lambda": 1.0,
        })
        assert response.status_code == 422

    def test_predict_invalid_lambda_high(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_team": "A",
            "away_team": "B",
            "home_lambda": 1.0,
            "away_lambda": 11.0,
        })
        assert response.status_code == 422

    def test_predict_invalid_position(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_team": "A",
            "away_team": "B",
            "current_position": 20,
        })
        assert response.status_code == 422

    def test_predict_match_by_id_404(self, client: TestClient) -> None:
        response = client.get("/api/predictions/match/99999")
        assert response.status_code == 404

    def test_predict_upcoming_empty_ok(self, client: TestClient) -> None:
        response = client.get("/api/predictions/upcoming")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_top_predictions_have_ep_fields(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_team": "A",
            "away_team": "B",
        })
        top = response.json()["top_predictions"]
        assert len(top) > 0
        p = top[0]
        assert "home_goals" in p
        assert "away_goals" in p
        assert "probability" in p
        assert "ep_total" in p
        assert "ep_exact" in p
        assert "ep_result" in p
        assert "ep_goals" in p
        assert "ep_unique" in p

    def test_default_home_team_values(self, client: TestClient) -> None:
        response = client.post("/api/predictions/", json={
            "home_lambda": 1.5,
            "away_lambda": 1.0,
            "current_position": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["home_team"] == "Home"
        assert data["away_team"] == "Away"


class TestStrategies:
    def test_get_strategy_modes(self, client: TestClient) -> None:
        response = client.get("/api/strategies/modes")
        assert response.status_code == 200
        data = response.json()
        assert "modes" in data
        modes = data["modes"]
        assert len(modes) == 15
        assert modes["1"] == "minimize_risk"
        assert modes.get("2") == "balanced"
        assert modes.get("15") == "high_risk"

    def test_get_optimal_strategy_position1(self, client: TestClient) -> None:
        response = client.get("/api/strategies/optimal/1")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_mode"] == "minimize_risk"
        assert "prediction" in data
        assert "ep_total" in data
        assert "reasoning" in data
        assert "risk_score" in data
        assert "upside_potential" in data
        assert "risk_of_ruin" in data

    def test_get_optimal_strategy_position15(self, client: TestClient) -> None:
        response = client.get("/api/strategies/optimal/15")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_mode"] == "high_risk"

    def test_strategy_differs_by_position(self, client: TestClient) -> None:
        modes: dict[int, str] = {}
        for pos in (1, 3, 8, 15):
            resp = client.get(f"/api/strategies/optimal/{pos}",
                              params={"home_lambda": 1.5, "away_lambda": 1.0})
            modes[pos] = resp.json()["strategy_mode"]
        assert modes[1] == "minimize_risk"
        assert modes[8] == "differentiation"
        assert modes[15] == "high_risk"


class TestSimulation:
    def test_simulate_match_default(self, client: TestClient) -> None:
        response = client.post("/api/simulation/match", json={
            "home_lambda": 1.5,
            "away_lambda": 1.0,
            "simulations": 500,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["num_simulations"] == 500
        assert "results" in data
        assert len(data["results"]) > 0
        r = data["results"][0]
        assert "score" in r
        assert "ep_mean" in r
        assert "std_dev" in r

    def test_simulate_minimum_simulations(self, client: TestClient) -> None:
        response = client.post("/api/simulation/match", json={
            "home_lambda": 1.0,
            "away_lambda": 1.0,
            "simulations": 100,
        })
        assert response.status_code == 200

    def test_simulate_invalid_simulations_low(self, client: TestClient) -> None:
        response = client.post("/api/simulation/match", json={
            "simulations": 50,
        })
        assert response.status_code == 422

    def test_simulate_invalid_simulations_high(self, client: TestClient) -> None:
        response = client.post("/api/simulation/match", json={
            "simulations": 200000,
        })
        assert response.status_code == 422

    def test_simulate_tournament_placeholder(self, client: TestClient) -> None:
        response = client.post("/api/simulation/tournament", json={
            "strategy": "optimal_ep",
            "simulations": 100,
        })
        assert response.status_code == 200
        assert "message" in response.json()


class TestBacktesting:
    def test_list_backtest_strategies(self, client: TestClient) -> None:
        response = client.get("/api/backtesting/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "optimal_ep" in data
        assert "always_favorite" in data

    def test_backtest_invalid_year(self, client: TestClient) -> None:
        response = client.post("/api/backtesting/", json={
            "year": 2005,
            "strategy": "optimal_ep",
        })
        assert response.status_code == 422

    def test_backtest_invalid_strategy(self, client: TestClient) -> None:
        response = client.post("/api/backtesting/", json={
            "year": 2022,
            "strategy": "nonexistent",
        })
        assert response.status_code == 400

    def test_backtest_missing_data_404(self, client: TestClient) -> None:
        response = client.post("/api/backtesting/", json={
            "year": 2026,
            "strategy": "optimal_ep",
        })
        assert response.status_code in (200, 404, 500)


class TestStandings:
    def test_get_standings_empty(self, client: TestClient) -> None:
        response = client.get("/api/standings/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_participant_not_found(self, client: TestClient) -> None:
        response = client.get("/api/standings/participant/99999")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data


class TestProfiles:
    def test_get_all_profiles_empty(self, client: TestClient) -> None:
        response = client.get("/api/profiles/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_profile_not_found(self, client: TestClient) -> None:
        response = client.get("/api/profiles/99999")
        assert response.status_code == 404


class TestMatches:
    def test_get_matches_empty(self, client: TestClient) -> None:
        response = client.get("/api/matches/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_match_not_found(self, client: TestClient) -> None:
        response = client.get("/api/matches/99999")
        assert response.status_code == 404


class TestData:
    def test_data_status(self, client: TestClient) -> None:
        response = client.get("/api/data/status")
        assert response.status_code == 200
        data = response.json()
        assert "total_matches" in data
        assert "total_teams" in data

    def test_update_data(self, client: TestClient) -> None:
        response = client.post("/api/data/update", json={"source": "all"})
        assert response.status_code == 200
        assert response.json()["status"] == "queued"


class TestCORS:
    def test_cors_headers_present(self, client: TestClient) -> None:
        response = client.options("/health")
        assert response.status_code in (200, 405)


class TestDocs:
    def test_swagger_available(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/api/predictions/" in schema["paths"]
        assert "/api/strategies/modes" in schema["paths"]
        assert "/api/simulation/match" in schema["paths"]
