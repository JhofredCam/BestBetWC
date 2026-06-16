from src.models.dixon_coles import DixonColes


def test_dixon_coles_initialization() -> None:
    model = DixonColes(max_goals=7)
    assert model.max_goals == 7
    assert model.rho == -0.13


def test_dixon_coles_predict_from_params() -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.5, mu_a=1.0)

    assert pred.score_matrix.shape == (8, 8)
    assert abs(pred.score_matrix.sum() - 1.0) < 1e-6
    assert pred.expected_home_goals == 1.5
    assert pred.expected_away_goals == 1.0


def test_dixon_coles_rho_effect() -> None:
    model = DixonColes(max_goals=7)

    pred_no_rho = model.predict_from_params(lambda_h=1.5, mu_a=1.0, rho=0.0)
    pred_with_rho = model.predict_from_params(lambda_h=1.5, mu_a=1.0, rho=-0.13)

    low_scores_no_rho = (
        pred_no_rho.score_matrix[0, 0]
        + pred_no_rho.score_matrix[1, 0]
        + pred_no_rho.score_matrix[0, 1]
        + pred_no_rho.score_matrix[1, 1]
    )
    low_scores_with_rho = (
        pred_with_rho.score_matrix[0, 0]
        + pred_with_rho.score_matrix[1, 0]
        + pred_with_rho.score_matrix[0, 1]
        + pred_with_rho.score_matrix[1, 1]
    )

    assert low_scores_with_rho != low_scores_no_rho


def test_dixon_coles_fit() -> None:
    model = DixonColes(max_goals=5)

    matches = [
        {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
        {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1},
        {"home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 0},
        {"home_team": "C", "away_team": "B", "home_goals": 1, "away_goals": 1},
        {"home_team": "B", "away_team": "C", "home_goals": 2, "away_goals": 2},
        {"home_team": "C", "away_team": "A", "home_goals": 0, "away_goals": 2},
    ]

    model.fit(matches)

    assert "A" in model.team_attack
    assert "B" in model.team_attack
    assert "C" in model.team_attack
    assert "A" in model.team_defense

    pred = model.predict_match("A", "B")
    assert pred.score_matrix.shape == (6, 6)
    assert abs(pred.score_matrix.sum() - 1.0) < 1e-6


def test_dixon_coles_marginal_distributions() -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.0, mu_a=1.5)

    home_marginal = pred.score_matrix.sum(axis=1)
    away_marginal = pred.score_matrix.sum(axis=0)

    assert abs(home_marginal.sum() - 1.0) < 1e-6
    assert abs(away_marginal.sum() - 1.0) < 1e-6

    expected_home = sum(i * home_marginal[i] for i in range(len(home_marginal)))
    expected_away = sum(j * away_marginal[j] for j in range(len(away_marginal)))

    assert abs(expected_home - 2.0) < 0.2
    assert abs(expected_away - 1.5) < 0.2


def test_dixon_coles_most_likely_score() -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=2.0, mu_a=0.5)

    assert pred.most_likely_score[0] >= pred.most_likely_score[1]
    assert pred.most_likely_score_prob > 0


def test_dixon_coles_symmetric_match() -> None:
    model = DixonColes(max_goals=7)
    pred = model.predict_from_params(lambda_h=1.2, mu_a=1.2)

    assert abs(pred.home_win_prob - pred.away_win_prob) < 0.1
    assert pred.draw_prob > 0.2
