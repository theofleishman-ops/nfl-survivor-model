import pandas as pd

from survivor.future_value import calculate_future_costs, calculate_week_scarcity


def test_week_scarcity_counts_thresholds_and_scores_scarce_weeks_higher():
    odds = pd.DataFrame(
        [
            {"week": 1, "team": "A", "opponent": "B", "no_vig_win_probability": 0.80},
            {"week": 1, "team": "B", "opponent": "A", "no_vig_win_probability": 0.20},
            {"week": 1, "team": "C", "opponent": "D", "no_vig_win_probability": 0.72},
            {"week": 1, "team": "D", "opponent": "C", "no_vig_win_probability": 0.28},
            {"week": 2, "team": "A", "opponent": "C", "no_vig_win_probability": 0.61},
            {"week": 2, "team": "C", "opponent": "A", "no_vig_win_probability": 0.39},
            {"week": 2, "team": "B", "opponent": "D", "no_vig_win_probability": 0.55},
            {"week": 2, "team": "D", "opponent": "B", "no_vig_win_probability": 0.45},
        ]
    )

    scarcity = calculate_week_scarcity(odds)
    week_one = scarcity[scarcity["week"] == 1].iloc[0]
    week_two = scarcity[scarcity["week"] == 2].iloc[0]

    assert week_one["count_ge_60"] == 2
    assert week_one["count_ge_70"] == 2
    assert week_two["count_ge_60"] == 1
    assert week_two["count_ge_70"] == 0
    assert week_two["week_scarcity_score"] > week_one["week_scarcity_score"]


def test_future_cost_is_higher_for_team_with_scarce_future_value():
    candidates = pd.DataFrame({"team": ["A", "B"]})
    odds = pd.DataFrame(
        [
            {"week": 2, "team": "A", "opponent": "C", "no_vig_win_probability": 0.82},
            {"week": 2, "team": "C", "opponent": "A", "no_vig_win_probability": 0.18},
            {"week": 2, "team": "B", "opponent": "D", "no_vig_win_probability": 0.54},
            {"week": 2, "team": "D", "opponent": "B", "no_vig_win_probability": 0.46},
            {"week": 3, "team": "A", "opponent": "D", "no_vig_win_probability": 0.63},
            {"week": 3, "team": "D", "opponent": "A", "no_vig_win_probability": 0.37},
            {"week": 3, "team": "B", "opponent": "C", "no_vig_win_probability": 0.60},
            {"week": 3, "team": "C", "opponent": "B", "no_vig_win_probability": 0.40},
        ]
    )

    costs = calculate_future_costs(current_week=1, candidates_df=candidates, odds_df=odds)
    by_team = costs.set_index("team")

    assert by_team.loc["A", "future_cost"] > by_team.loc["B", "future_cost"]
    assert by_team.loc["A", "scarcity_adjusted_future_cost"] > 0
    assert by_team.loc["A", "best_future_week"] == 2
