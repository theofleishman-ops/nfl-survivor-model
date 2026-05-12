import pandas as pd
import pandas.testing as pdt

from survivor.loaders import load_sample_data
from survivor.simulator import (
    PersonalEntryState,
    simulate_game_outcomes,
    simulate_many_seasons,
    simulate_week,
)


def _one_game_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week": 1,
                "game_id": "G1",
                "team": "Alpha",
                "opponent": "Beta",
                "is_home": True,
                "no_vig_win_probability": 1.0,
            },
            {
                "week": 1,
                "game_id": "G1",
                "team": "Beta",
                "opponent": "Alpha",
                "is_home": False,
                "no_vig_win_probability": 0.0,
            },
        ]
    )


def test_simulation_seed_reproducibility():
    data = load_sample_data()

    first = simulate_many_seasons(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        simulations=100,
        seed=42,
    )
    second = simulate_many_seasons(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        simulations=100,
        seed=42,
    )

    pdt.assert_frame_equal(first.week_summary, second.week_summary)
    pdt.assert_frame_equal(first.leverage_summary, second.leverage_summary)
    assert first.summary == second.summary


def test_game_outcome_probability_sanity():
    rows = []
    for game_number in range(5000):
        rows.extend(
            [
                {
                    "week": 1,
                    "game_id": f"G{game_number}",
                    "team": "Favorite",
                    "opponent": "Dog",
                    "is_home": True,
                    "no_vig_win_probability": 0.8,
                },
                {
                    "week": 1,
                    "game_id": f"G{game_number}",
                    "team": "Dog",
                    "opponent": "Favorite",
                    "is_home": False,
                    "no_vig_win_probability": 0.2,
                },
            ]
        )
    odds = pd.DataFrame(rows)

    outcomes = simulate_game_outcomes(odds, seed=123)
    favorite_win_rate = (outcomes["winning_team"] == "Favorite").mean()

    assert 0.77 <= favorite_win_rate <= 0.83


def test_week_elimination_and_contest_equity():
    odds = _one_game_odds()
    public_picks = pd.DataFrame(
        [{"week": 1, "team": "Beta", "public_pick_pct": 1.0}]
    )
    rankings = pd.DataFrame(
        [
            {
                "week": 1,
                "team": "Alpha",
                "no_vig_win_probability": 1.0,
                "public_pick_pct": 0.0,
                "final_score": 1.0,
                "rank": 1,
            }
        ]
    )
    personal_states = [PersonalEntryState("Mine")]

    result = simulate_week(
        week=1,
        odds_df=odds,
        public_picks_df=public_picks,
        rankings_df=rankings,
        public_entries=10,
        personal_states=personal_states,
        seed=7,
    )

    assert result.public_entries_survived == 0
    assert result.public_entries_eliminated == 10
    assert result.personal_entries_survived == 1
    assert result.contest_equity == 1.0
    assert personal_states[0].alive


def test_field_size_decreases_over_time_and_probabilities_are_valid():
    data = load_sample_data()

    result = simulate_many_seasons(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        simulations=200,
        pool_size=1000,
        personal_entry_count=2,
        seed=2026,
    )

    public_entries = result.week_summary["expected_public_entries"].tolist()
    assert public_entries == sorted(public_entries, reverse=True)
    assert result.week_summary[
        "probability_at_least_one_personal_survives"
    ].between(0, 1).all()
    assert result.week_summary["expected_contest_equity"].between(0, 1).all()
    assert 0 <= result.summary["probability_at_least_one_personal_survives"] <= 1
    assert 0 <= result.summary["expected_contest_equity"] <= 1


def test_leverage_summary_contains_conditional_upset_metrics():
    data = load_sample_data()

    result = simulate_many_seasons(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        simulations=100,
        seed=11,
    )

    expected_columns = {
        "avg_field_eliminated_if_team_loses",
        "avg_field_shrink_pct_if_team_loses",
        "contest_equity_lift_if_team_loses",
        "uniqueness_value",
        "expected_upset_equity_gain",
    }
    assert expected_columns.issubset(result.leverage_summary.columns)
    assert result.leverage_summary["uniqueness_value"].between(0, 1).all()
