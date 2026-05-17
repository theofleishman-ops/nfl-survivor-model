import pandas as pd
import pandas.testing as pdt
import pytest

from survivor.path_ev import evaluate_path_ev
from survivor.public_field import (
    choose_public_pick,
    simulate_public_field_paths,
    update_remaining_public_entries,
)


def test_public_entries_cannot_reuse_teams():
    result = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_public_field_probabilities(),
        start_week=1,
        pool_size=24,
        simulations=3,
        random_seed=7,
        public_field_sample_size=24,
        include_entry_paths=True,
    )

    assert not result.entry_paths.empty
    picked = result.entry_paths.dropna(subset=["team"])
    team_counts = picked.groupby(["simulation", "entry_id"])["team"].count()
    unique_counts = picked.groupby(["simulation", "entry_id"])["team"].nunique()
    assert (team_counts == unique_counts).all()


def test_choose_public_pick_respects_used_teams():
    week = _public_field_probabilities()
    week = week[week["week"] == 1].copy()

    pick = choose_public_pick(
        week,
        used_teams={"A", "C", "E"},
        rng=None,
        chalkiness=10,
        randomness=0,
    )

    assert pick is not None
    assert pick["team"] not in {"A", "C", "E"}


def test_ownership_changes_after_popular_team_is_exhausted():
    result = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_public_field_probabilities(),
        start_week=1,
        pool_size=120,
        simulations=50,
        random_seed=11,
        public_field_sample_size=120,
        include_entry_paths=False,
        chalkiness=4,
        randomness=0,
    )

    ownership = result.week_by_week_public_ownership
    a_week_1 = _value_for(ownership, 1, "A", "projected_ownership_pct")
    a_week_2 = _value_for(ownership, 2, "A", "projected_ownership_pct")
    a_burned_week_1 = _value_for(result.team_usage_exhaustion, 1, "A", "burned_pct")

    assert a_burned_week_1 > 0.30
    assert a_week_2 < a_week_1


def test_future_scarcity_emerges_and_late_ownership_concentrates():
    result = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_scarcity_probabilities(),
        start_week=1,
        pool_size=160,
        simulations=60,
        random_seed=17,
        public_field_sample_size=160,
        include_entry_paths=False,
        chalkiness=4,
        randomness=0.02,
    )

    scarcity = result.scarcity_by_week.set_index("week")

    assert scarcity.loc[4, "average_available_teams"] < scarcity.loc[1, "average_available_teams"]
    assert scarcity.loc[4, "scarcity_index"] > scarcity.loc[1, "scarcity_index"]
    assert scarcity.loc[4, "expected_chalk_concentration"] > 0.70


def test_public_field_seed_behavior_is_deterministic():
    kwargs = {
        "schedule_df": pd.DataFrame(),
        "odds_df": pd.DataFrame(),
        "public_picks_df": pd.DataFrame(),
        "team_probabilities_df": _public_field_probabilities(),
        "start_week": 1,
        "pool_size": 80,
        "simulations": 20,
        "random_seed": 2026,
        "public_field_sample_size": 80,
        "include_entry_paths": False,
    }

    first = simulate_public_field_paths(**kwargs)
    second = simulate_public_field_paths(**kwargs)

    pdt.assert_frame_equal(
        first.week_by_week_public_ownership,
        second.week_by_week_public_ownership,
    )
    pdt.assert_frame_equal(first.team_usage_exhaustion, second.team_usage_exhaustion)
    assert (first.pick_matrix == second.pick_matrix).all()


def test_update_remaining_public_entries_applies_outcomes():
    entry_paths = pd.DataFrame(
        [
            {"simulation": 1, "entry_id": 1, "week": 1, "team": "A"},
            {"simulation": 1, "entry_id": 1, "week": 2, "team": "C"},
            {"simulation": 1, "entry_id": 2, "week": 1, "team": "B"},
            {"simulation": 1, "entry_id": 2, "week": 2, "team": "D"},
        ],
    )
    outcomes = pd.DataFrame(
        [
            {"simulation": 1, "week": 1, "winning_team": "A"},
            {"simulation": 1, "week": 2, "winning_team": "D"},
        ],
    )

    updated = update_remaining_public_entries(entry_paths, outcomes)

    entry_1_week_2 = updated[
        (updated["entry_id"] == 1) & (updated["week"] == 2)
    ].iloc[0]
    entry_2_week_2 = updated[
        (updated["entry_id"] == 2) & (updated["week"] == 2)
    ].iloc[0]
    assert bool(entry_1_week_2["alive_before"])
    assert not bool(entry_1_week_2["alive_after"])
    assert not bool(entry_2_week_2["alive_before"])
    assert not bool(entry_2_week_2["alive_after"])


def test_path_ev_changes_when_public_field_simulation_is_enabled():
    path = pd.DataFrame(
        [
            {"week": 1, "team": "A"},
            {"week": 2, "team": "B"},
            {"week": 3, "team": "C"},
            {"week": 4, "team": "D"},
        ],
    )
    schedule = _schedule_from_probabilities(_scarcity_probabilities())
    odds = _odds_from_probabilities(_scarcity_probabilities())
    public_picks = _public_picks_from_probabilities(_scarcity_probabilities())

    independent = evaluate_path_ev(
        path,
        schedule,
        odds,
        public_picks,
        pool_size=300,
        simulations=200,
        random_seed=5,
        use_public_field_simulation=False,
    )
    pathwise = evaluate_path_ev(
        path,
        schedule,
        odds,
        public_picks,
        pool_size=300,
        simulations=200,
        random_seed=5,
        use_public_field_simulation=True,
        public_field_sample_size=120,
    )

    assert pathwise.diagnostics["public_field_path_simulation_enabled"]
    assert pathwise.path_ev != pytest.approx(independent.path_ev)


def _public_field_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _probability_row(1, "W1_A_B", "A", "B", 0.90, 0.62),
            _probability_row(1, "W1_A_B", "B", "A", 0.10, 0.03),
            _probability_row(1, "W1_C_D", "C", "D", 0.72, 0.18),
            _probability_row(1, "W1_C_D", "D", "C", 0.28, 0.03),
            _probability_row(1, "W1_E_F", "E", "F", 0.65, 0.12),
            _probability_row(1, "W1_E_F", "F", "E", 0.35, 0.02),
            _probability_row(1, "W1_G_H", "G", "H", 0.58, 0.00),
            _probability_row(1, "W1_G_H", "H", "G", 0.42, 0.00),
            _probability_row(2, "W2_A_C", "A", "C", 0.88, 0.38),
            _probability_row(2, "W2_A_C", "C", "A", 0.12, 0.02),
            _probability_row(2, "W2_B_D", "B", "D", 0.82, 0.28),
            _probability_row(2, "W2_B_D", "D", "B", 0.18, 0.03),
            _probability_row(2, "W2_E_G", "E", "G", 0.64, 0.18),
            _probability_row(2, "W2_E_G", "G", "E", 0.36, 0.03),
            _probability_row(2, "W2_F_H", "F", "H", 0.55, 0.06),
            _probability_row(2, "W2_F_H", "H", "F", 0.45, 0.02),
            _probability_row(3, "W3_A_D", "A", "D", 0.80, 0.32),
            _probability_row(3, "W3_A_D", "D", "A", 0.20, 0.02),
            _probability_row(3, "W3_B_E", "B", "E", 0.78, 0.30),
            _probability_row(3, "W3_B_E", "E", "B", 0.22, 0.04),
            _probability_row(3, "W3_C_F", "C", "F", 0.70, 0.22),
            _probability_row(3, "W3_C_F", "F", "C", 0.30, 0.04),
            _probability_row(3, "W3_G_H", "G", "H", 0.58, 0.04),
            _probability_row(3, "W3_G_H", "H", "G", 0.42, 0.02),
        ],
    )


def _scarcity_probabilities() -> pd.DataFrame:
    base = _public_field_probabilities()
    week_4 = pd.DataFrame(
        [
            _probability_row(4, "W4_A_E", "A", "E", 0.78, 0.42),
            _probability_row(4, "W4_A_E", "E", "A", 0.22, 0.03),
            _probability_row(4, "W4_B_F", "B", "F", 0.76, 0.34),
            _probability_row(4, "W4_B_F", "F", "B", 0.24, 0.03),
            _probability_row(4, "W4_C_G", "C", "G", 0.62, 0.14),
            _probability_row(4, "W4_C_G", "G", "C", 0.38, 0.02),
            _probability_row(4, "W4_D_H", "D", "H", 0.52, 0.02),
            _probability_row(4, "W4_D_H", "H", "D", 0.48, 0.00),
        ],
    )
    return pd.concat([base, week_4], ignore_index=True)


def _probability_row(
    week: int,
    game_id: str,
    team: str,
    opponent: str,
    win_probability: float,
    public_pick_pct: float,
) -> dict[str, object]:
    return {
        "week": week,
        "game_id": game_id,
        "team": team,
        "opponent": opponent,
        "win_probability": win_probability,
        "projected_public_pick_pct": public_pick_pct,
    }


def _schedule_from_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (week, game_id), game in probabilities.groupby(["week", "game_id"], sort=True):
        teams = game["team"].astype(str).tolist()
        rows.append(
            {
                "week": int(week),
                "game_id": str(game_id),
                "home_team": teams[0],
                "away_team": teams[1],
            },
        )
    return pd.DataFrame(rows)


def _odds_from_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    return probabilities[
        ["week", "game_id", "team", "opponent", "win_probability"]
    ].rename(columns={"win_probability": "no_vig_win_probability"})


def _public_picks_from_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    return probabilities[
        ["week", "team", "projected_public_pick_pct"]
    ].rename(columns={"projected_public_pick_pct": "public_pick_pct"})


def _value_for(df: pd.DataFrame, week: int, team: str, column: str) -> float:
    return float(
        df[
            (df["week"].astype(int) == int(week))
            & (df["team"].astype(str) == team)
        ].iloc[0][column],
    )
