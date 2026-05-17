import pandas as pd
import pandas.testing as pdt

from scripts.run_public_behavior_calibration import run_public_behavior_calibration
from survivor.public_behavior import (
    PUBLIC_BEHAVIOR_PRESET_ORDER,
    PublicBehaviorConfig,
    get_public_behavior_config,
)
from survivor.public_behavior_reports import (
    build_public_behavior_calibration_report,
    write_public_behavior_calibration_report,
)
from survivor.public_field import simulate_public_field_paths


def test_behavior_presets_load():
    for preset in PUBLIC_BEHAVIOR_PRESET_ORDER:
        config = get_public_behavior_config(preset)

        assert config.name == preset
        assert config.description
        assert config.chalkiness > 0
        assert 0 <= config.randomness <= 1
        assert config.future_awareness >= 0
        assert 0 < config.max_single_team_ownership <= 1
        assert config.ownership_temperature > 0


def test_different_chalkiness_changes_ownership_concentration():
    low_chalk = _behavior("low_chalk", chalkiness=1.1)
    high_chalk = _behavior("high_chalk", chalkiness=6.0)

    low = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_calibration_probabilities(),
        pool_size=160,
        simulations=40,
        random_seed=7,
        public_field_sample_size=160,
        behavior_config=low_chalk,
        include_entry_paths=False,
    )
    high = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_calibration_probabilities(),
        pool_size=160,
        simulations=40,
        random_seed=7,
        public_field_sample_size=160,
        behavior_config=high_chalk,
        include_entry_paths=False,
    )

    assert _week_value(high.scarcity_by_week, 1, "max_projected_ownership_pct") > _week_value(
        low.scarcity_by_week,
        1,
        "max_projected_ownership_pct",
    )


def test_high_future_awareness_burns_elite_teams_more_slowly():
    naive = _behavior("naive", future_awareness=0.0, scarcity_weight=0.0)
    aware = _behavior("aware", future_awareness=3.0, scarcity_weight=2.0)

    naive_result = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_future_burn_probabilities(),
        pool_size=200,
        simulations=50,
        random_seed=11,
        public_field_sample_size=200,
        behavior_config=naive,
        include_entry_paths=False,
    )
    aware_result = simulate_public_field_paths(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        team_probabilities_df=_future_burn_probabilities(),
        pool_size=200,
        simulations=50,
        random_seed=11,
        public_field_sample_size=200,
        behavior_config=aware,
        include_entry_paths=False,
    )

    naive_burn = _team_week_value(naive_result.team_usage_exhaustion, 1, "A", "burned_pct")
    aware_burn = _team_week_value(aware_result.team_usage_exhaustion, 1, "A", "burned_pct")

    assert aware_burn < naive_burn


def test_calibration_runner_outputs_all_presets():
    calibration = run_public_behavior_calibration(
        schedule_df=_schedule(),
        odds_df=_odds(),
        public_picks_df=_public_picks(),
        start_week=1,
        used_teams=None,
        pool_size=120,
        max_paths=None,
        simulations=40,
        random_seed=2026,
        top_k=2,
        beam_width=4,
        entry_fee=10,
        prize_pool=1200,
        team_strength_df=None,
        team_strength_home_field_adjustment=0.0,
        team_strength_scale=1.0,
        public_field_sample_size=80,
        presets=PUBLIC_BEHAVIOR_PRESET_ORDER,
    )

    assert calibration["preset"].tolist() == list(PUBLIC_BEHAVIOR_PRESET_ORDER)
    assert calibration["ev_dollars"].notna().all()
    assert calibration["average_chalk_concentration"].notna().all()


def test_public_behavior_report_generation(tmp_path):
    calibration = run_public_behavior_calibration(
        schedule_df=_schedule(),
        odds_df=_odds(),
        public_picks_df=_public_picks(),
        start_week=1,
        used_teams=None,
        pool_size=120,
        max_paths=None,
        simulations=30,
        random_seed=3,
        top_k=2,
        beam_width=4,
        entry_fee=10,
        prize_pool=1200,
        team_strength_df=None,
        team_strength_home_field_adjustment=0.0,
        team_strength_scale=1.0,
        public_field_sample_size=60,
        presets=("balanced_public", "contrarian_public"),
    )

    report = build_public_behavior_calibration_report(calibration, week=1)
    path = write_public_behavior_calibration_report(calibration, week=1, output_dir=tmp_path)

    assert "Public Behavior Calibration" in report
    assert "Sensitivity Table" in report
    assert "What Drives EV" in report
    assert "Conservative" in report
    assert path.read_text(encoding="utf-8").startswith("# Week 1 Public Behavior Calibration")


def test_calibration_seed_behavior_is_deterministic():
    kwargs = {
        "schedule_df": _schedule(),
        "odds_df": _odds(),
        "public_picks_df": _public_picks(),
        "start_week": 1,
        "used_teams": None,
        "pool_size": 120,
        "max_paths": None,
        "simulations": 30,
        "random_seed": 99,
        "top_k": 2,
        "beam_width": 4,
        "entry_fee": 10,
        "prize_pool": 1200,
        "team_strength_df": None,
        "team_strength_home_field_adjustment": 0.0,
        "team_strength_scale": 1.0,
        "public_field_sample_size": 60,
        "presets": ("balanced_public", "future_aware_public"),
    }

    first = run_public_behavior_calibration(**kwargs)
    second = run_public_behavior_calibration(**kwargs)

    pdt.assert_frame_equal(first, second)


def _behavior(name: str, **overrides: float) -> PublicBehaviorConfig:
    values = {
        "name": name,
        "description": name,
        "chalkiness": 3.0,
        "randomness": 0.0,
        "future_awareness": 0.0,
        "popularity_weight": 1.0,
        "scarcity_weight": 1.0,
        "contrarian_rate": 0.0,
        "max_single_team_ownership": 1.0,
        "ownership_temperature": 1.0,
    }
    values.update(overrides)
    return PublicBehaviorConfig(**values)


def _calibration_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _probability_row(1, "W1_A_B", "A", "B", 0.82, 0.0),
            _probability_row(1, "W1_A_B", "B", "A", 0.18, 0.0),
            _probability_row(1, "W1_C_D", "C", "D", 0.66, 0.0),
            _probability_row(1, "W1_C_D", "D", "C", 0.34, 0.0),
            _probability_row(2, "W2_A_C", "A", "C", 0.70, 0.0),
            _probability_row(2, "W2_A_C", "C", "A", 0.30, 0.0),
            _probability_row(2, "W2_B_D", "B", "D", 0.60, 0.0),
            _probability_row(2, "W2_B_D", "D", "B", 0.40, 0.0),
        ],
    )


def _future_burn_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _probability_row(1, "W1_A_B", "A", "B", 0.76, 0.0),
            _probability_row(1, "W1_A_B", "B", "A", 0.24, 0.0),
            _probability_row(1, "W1_C_D", "C", "D", 0.74, 0.0),
            _probability_row(1, "W1_C_D", "D", "C", 0.26, 0.0),
            _probability_row(2, "W2_A_C", "A", "C", 0.95, 0.0),
            _probability_row(2, "W2_A_C", "C", "A", 0.05, 0.0),
            _probability_row(2, "W2_B_D", "B", "D", 0.60, 0.0),
            _probability_row(2, "W2_B_D", "D", "B", 0.40, 0.0),
        ],
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"week": 1, "game_id": "W1_A_B", "home_team": "A", "away_team": "B"},
            {"week": 1, "game_id": "W1_C_D", "home_team": "C", "away_team": "D"},
            {"week": 2, "game_id": "W2_A_C", "home_team": "A", "away_team": "C"},
            {"week": 2, "game_id": "W2_B_D", "home_team": "B", "away_team": "D"},
            {"week": 3, "game_id": "W3_A_D", "home_team": "A", "away_team": "D"},
            {"week": 3, "game_id": "W3_B_C", "home_team": "B", "away_team": "C"},
        ],
    )


def _odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _odds_row(1, "W1_A_B", "A", "B", 0.80),
            _odds_row(1, "W1_A_B", "B", "A", 0.20),
            _odds_row(1, "W1_C_D", "C", "D", 0.65),
            _odds_row(1, "W1_C_D", "D", "C", 0.35),
            _odds_row(2, "W2_A_C", "A", "C", 0.60),
            _odds_row(2, "W2_A_C", "C", "A", 0.40),
            _odds_row(2, "W2_B_D", "B", "D", 0.75),
            _odds_row(2, "W2_B_D", "D", "B", 0.25),
            _odds_row(3, "W3_A_D", "A", "D", 0.58),
            _odds_row(3, "W3_A_D", "D", "A", 0.42),
            _odds_row(3, "W3_B_C", "B", "C", 0.45),
            _odds_row(3, "W3_B_C", "C", "B", 0.55),
        ],
    )


def _public_picks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"week": 1, "team": "A", "public_pick_pct": 0.55},
            {"week": 1, "team": "C", "public_pick_pct": 0.30},
            {"week": 1, "team": "B", "public_pick_pct": 0.10},
            {"week": 1, "team": "D", "public_pick_pct": 0.05},
        ],
    )


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


def _odds_row(
    week: int,
    game_id: str,
    team: str,
    opponent: str,
    probability: float,
) -> dict[str, object]:
    return {
        "week": week,
        "game_id": game_id,
        "team": team,
        "opponent": opponent,
        "no_vig_win_probability": probability,
    }


def _week_value(df: pd.DataFrame, week: int, column: str) -> float:
    return float(df[df["week"].astype(int) == int(week)].iloc[0][column])


def _team_week_value(df: pd.DataFrame, week: int, team: str, column: str) -> float:
    return float(
        df[
            (df["week"].astype(int) == int(week))
            & (df["team"].astype(str) == team)
        ].iloc[0][column],
    )
