import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from survivor.path_ev import (
    build_forward_game_probabilities,
    evaluate_path_ev,
    generate_candidate_paths,
    optimize_single_entry_path,
)
from survivor.path_ev_reports import (
    build_single_entry_path_ev_report,
    write_single_entry_path_ev_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_path_generation_respects_used_teams():
    schedule = _multiweek_schedule()
    odds = _multiweek_odds()
    public_picks = _multiweek_public_picks()

    paths = generate_candidate_paths(
        schedule,
        odds,
        public_picks,
        start_week=1,
        used_teams={"A"},
        top_k=3,
        beam_width=5,
    )

    assert not paths.empty
    assert "A" not in set(paths["team"])


def test_beam_search_produces_paths():
    paths = generate_candidate_paths(
        _multiweek_schedule(),
        _multiweek_odds(),
        _multiweek_public_picks(),
        start_week=1,
        top_k=2,
        beam_width=4,
    )

    assert paths["path_id"].nunique() > 0
    assert paths["path_id"].nunique() <= 4
    assert paths.groupby("path_id")["week"].nunique().min() == 3


def test_ev_is_higher_for_obviously_stronger_path():
    schedule = _one_week_schedule()
    odds = _one_week_odds(a_probability=1.0, c_probability=0.0)
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "A", "public_pick_pct": 0.5},
            {"week": 1, "team": "C", "public_pick_pct": 0.5},
        ],
    )

    strong = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "A"}]),
        schedule,
        odds,
        public_picks,
        pool_size=100,
        simulations=200,
        random_seed=7,
    )
    weak = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "C"}]),
        schedule,
        odds,
        public_picks,
        pool_size=100,
        simulations=200,
        random_seed=7,
    )

    assert strong.path_ev > weak.path_ev
    assert weak.path_ev == 0


def test_public_ownership_affects_ev():
    schedule = _one_week_schedule()
    odds = _one_week_odds(a_probability=0.55, c_probability=0.55)
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "A", "public_pick_pct": 0.90},
            {"week": 1, "team": "C", "public_pick_pct": 0.10},
        ],
    )

    popular = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "A"}]),
        schedule,
        odds,
        public_picks,
        pool_size=1000,
        simulations=5000,
        random_seed=11,
    )
    contrarian = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "C"}]),
        schedule,
        odds,
        public_picks,
        pool_size=1000,
        simulations=5000,
        random_seed=11,
    )

    assert contrarian.path_ev > popular.path_ev


def test_low_survival_pick_can_have_higher_dollar_ev():
    schedule = pd.DataFrame(
        [{"week": 1, "game_id": "NY", "home_team": "Jets", "away_team": "Giants"}],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "NY", "Jets", "Giants", 0.80),
            _odds_row(1, "NY", "Giants", "Jets", 0.20),
        ],
    )
    public_picks = pd.DataFrame(
        [{"week": 1, "team": "Jets", "public_pick_pct": 1.0}],
    )

    jets = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "Jets"}]),
        schedule,
        odds,
        public_picks,
        pool_size=10,
        simulations=5000,
        random_seed=12,
        entry_fee=10,
        prize_pool=100,
    )
    giants = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "Giants"}]),
        schedule,
        odds,
        public_picks,
        pool_size=10,
        simulations=5000,
        random_seed=12,
        entry_fee=10,
        prize_pool=100,
    )

    assert giants.path_survival_probability < jets.path_survival_probability
    assert giants.ev_dollars > jets.ev_dollars
    assert giants.ev_multiple > 1
    assert giants.expected_edge > 0


def test_single_week_path_ev_uses_contest_equity_fraction_for_low_owned_upset():
    schedule = pd.DataFrame(
        [{"week": 1, "game_id": "NY", "home_team": "Jets", "away_team": "Giants"}],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "NY", "Jets", "Giants", 0.80),
            _odds_row(1, "NY", "Giants", "Jets", 0.20),
        ],
    )
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "Jets", "public_pick_pct": 0.90},
            {"week": 1, "team": "Giants", "public_pick_pct": 0.10},
        ],
    )

    result = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "Giants"}]),
        schedule,
        odds,
        public_picks,
        pool_size=10,
        simulations=10,
        random_seed=1,
        entry_fee=10,
        prize_pool=100,
    )

    assert result.path_ev == pytest.approx(0.20)
    assert result.ev_dollars == pytest.approx(20.0)
    assert result.ev_multiple == pytest.approx(2.0)
    assert result.expected_edge == pytest.approx(1.0)
    assert result.baseline_value == pytest.approx(10.0)
    assert result.ev_multiple_vs_baseline == pytest.approx(2.0)
    assert result.expected_edge_vs_baseline == pytest.approx(1.0)


def test_single_week_path_ev_denominator_uses_outcome_state_survivor_counts():
    schedule = pd.DataFrame(
        [
            {"week": 1, "game_id": "NY", "home_team": "Jets", "away_team": "Giants"},
            {"week": 1, "game_id": "KC", "home_team": "Chiefs", "away_team": "Chargers"},
        ],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "NY", "Jets", "Giants", 0.80),
            _odds_row(1, "NY", "Giants", "Jets", 0.20),
            _odds_row(1, "KC", "Chiefs", "Chargers", 0.50),
            _odds_row(1, "KC", "Chargers", "Chiefs", 0.50),
        ],
    )
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "Jets", "public_pick_pct": 0.50},
            {"week": 1, "team": "Giants", "public_pick_pct": 0.10},
            {"week": 1, "team": "Chiefs", "public_pick_pct": 0.30},
            {"week": 1, "team": "Chargers", "public_pick_pct": 0.10},
        ],
    )

    result = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "Chargers"}]),
        schedule,
        odds,
        public_picks,
        pool_size=10,
        simulations=20000,
        random_seed=7,
        entry_fee=10,
        prize_pool=100,
    )

    assert result.ev_dollars == pytest.approx(11.67, abs=0.15)
    assert result.ev_multiple == pytest.approx(1.167, abs=0.015)
    assert result.expected_edge == pytest.approx(0.167, abs=0.015)


def test_symmetric_random_entry_ev_is_baseline_fair_value():
    schedule = pd.DataFrame(
        [
            {"week": 1, "game_id": "G1", "home_team": "A", "away_team": "B"},
            {"week": 1, "game_id": "G2", "home_team": "C", "away_team": "D"},
        ],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "G1", "A", "B", 0.50),
            _odds_row(1, "G1", "B", "A", 0.50),
            _odds_row(1, "G2", "C", "D", 0.50),
            _odds_row(1, "G2", "D", "C", 0.50),
        ],
    )
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "A", "public_pick_pct": 0.25},
            {"week": 1, "team": "B", "public_pick_pct": 0.25},
            {"week": 1, "team": "C", "public_pick_pct": 0.25},
            {"week": 1, "team": "D", "public_pick_pct": 0.25},
        ],
    )

    result = evaluate_path_ev(
        pd.DataFrame([{"week": 1, "team": "A"}]),
        schedule,
        odds,
        public_picks,
        pool_size=10,
        simulations=2000,
        random_seed=42,
        entry_fee=10,
        prize_pool=100,
    )

    assert result.baseline_value == pytest.approx(10.0)
    assert result.ev_dollars == pytest.approx(10.0, abs=0.05)
    assert result.ev_multiple_vs_baseline == pytest.approx(1.0, abs=0.005)
    assert result.expected_edge_vs_baseline == pytest.approx(0.0, abs=0.005)


def test_available_odds_horizon_ignores_future_fallback_weeks_for_toy_optimizer():
    schedule = pd.DataFrame(
        [
            {"week": 1, "game_id": "NY", "home_team": "Jets", "away_team": "Giants"},
            {"week": 2, "game_id": "KC", "home_team": "Chiefs", "away_team": "Raiders"},
        ],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "NY", "Jets", "Giants", 0.80),
            _odds_row(1, "NY", "Giants", "Jets", 0.20),
        ],
    )
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "Jets", "public_pick_pct": 0.90},
            {"week": 1, "team": "Giants", "public_pick_pct": 0.10},
        ],
    )

    result = optimize_single_entry_path(
        schedule,
        odds,
        public_picks,
        start_week=1,
        pool_size=10,
        simulations=200,
        random_seed=5,
        top_k=2,
        beam_width=4,
        entry_fee=10,
        prize_pool=100,
        path_ev_horizon="available",
    )

    assert result.best_path["week"].tolist() == [1]
    assert result.best_path.iloc[0]["team"] == "Giants"
    assert result.ev_dollars == pytest.approx(20.0)
    assert result.ev_multiple_vs_baseline == pytest.approx(2.0)
    assert result.diagnostics["weeks_using_fallback_probabilities"] == []


def test_forward_probabilities_tag_real_projected_and_fallback_sources():
    schedule = pd.DataFrame(
        [
            {"week": 1, "game_id": "W1_A_B", "home_team": "A", "away_team": "B"},
            {"week": 2, "game_id": "W2_C_D", "home_team": "C", "away_team": "D"},
            {"week": 3, "game_id": "W3_E_F", "home_team": "E", "away_team": "F"},
            {"week": 4, "game_id": "W4_G_H", "home_team": "G", "away_team": "H"},
        ],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "W1_A_B", "A", "B", 0.70),
            _odds_row(1, "W1_A_B", "B", "A", 0.30),
            {"week": 2, "game_id": "W2_C_D", "team": "C", "opponent": "D", "spread": -4.0},
            {"week": 2, "game_id": "W2_C_D", "team": "D", "opponent": "C", "spread": 4.0},
            {"team": "E", "team_strength_score": 0.70},
            {"team": "F", "team_strength_score": 0.45},
        ],
    )

    probabilities = build_forward_game_probabilities(schedule, odds)

    assert len(probabilities) == 8
    assert _source_for(probabilities, 1, "A") == "real_moneyline"
    assert _source_for(probabilities, 2, "C") == "real_spread"
    assert _source_for(probabilities, 3, "E") == "projected_team_strength"
    assert _source_for(probabilities, 4, "G") == "fallback_default"
    game_totals = probabilities.groupby(["week", "game_id"])["win_probability"].sum()
    assert game_totals.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_full_season_horizon_uses_projected_team_strength_without_breaking_available_mode():
    schedule = pd.DataFrame(
        [
            {"week": 1, "game_id": "W1_A_B", "home_team": "A", "away_team": "B"},
            {"week": 2, "game_id": "W2_C_D", "home_team": "C", "away_team": "D"},
        ],
    )
    odds = pd.DataFrame(
        [
            _odds_row(1, "W1_A_B", "A", "B", 0.75),
            _odds_row(1, "W1_A_B", "B", "A", 0.25),
            {"team": "C", "team_strength_score": 0.80},
            {"team": "D", "team_strength_score": 0.35},
        ],
    )
    public_picks = pd.DataFrame(
        [
            {"week": 1, "team": "A", "public_pick_pct": 0.80},
            {"week": 1, "team": "B", "public_pick_pct": 0.20},
        ],
    )

    available = optimize_single_entry_path(
        schedule,
        odds,
        public_picks,
        start_week=1,
        pool_size=100,
        simulations=200,
        random_seed=9,
        top_k=2,
        beam_width=4,
        path_ev_horizon="available",
    )
    full_season = optimize_single_entry_path(
        schedule,
        odds,
        public_picks,
        start_week=1,
        pool_size=100,
        simulations=200,
        random_seed=9,
        top_k=2,
        beam_width=4,
        path_ev_horizon="full-season",
    )

    assert available.best_path["week"].tolist() == [1]
    assert available.diagnostics["weeks_with_projected_odds"] == []
    assert full_season.best_path["week"].tolist() == [1, 2]
    assert full_season.diagnostics["weeks_with_real_odds"] == [1]
    assert full_season.diagnostics["weeks_with_projected_odds"] == [2]
    assert full_season.diagnostics["weeks_using_fallback_probabilities"] == []
    assert "cumulative_path_ev" in full_season.expected_field_size_by_week.columns
    assert "weekly_ev_delta" in full_season.expected_field_size_by_week.columns


def test_projected_ownership_uses_team_popularity_placeholder_and_scarcity():
    schedule = pd.DataFrame(
        [
            {"week": 1, "game_id": "W1_DAL_JAX", "home_team": "DAL", "away_team": "JAX"},
            {"week": 1, "game_id": "W1_SEA_TEN", "home_team": "SEA", "away_team": "TEN"},
        ],
    )
    odds = pd.DataFrame(
        [
            {"team": "DAL", "team_strength_score": 0.50},
            {"team": "JAX", "team_strength_score": 0.50},
            {"team": "SEA", "team_strength_score": 0.50},
            {"team": "TEN", "team_strength_score": 0.50},
        ],
    )

    probabilities = build_forward_game_probabilities(schedule, odds)

    dal = probabilities[probabilities["team"] == "DAL"].iloc[0]
    sea = probabilities[probabilities["team"] == "SEA"].iloc[0]
    assert dal["win_probability"] == pytest.approx(sea["win_probability"])
    assert dal["projected_public_pick_pct"] > sea["projected_public_pick_pct"]
    assert set(probabilities["public_pick_source"]) == {"projected_ownership"}
    assert probabilities["week_alternative_scarcity"].between(0, 1).all()


def test_repeated_seed_is_deterministic():
    schedule = _multiweek_schedule()
    odds = _multiweek_odds()
    public_picks = _multiweek_public_picks()
    path = pd.DataFrame(
        [
            {"week": 1, "team": "A"},
            {"week": 2, "team": "B"},
            {"week": 3, "team": "C"},
        ],
    )

    first = evaluate_path_ev(
        path,
        schedule,
        odds,
        public_picks,
        pool_size=250,
        simulations=500,
        random_seed=2026,
    )
    second = evaluate_path_ev(
        path,
        schedule,
        odds,
        public_picks,
        pool_size=250,
        simulations=500,
        random_seed=2026,
    )

    assert first.path_ev == second.path_ev
    pdt.assert_frame_equal(
        first.survival_probability_by_week,
        second.survival_probability_by_week,
    )
    pdt.assert_frame_equal(
        first.expected_field_size_by_week,
        second.expected_field_size_by_week,
    )


def test_report_generation(tmp_path):
    result = optimize_single_entry_path(
        _multiweek_schedule(),
        _multiweek_odds(),
        _multiweek_public_picks(),
        start_week=1,
        pool_size=250,
        simulations=300,
        random_seed=3,
        top_k=2,
        beam_width=4,
        entry_fee=10,
        prize_pool=2500,
    )

    report = build_single_entry_path_ev_report(result, week=1)
    report_path = write_single_entry_path_ev_report(result, week=1, output_dir=tmp_path)

    assert "Single-Entry Path EV" in report
    assert "EV dollars" in report
    assert "EV multiple vs entry fee" in report
    assert "EV multiple vs baseline fair value" in report
    assert "Horizon Comparison" in report
    assert "Weeks with projected odds" in report
    assert "Cumulative path EV" in report
    assert "Path EV Creation By Week" in report
    assert "Heuristic Ranking Comparison" in report
    assert "Assumptions" in report_path.read_text(encoding="utf-8")


def test_report_does_not_invent_money_metrics_without_value_inputs():
    result = optimize_single_entry_path(
        _multiweek_schedule(),
        _multiweek_odds(),
        _multiweek_public_picks(),
        start_week=1,
        pool_size=250,
        simulations=300,
        random_seed=3,
        top_k=2,
        beam_width=4,
    )

    report = build_single_entry_path_ev_report(result, week=1)

    assert "EV dollars: not provided" in report
    assert "EV multiple vs entry fee: not provided" in report
    assert "EV multiple vs baseline fair value: not provided" in report
    assert "$0.00" not in report
    assert "0.000x" not in report


def test_single_entry_optimizer_cli_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_single_entry_optimizer.py",
            "--season",
            "2026",
            "--week",
            "1",
            "--data-dir",
            "data/sample/real_week_1",
            "--simulations",
            "100",
            "--beam-width",
            "10",
            "--top-k",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report_path = tmp_path / "week_1_single_entry_path_ev.md"
    assert "single-entry path EV optimizer" in result.stdout
    assert "Best current-week pick" in result.stdout
    assert report_path.exists()


def _one_week_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"week": 1, "game_id": "G1", "home_team": "A", "away_team": "B"},
            {"week": 1, "game_id": "G2", "home_team": "C", "away_team": "D"},
        ],
    )


def _one_week_odds(a_probability: float, c_probability: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _odds_row(1, "G1", "A", "B", a_probability),
            _odds_row(1, "G1", "B", "A", 1 - a_probability),
            _odds_row(1, "G2", "C", "D", c_probability),
            _odds_row(1, "G2", "D", "C", 1 - c_probability),
        ],
    )


def _multiweek_schedule() -> pd.DataFrame:
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


def _multiweek_odds() -> pd.DataFrame:
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


def _multiweek_public_picks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"week": 1, "team": "A", "public_pick_pct": 0.55},
            {"week": 1, "team": "C", "public_pick_pct": 0.30},
            {"week": 1, "team": "B", "public_pick_pct": 0.10},
            {"week": 1, "team": "D", "public_pick_pct": 0.05},
            {"week": 2, "team": "B", "public_pick_pct": 0.40},
            {"week": 2, "team": "A", "public_pick_pct": 0.30},
            {"week": 2, "team": "C", "public_pick_pct": 0.20},
            {"week": 2, "team": "D", "public_pick_pct": 0.10},
        ],
    )


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


def _source_for(probabilities: pd.DataFrame, week: int, team: str) -> str:
    row = probabilities[
        (probabilities["week"].astype(int) == int(week))
        & (probabilities["team"].astype(str) == team)
    ].iloc[0]
    return str(row["probability_source"])
