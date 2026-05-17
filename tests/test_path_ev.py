import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from survivor.path_ev import (
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
    assert "EV multiple" in report
    assert "Heuristic Ranking Comparison" in report
    assert "Assumptions" in report_path.read_text(encoding="utf-8")


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
