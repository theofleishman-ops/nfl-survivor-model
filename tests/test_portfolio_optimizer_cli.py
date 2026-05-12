import subprocess
import sys
from pathlib import Path

import pandas as pd

from survivor.portfolio import optimize_portfolio_for_week
from survivor.portfolio_reports import build_portfolio_markdown_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "pytest_portfolio_outputs"


def _rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _team("Atlas", 0.75, 0.35, 0.10, 0.40, 0.40, 1.00),
            _team("Beacon", 0.68, 0.18, 0.30, 0.25, 0.25, 0.92),
            _team("Cipher", 0.61, 0.08, 0.45, 0.10, 0.10, 0.88),
            _team("Delta", 0.54, 0.03, 0.55, 0.05, 0.05, 0.80),
            _team("Ember", 0.48, 0.01, 1.50, 0.02, 0.02, 1.20),
            _team("Forge", 0.44, 0.01, 1.30, 0.01, 0.01, 1.10),
        ],
    )


def _team(
    team: str,
    win_probability: float,
    public_pick_pct: float,
    leverage_score: float,
    future_cost: float,
    scarcity_adjusted_future_cost: float,
    final_score: float,
) -> dict[str, object]:
    return {
        "week": 1,
        "team": team,
        "opponent": f"{team} Opp",
        "no_vig_win_probability": win_probability,
        "public_pick_pct": public_pick_pct,
        "leverage_score": leverage_score,
        "future_cost": future_cost,
        "scarcity_adjusted_future_cost": scarcity_adjusted_future_cost,
        "final_score": final_score,
        "rank": 1,
    }


def _entries(count: int, active: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"entry_id": f"E{number:03d}", "active": active, "used_teams": ""}
            for number in range(1, count + 1)
        ],
    )


def test_inactive_entries_are_excluded():
    entries = pd.DataFrame(
        [
            {"entry_id": "E001", "active": True, "used_teams": ""},
            {"entry_id": "E002", "active": False, "used_teams": ""},
        ],
    )

    allocation, _, metrics = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=entries,
        random_seed=7,
    )

    assert set(allocation["entry_id"]) == {"E001"}
    assert metrics["active_entries"] == 1


def test_used_teams_are_respected():
    entries = pd.DataFrame(
        [
            {"entry_id": "E001", "active": True, "used_teams": "Atlas"},
            {"entry_id": "E002", "active": True, "used_teams": ""},
        ],
    )

    allocation, _, _ = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=entries,
        min_teams=1,
        random_seed=7,
    )

    entry_one_team = allocation.set_index("entry_id").loc["E001", "team"]
    assert entry_one_team != "Atlas"


def test_max_team_exposure_is_enforced():
    allocation, exposure, _ = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=_entries(10),
        max_team_exposure_pct=0.40,
        random_seed=7,
    )

    assert len(allocation) == 10
    assert exposure["entries_allocated"].max() <= 4
    assert exposure["exposure_pct"].max() <= 0.40


def test_minimum_teams_used_when_viable():
    _, exposure, metrics = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=_entries(10),
        min_teams=3,
        random_seed=7,
    )

    assert exposure["team"].nunique() >= 3
    assert metrics["number_of_teams_used"] >= 3


def test_balanced_mode_excludes_sub_50_percent_teams():
    allocation, _, _ = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=_entries(10),
        aggression="balanced",
        random_seed=7,
    )

    assert allocation["no_vig_win_probability"].min() >= 0.50


def test_aggressive_mode_can_take_small_sub_50_percent_exposure():
    allocation, exposure, _ = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=_entries(10),
        aggression="aggressive",
        random_seed=7,
    )

    low_probability_allocations = allocation[
        allocation["no_vig_win_probability"] < 0.50
    ]
    low_probability_teams = set(low_probability_allocations["team"])

    assert not low_probability_allocations.empty
    assert exposure[exposure["team"].isin(low_probability_teams)][
        "entries_allocated"
    ].max() == 1


def test_portfolio_metrics_are_valid_ranges():
    allocation, exposure, metrics = optimize_portfolio_for_week(
        week=1,
        rankings_df=_rankings(),
        entries_df=_entries(10),
        random_seed=7,
    )
    report = build_portfolio_markdown_report(
        week=1,
        allocation_df=allocation,
        exposure_df=exposure,
        metrics=metrics,
        rankings_df=_rankings(),
    )

    assert 0 <= metrics["max_team_exposure"] <= 1
    assert 0 <= metrics["average_win_probability"] <= 1
    assert 0 <= metrics["weighted_public_ownership"] <= 1
    assert 0 <= metrics["concentration_hhi"] <= 1
    assert 0 <= metrics["prob_at_least_one_survives_independent"] <= 1
    assert 0 <= metrics["prob_at_least_one_survives_correlated"] <= 1
    assert "Correlation Risk Notes" in report


def test_portfolio_cli_smoke_writes_report():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_portfolio_optimizer.py",
            "--week",
            "1",
            "--entries",
            "12",
            "--aggression",
            "balanced",
            "--output-dir",
            str(TEST_OUTPUT_DIR),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report_path = TEST_OUTPUT_DIR / "week_1_portfolio_report.md"
    assert "Week 1 portfolio optimization" in result.stdout
    assert report_path.exists()
    assert "Allocation By Team" in report_path.read_text(encoding="utf-8")
