import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from survivor.loaders import load_season_data
from survivor.schemas import (
    validate_double_pick_weeks_df,
    validate_entries_df,
    validate_odds_df,
    validate_pool_history_df,
    validate_public_picks_df,
    validate_schedule_df,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "data" / "raw" / "templates"

TEMPLATE_VALIDATORS = {
    "schedule_template.csv": validate_schedule_df,
    "odds_template.csv": validate_odds_df,
    "public_picks_template.csv": validate_public_picks_df,
    "entries_template.csv": validate_entries_df,
    "pool_history_template.csv": validate_pool_history_df,
    "double_pick_weeks_template.csv": validate_double_pick_weeks_df,
}

SEASON_FILES = [
    "schedule.csv",
    "odds.csv",
    "public_picks.csv",
    "entries.csv",
    "pool_history.csv",
    "double_pick_weeks.csv",
]


def test_template_files_validate_successfully():
    for file_name, validator in TEMPLATE_VALIDATORS.items():
        df = pd.read_csv(TEMPLATE_DIR / file_name)

        assert validator(df) == [], f"{file_name} should validate cleanly"


def test_validate_data_files_script_passes_templates():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_files.py",
            "--data-dir",
            str(TEMPLATE_DIR),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "PASS schedule" in result.stdout
    assert "PASS odds_vs_schedule" in result.stdout
    assert "PASS public_picks_vs_schedule" in result.stdout
    assert "PASS double_pick_weeks" in result.stdout


def test_validate_data_files_script_rejects_non_joining_odds(tmp_path):
    _run_create_workspace(tmp_path, season=2026)
    odds_path = tmp_path / "2026" / "odds.csv"
    odds = pd.read_csv(odds_path)
    odds.loc[0, ["week", "game_id", "home_team", "away_team"]] = [
        3,
        "2026_W03_BAL_AT_KC",
        "KC",
        "BAL",
    ]
    odds.to_csv(odds_path, index=False)

    result = _run_validate_workspace(tmp_path / "2026", check=False)

    assert result.returncode == 1
    assert "FAIL odds_vs_schedule" in result.stdout
    assert "must exist in schedule" in result.stdout


def test_validate_data_files_script_rejects_unscheduled_public_picks(tmp_path):
    _run_create_workspace(tmp_path, season=2026)
    public_picks_path = tmp_path / "2026" / "public_picks.csv"
    public_picks = pd.read_csv(public_picks_path)
    public_picks.loc[0, ["week", "team"]] = [2, "KC"]
    public_picks.to_csv(public_picks_path, index=False)

    result = _run_validate_workspace(tmp_path / "2026", check=False)

    assert result.returncode == 1
    assert "FAIL public_picks_vs_schedule" in result.stdout
    assert "must appear in the schedule" in result.stdout


def test_invalid_public_pick_percentage_fails():
    df = pd.read_csv(TEMPLATE_DIR / "public_picks_template.csv")
    df.loc[0, "public_pick_pct"] = 1.38

    errors = validate_public_picks_df(df)

    assert any("public_pick_pct" in error and "between 0 and 1" in error for error in errors)


def test_public_pick_week_totals_cannot_exceed_one():
    df = pd.read_csv(TEMPLATE_DIR / "public_picks_template.csv")
    df.loc[:, "public_pick_pct"] = [0.40, 0.35, 0.20, 0.10]

    errors = validate_public_picks_df(df)

    assert any("totals cannot exceed 1.0" in error for error in errors)


def test_invalid_moneyline_fails():
    df = pd.read_csv(TEMPLATE_DIR / "odds_template.csv")
    df.loc[0, "home_moneyline"] = 50

    errors = validate_odds_df(df)

    assert any("home_moneyline" in error and "American odds" in error for error in errors)


def test_missing_required_column_fails_clearly():
    df = pd.read_csv(TEMPLATE_DIR / "schedule_template.csv").drop(columns=["game_id"])

    errors = validate_schedule_df(df)

    assert any("missing required columns: game_id" in error for error in errors)


def test_create_real_data_workspace_creates_expected_files(tmp_path):
    result = _run_create_workspace(tmp_path, season=2026)
    season_dir = tmp_path / "2026"

    assert "CREATED" in result.stdout
    for file_name in SEASON_FILES:
        assert (season_dir / file_name).exists()


def test_create_real_data_workspace_does_not_overwrite_by_default(tmp_path):
    _run_create_workspace(tmp_path, season=2026)
    schedule_path = tmp_path / "2026" / "schedule.csv"
    original_text = schedule_path.read_text(encoding="utf-8")
    custom_text = original_text.replace("PHI", "CUSTOM", 1)
    schedule_path.write_text(custom_text, encoding="utf-8")

    result = _run_create_workspace(tmp_path, season=2026)

    assert "SKIPPED" in result.stdout
    assert schedule_path.read_text(encoding="utf-8") == custom_text


def test_load_season_data_works_with_copied_template_workspace(tmp_path):
    _run_create_workspace(tmp_path, season=2026)

    data = load_season_data(season=2026, data_dir=tmp_path)

    assert len(data["schedule_df"]) == 4
    assert len(data["odds_df"]) == 4
    assert len(data["public_picks_df"]) == 4
    assert data["entries_df"]["active"].dtype == bool
    assert data["pool_history_df"] is not None
    assert data["double_pick_weeks_df"] is not None


def test_load_season_data_rejects_non_joining_odds(tmp_path):
    _run_create_workspace(tmp_path, season=2026)
    odds_path = tmp_path / "2026" / "odds.csv"
    odds = pd.read_csv(odds_path)
    odds.loc[0, ["week", "game_id", "home_team", "away_team"]] = [
        3,
        "2026_W03_BAL_AT_KC",
        "KC",
        "BAL",
    ]
    odds.to_csv(odds_path, index=False)

    with pytest.raises(ValueError, match="season data relationships"):
        load_season_data(season=2026, data_dir=tmp_path)


def test_clis_still_work_with_use_sample(tmp_path):
    weekly = subprocess.run(
        [
            sys.executable,
            "scripts/run_weekly_rankings.py",
            "--week",
            "1",
            "--use-sample",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    simulations = subprocess.run(
        [
            sys.executable,
            "scripts/run_simulations.py",
            "--week",
            "1",
            "--simulations",
            "5",
            "--output-dir",
            str(tmp_path / "simulations"),
            "--use-sample",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    portfolio = subprocess.run(
        [
            sys.executable,
            "scripts/run_portfolio_optimizer.py",
            "--week",
            "1",
            "--entries",
            "4",
            "--output-dir",
            str(tmp_path / "reports"),
            "--use-sample",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Week 1 top survivor recommendations" in weekly.stdout
    assert "Monte Carlo survivor simulation from week 1" in simulations.stdout
    assert "Week 1 portfolio optimization" in portfolio.stdout


def _run_create_workspace(tmp_path: Path, season: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/create_real_data_workspace.py",
            "--season",
            str(season),
            "--data-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _run_validate_workspace(
    data_dir: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_files.py",
            "--data-dir",
            str(data_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )
