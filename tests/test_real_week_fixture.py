import subprocess
import sys
from pathlib import Path

import pandas as pd

from survivor.loaders import load_sample_data
from survivor.optimizer import rank_weekly_picks
from survivor.portfolio import optimize_portfolio_for_week
from survivor.schemas import validate_season_data_relationships
from survivor.simulator import simulate_many_seasons


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "data" / "sample" / "real_week_1"
RAW_2026_SCHEDULE = PROJECT_ROOT / "data" / "raw" / "2026" / "schedule.csv"


def test_real_week_fixture_files_validate_and_join():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_data_files.py",
            "--data-dir",
            str(FIXTURE_DIR),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    data = load_sample_data(FIXTURE_DIR)

    assert "PASS odds_vs_schedule" in result.stdout
    assert "PASS public_picks_vs_schedule" in result.stdout
    assert validate_season_data_relationships(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
    ) == []


def test_real_week_fixture_uses_real_2026_week_1_game_ids():
    fixture_schedule = pd.read_csv(FIXTURE_DIR / "schedule.csv")
    raw_schedule = pd.read_csv(RAW_2026_SCHEDULE)
    raw_week_1 = raw_schedule[raw_schedule["week"] == 1]

    assert set(fixture_schedule["game_id"]) == set(raw_week_1["game_id"])
    assert set(pd.read_csv(FIXTURE_DIR / "odds.csv")["game_id"]) == set(
        raw_week_1["game_id"],
    )
    assert set(pd.read_csv(FIXTURE_DIR / "public_picks.csv")["team"]) == set(
        raw_week_1["home_team"],
    ) | set(raw_week_1["away_team"])


def test_real_week_fixture_rankings_run():
    data = load_sample_data(FIXTURE_DIR)

    rankings = rank_weekly_picks(
        week=1,
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        entries_df=data["entries"],
    )

    assert len(rankings) == 32
    assert rankings["game_id"].nunique() == 16
    assert rankings["public_pick_pct"].notna().all()


def test_real_week_fixture_simulations_run():
    data = load_sample_data(FIXTURE_DIR)

    result = simulate_many_seasons(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        entries_df=data["entries"],
        start_week=1,
        simulations=25,
        seed=2026,
    )

    assert result.summary["start_week"] == 1
    assert result.summary["end_week"] == 1
    assert not result.week_summary.empty


def test_real_week_fixture_portfolio_optimizer_runs():
    data = load_sample_data(FIXTURE_DIR)
    rankings = rank_weekly_picks(
        week=1,
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        entries_df=data["entries"],
    )

    allocation, exposure, metrics = optimize_portfolio_for_week(
        week=1,
        rankings_df=rankings,
        entries_df=data["entries"],
        personal_entry_count=40,
        aggression="balanced",
        random_seed=2026,
    )

    assert metrics["active_entries"] == 40
    assert len(allocation) == 40
    assert not exposure.empty


def test_real_week_fixture_cli_smoke(tmp_path):
    weekly = subprocess.run(
        [
            sys.executable,
            "scripts/run_weekly_rankings.py",
            "--week",
            "1",
            "--data-dir",
            str(FIXTURE_DIR),
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
            "--data-dir",
            str(FIXTURE_DIR),
            "--simulations",
            "10",
            "--output-dir",
            str(tmp_path / "simulations"),
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
            "--data-dir",
            str(FIXTURE_DIR),
            "--entries",
            "40",
            "--aggression",
            "balanced",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Week 1 top survivor recommendations" in weekly.stdout
    assert "Monte Carlo survivor simulation from week 1" in simulations.stdout
    assert "Week 1 portfolio optimization" in portfolio.stdout
