import subprocess
import sys
from pathlib import Path

from survivor.loaders import load_sample_data
from survivor.optimizer import RANKING_COLUMNS, rank_weekly_picks
from survivor.reports import build_weekly_report, write_weekly_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def test_rank_weekly_picks_returns_expected_columns_and_sorted_ranks():
    data = load_sample_data(SAMPLE_DIR)

    rankings = rank_weekly_picks(
        week=1,
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        entries_df=data["entries"],
    )

    for column in RANKING_COLUMNS:
        assert column in rankings.columns
    assert list(rankings["rank"]) == list(range(1, len(rankings) + 1))
    assert rankings["final_score"].is_monotonic_decreasing


def test_build_and_write_weekly_report():
    data = load_sample_data(SAMPLE_DIR)
    rankings = rank_weekly_picks(
        week=1,
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        entries_df=data["entries"],
    )

    report = build_weekly_report(rankings, week=1)
    report_path = write_weekly_report(
        rankings,
        week=1,
        output_dir=PROJECT_ROOT / "outputs" / "reports",
    )

    assert "Top 10 Picks" in report
    assert "Why the Top Pick Ranks First" in report
    assert report_path.read_text(encoding="utf-8").startswith("# Week 1")


def test_cli_smoke_writes_report():
    result = subprocess.run(
        [sys.executable, "scripts/run_weekly_rankings.py", "--week", "1"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report_path = PROJECT_ROOT / "outputs" / "reports" / "week_1_report.md"
    assert "Week 1 top survivor recommendations" in result.stdout
    assert report_path.exists()
