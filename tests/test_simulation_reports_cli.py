import subprocess
import sys
from pathlib import Path

from survivor.loaders import load_sample_data
from survivor.simulation_reports import (
    build_simulation_markdown_report,
    write_simulation_outputs,
)
from survivor.simulator import simulate_many_seasons


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "simulations" / "pytest_simulation_outputs"


def test_simulation_report_generation():
    data = load_sample_data()
    result = simulate_many_seasons(
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        simulations=25,
        seed=99,
    )

    report = build_simulation_markdown_report(result)
    paths = write_simulation_outputs(result, output_dir=TEST_OUTPUT_DIR)

    assert "Survivor Simulation Report" in report
    assert "Expected Remaining Entries By Week" in report
    assert paths["markdown_report"].exists()
    assert paths["week_summary"].exists()
    assert paths["leverage_summary"].exists()
    assert paths["path_summary"].exists()


def test_simulation_cli_smoke():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_simulations.py",
            "--week",
            "1",
            "--simulations",
            "25",
            "--output-dir",
            str(TEST_OUTPUT_DIR),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Monte Carlo survivor simulation from week 1" in result.stdout
    assert (TEST_OUTPUT_DIR / "week_1_simulation_report.md").exists()
    assert (TEST_OUTPUT_DIR / "week_1_week_summary.csv").exists()
