import csv
from pathlib import Path

from survivor.public_picks import load_public_picks
from survivor.schedule import load_schedule
from survivor.simulator import run_placeholder_simulation


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_sample_csv_files_have_rows():
    sample_files = [
        "schedule_sample.csv",
        "odds_sample.csv",
        "public_picks_sample.csv",
        "entries_sample.csv",
        "pool_history_sample.csv",
    ]

    for file_name in sample_files:
        with (SAMPLE_DIR / file_name).open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert rows, f"{file_name} should contain fake sample rows"


def test_load_schedule_sample():
    rows = load_schedule(SAMPLE_DIR / "schedule_sample.csv")

    assert len(rows) == 3
    assert rows[0]["week"] == 1
    assert rows[0]["game_id"] == "FAKE-2026-W01-001"
    assert rows[0]["home_team"] == "Metro Mammoths"


def test_load_public_picks_sample():
    rows = load_public_picks(SAMPLE_DIR / "public_picks_sample.csv")

    assert len(rows) == 4
    assert rows[0]["week"] == 1
    assert rows[0]["pick_share"] == 0.32


def test_placeholder_simulation_loads_sample_data():
    result = run_placeholder_simulation(
        SAMPLE_DIR / "schedule_sample.csv",
        SAMPLE_DIR / "public_picks_sample.csv",
    )

    assert result == {
        "status": "ok",
        "games_loaded": 3,
        "public_pick_rows_loaded": 4,
    }
