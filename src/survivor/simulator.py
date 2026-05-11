"""Placeholder simulation entry points."""

from pathlib import Path

from survivor.public_picks import load_public_picks
from survivor.schedule import load_schedule


def run_placeholder_simulation(
    schedule_path: str | Path,
    public_picks_path: str | Path,
) -> dict[str, int | str]:
    """Prove the starter data flow works by loading sample inputs."""
    schedule_rows = load_schedule(schedule_path)
    public_pick_rows = load_public_picks(public_picks_path)

    return {
        "status": "ok",
        "games_loaded": len(schedule_rows),
        "public_pick_rows_loaded": len(public_pick_rows),
    }
