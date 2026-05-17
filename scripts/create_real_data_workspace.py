"""Create a season-specific real-data workspace from checked-in CSV templates."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_DIR = PROJECT_ROOT / "data" / "raw" / "templates"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TEMPLATE_TARGETS = {
    "schedule_template.csv": "schedule.csv",
    "odds_template.csv": "odds.csv",
    "public_picks_template.csv": "public_picks.csv",
    "entries_template.csv": "entries.csv",
    "pool_history_template.csv": "pool_history.csv",
    "double_pick_weeks_template.csv": "double_pick_weeks.csv",
    "team_strength_template.csv": "team_strength.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a real-data CSV workspace for one NFL season.",
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root raw data directory. The season folder is created inside it.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=TEMPLATE_DIR,
        help="Directory containing *_template.csv files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing season CSV files with fresh template copies.",
    )
    args = parser.parse_args()

    try:
        actions = create_real_data_workspace(
            season=args.season,
            data_dir=args.data_dir,
            template_dir=args.template_dir,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    for action, path in actions:
        print(f"{action.upper()} {path}")
    return 0


def create_real_data_workspace(
    season: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    template_dir: Path = TEMPLATE_DIR,
    overwrite: bool = False,
) -> list[tuple[str, Path]]:
    """Copy template CSVs into ``data_dir/season`` without clobbering by default."""
    season_dir = Path(data_dir) / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    actions: list[tuple[str, Path]] = []
    for template_name, target_name in TEMPLATE_TARGETS.items():
        source = Path(template_dir) / template_name
        if not source.exists():
            raise FileNotFoundError(f"template not found: {source}")

        target = season_dir / target_name
        existed_before = target.exists()
        if existed_before and not overwrite:
            actions.append(("skipped", target))
            continue

        shutil.copyfile(source, target)
        actions.append(("overwritten" if existed_before else "created", target))

    return actions


if __name__ == "__main__":
    raise SystemExit(main())
