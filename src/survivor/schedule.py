"""Schedule loading utilities."""

import csv
from pathlib import Path


REQUIRED_COLUMNS = {"week", "game_id", "home_team", "away_team"}


def load_schedule(path: str | Path) -> list[dict[str, object]]:
    """Load a schedule CSV into a list of dictionaries."""
    rows = _read_csv(path)
    _require_columns(rows, REQUIRED_COLUMNS, path)

    for row in rows:
        row["week"] = int(row["week"])

    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _require_columns(
    rows: list[dict[str, str]],
    required_columns: set[str],
    path: str | Path,
) -> None:
    if not rows:
        raise ValueError(f"{path} did not contain any rows.")

    missing = required_columns - set(rows[0])
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")
