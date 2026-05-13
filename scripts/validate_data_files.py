"""Validate survivor CSV imports against the Phase 4 real-data schemas."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.schemas import (  # noqa: E402
    validate_double_pick_weeks_df,
    validate_entries_df,
    validate_odds_df,
    validate_pool_history_df,
    validate_public_picks_df,
    validate_schedule_df,
)


Validator = Callable[[pd.DataFrame], list[str]]

DATASET_VALIDATORS: dict[str, Validator] = {
    "schedule": validate_schedule_df,
    "odds": validate_odds_df,
    "public_picks": validate_public_picks_df,
    "entries": validate_entries_df,
    "pool_history": validate_pool_history_df,
    "double_pick_weeks": validate_double_pick_weeks_df,
}

REQUIRED_DATASETS = {"schedule", "odds", "public_picks", "entries"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local survivor CSV files against expected schemas.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing season CSVs or template CSVs.",
    )
    args = parser.parse_args()

    return validate_data_dir(args.data_dir)


def validate_data_dir(data_dir: Path) -> int:
    """Validate all known data files in ``data_dir`` and print PASS/FAIL lines."""
    if not data_dir.exists():
        print(f"FAIL data directory not found: {data_dir}")
        return 1

    failed = False
    for dataset, validator in DATASET_VALIDATORS.items():
        path = _find_dataset_file(data_dir, dataset)
        if path is None:
            if dataset in REQUIRED_DATASETS:
                failed = True
                expected = f"{dataset}.csv or {dataset}_template.csv"
                print(f"FAIL {dataset}: missing required file ({expected})")
            else:
                print(f"SKIP {dataset}: optional file not present")
            continue

        errors = _validate_file(path, validator)
        if errors:
            failed = True
            print(f"FAIL {dataset}: {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {dataset}: {path}")

    return 1 if failed else 0


def _find_dataset_file(data_dir: Path, dataset: str) -> Path | None:
    candidates = [data_dir / f"{dataset}.csv", data_dir / f"{dataset}_template.csv"]
    return next((path for path in candidates if path.exists()), None)


def _validate_file(path: Path, validator: Validator) -> list[str]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas includes parser details
        return [f"could not read CSV: {exc}"]

    errors: list[str] = []
    if df.empty:
        errors.append("file has a header but no data rows.")
    errors.extend(validator(df))
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
