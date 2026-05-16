"""Import and normalize manual odds exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import load_schedule_df  # noqa: E402
from survivor.odds_ingestion import (  # noqa: E402
    load_odds_csv,
    load_odds_json,
    normalize_odds_records,
    validate_odds_against_schedule,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import raw odds and normalize them to survivor odds schema.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw odds CSV/JSON.")
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        required=True,
        help="Input file format.",
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year.")
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="Canonical schedule CSV to join against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Normalized odds CSV path to write.",
    )
    args = parser.parse_args()

    schedule_df = load_schedule_df(args.schedule)
    if "season" not in schedule_df.columns:
        schedule_df = schedule_df.copy()
        schedule_df.insert(0, "season", args.season)

    records = load_odds_csv(args.input) if args.format == "csv" else load_odds_json(args.input)
    records = _ensure_record_season(records, args.season)
    normalized = normalize_odds_records(records, schedule_df)
    validate_odds_against_schedule(normalized, schedule_df, raise_on_error=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)

    print(f"Imported {len(records)} raw odds rows.")
    print(f"Wrote {len(normalized)} normalized odds rows to {args.output}.")
    return 0


def _ensure_record_season(
    records: list[dict[str, object]],
    season: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        copied = dict(record)
        if _is_blank(copied.get("season")):
            copied["season"] = season
        output.append(copied)
    return output


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and value.strip() == ""


if __name__ == "__main__":
    raise SystemExit(main())
