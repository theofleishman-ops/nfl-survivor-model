"""Normalize a raw schedule CSV into canonical survivor schedule rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import normalize_schedule_df, schedule_df_for_csv  # noqa: E402
from survivor.schemas import validate_or_raise, validate_schedule_df  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a raw NFL schedule CSV to canonical survivor IDs.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw schedule CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the normalized schedule CSV.",
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year.")
    args = parser.parse_args()

    try:
        raw = pd.read_csv(args.input)
        normalized = normalize_schedule_df(raw, season=args.season)
        validate_or_raise("schedule", normalized, validate_schedule_df(normalized))
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule_df_for_csv(normalized).to_csv(args.output, index=False)
    print(f"Wrote {len(normalized)} normalized schedule rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
