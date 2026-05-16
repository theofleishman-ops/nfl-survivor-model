"""Import and normalize public survivor pick percentages."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import load_schedule_df  # noqa: E402
from survivor.public_pick_ingestion import (  # noqa: E402
    aggregate_public_pick_sources,
    normalize_public_pick_records,
    validate_public_picks_against_schedule,
)
from survivor.public_pick_providers.manual import ManualPublicPickProvider  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    schedule_df = _load_schedule(args.schedule, args.season)
    raw_records = []
    for input_path in args.input:
        provider = ManualPublicPickProvider(
            input_path,
            args.format,
            source=args.source,
            season=args.season,
            week=args.week,
        )
        raw_records.extend(provider.fetch_public_picks())
    normalized = normalize_public_pick_records(raw_records, schedule_df)
    validate_public_picks_against_schedule(
        normalized,
        schedule_df,
        raise_on_error=True,
    )

    output_df = normalized
    output_label = "normalized public pick"
    if args.aggregate:
        output_df = aggregate_public_pick_sources(normalized)
        validate_public_picks_against_schedule(
            output_df,
            schedule_df,
            raise_on_error=True,
        )
        output_label = "consensus public pick"

    _write_or_preview(output_df, args)

    print(f"Imported {len(raw_records)} raw public pick rows.")
    if args.aggregate:
        print(f"Aggregated to {len(output_df)} consensus public pick rows.")
    if args.dry_run:
        print("Dry run complete; no output file was written.")
    elif args.no_write:
        print("No-write mode complete; no output file was written.")
    else:
        print(f"Wrote {len(output_df)} {output_label} rows to {args.output}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import public survivor picks from manual CSV/JSON files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more raw public pick CSV/JSON files.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        required=True,
        help="Input file format.",
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year.")
    parser.add_argument(
        "--week",
        type=int,
        help="NFL week. Used as a default when input rows omit week.",
    )
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
        help="Public picks CSV path to write.",
    )
    parser.add_argument(
        "--source",
        default="manual_import",
        help="Source label to apply when input rows omit source, e.g. yahoo.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate source rows into consensus_public_pick_pct.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize, validate, and print a preview without writing output.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Normalize and validate without writing output.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.dry_run and args.no_write:
        parser.error("--dry-run and --no-write are redundant; choose one.")


def _load_schedule(path: Path, season: int) -> pd.DataFrame:
    schedule_df = load_schedule_df(path)
    if "season" not in schedule_df.columns:
        schedule_df = schedule_df.copy()
        schedule_df.insert(0, "season", season)
    return schedule_df


def _write_or_preview(output_df: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.dry_run:
        preview = output_df.head(10)
        if not preview.empty:
            print(preview.to_string(index=False))
        return
    if args.no_write:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
