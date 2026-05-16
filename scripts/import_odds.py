"""Import and normalize odds from manual files or supported API providers."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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
from survivor.odds_providers.the_odds_api import TheOddsAPIProvider  # noqa: E402


ProviderFactory = Callable[..., Any]


def main(
    argv: Sequence[str] | None = None,
    *,
    provider_factory: ProviderFactory | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    schedule_df = _load_schedule(args.schedule, args.season)
    if args.provider == "manual":
        normalized, raw_count = _normalize_manual(args, schedule_df)
    else:
        normalized, raw_count = _normalize_the_odds_api(
            args,
            schedule_df,
            provider_factory=provider_factory,
        )

    validate_odds_against_schedule(normalized, schedule_df, raise_on_error=True)
    _write_or_preview(normalized, args)

    print(f"Imported {raw_count} raw odds rows.")
    if args.dry_run:
        print("Dry run complete; no output file was written.")
    elif args.no_write:
        print("No-write mode complete; no output file was written.")
    else:
        print(f"Wrote {len(normalized)} normalized odds rows to {args.output}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import raw odds and normalize them to survivor odds schema.",
    )
    parser.add_argument(
        "--provider",
        choices=("manual", "the-odds-api"),
        default="manual",
        help="Odds source provider. Defaults to manual CSV/JSON import.",
    )
    parser.add_argument("--input", type=Path, help="Raw odds CSV/JSON for manual import.")
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        help="Manual input file format.",
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year.")
    parser.add_argument("--week", type=int, help="NFL week to import for API providers.")
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
    parser.add_argument(
        "--regions",
        default="us",
        help="The Odds API region list, for example us or us,us2.",
    )
    parser.add_argument(
        "--markets",
        default="h2h,spreads,totals",
        help="The Odds API market list. Supported: h2h,spreads,totals.",
    )
    parser.add_argument(
        "--odds-format",
        choices=("american", "decimal"),
        default="american",
        help="The Odds API odds format. Output is converted to American odds.",
    )
    parser.add_argument(
        "--sportsbook",
        help="Comma-separated The Odds API bookmaker keys such as fanduel,draftkings,betmgm.",
    )
    parser.add_argument(
        "--save-raw",
        type=Path,
        help="Optional explicit path for saving the raw API response for debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and normalize odds, print a small preview, but do not write output.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Normalize and validate odds without writing output.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.provider == "manual":
        if args.input is None:
            parser.error("--input is required for manual odds import.")
        if args.format is None:
            parser.error("--format is required for manual odds import.")
        if args.save_raw is not None:
            parser.error("--save-raw is only supported with --provider the-odds-api.")
    else:
        if args.week is None:
            parser.error("--week is required with --provider the-odds-api.")
        if args.input is not None or args.format is not None:
            parser.error("--input and --format are only used for manual odds import.")
        if args.save_raw is not None:
            try:
                _validate_save_raw_path(args.save_raw)
            except ValueError as exc:
                parser.error(str(exc))


def _load_schedule(path: Path, season: int) -> pd.DataFrame:
    schedule_df = load_schedule_df(path)
    if "season" not in schedule_df.columns:
        schedule_df = schedule_df.copy()
        schedule_df.insert(0, "season", season)
    return schedule_df


def _normalize_manual(
    args: argparse.Namespace,
    schedule_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    records = load_odds_csv(args.input) if args.format == "csv" else load_odds_json(args.input)
    records = _ensure_record_season(records, args.season)
    normalized = normalize_odds_records(records, schedule_df)
    return normalized, len(records)


def _normalize_the_odds_api(
    args: argparse.Namespace,
    schedule_df: pd.DataFrame,
    *,
    provider_factory: ProviderFactory | None,
) -> tuple[pd.DataFrame, int]:
    factory = provider_factory or TheOddsAPIProvider
    provider = factory(
        season=args.season,
        week=args.week,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
        bookmakers=args.sportsbook,
        save_raw_path=args.save_raw,
    )
    if getattr(provider, "unsupported_markets", ()):
        print(
            "Ignoring unsupported The Odds API markets: "
            + ", ".join(getattr(provider, "unsupported_markets")),
        )
    normalized = provider.normalize(schedule_df)
    return normalized, len(normalized)


def _write_or_preview(normalized: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.dry_run:
        preview = normalized.head(10)
        if not preview.empty:
            print(preview.to_string(index=False))
        return
    if args.no_write:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)


def _validate_save_raw_path(path: Path) -> None:
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    allowed_untracked_dirs = [
        project_root / ".odds-api-raw",
        project_root / "outputs",
        project_root / "data" / "raw",
    ]
    tests_fixture_dir = project_root / "tests" / "fixtures"

    if _is_relative_to(resolved, tests_fixture_dir):
        return
    if any(_is_relative_to(resolved, directory) for directory in allowed_untracked_dirs):
        if _is_relative_to(resolved, project_root / "data" / "raw" / "templates"):
            raise ValueError(
                "--save-raw must not write under data/raw/templates; choose an ignored "
                "debug directory such as .odds-api-raw/.",
            )
        return
    raise ValueError(
        "--save-raw must be under an ignored local directory "
        "(.odds-api-raw/, outputs/, data/raw/) or under tests/fixtures for sanitized "
        "mock fixtures.",
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


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
