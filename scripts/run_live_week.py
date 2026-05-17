"""Run the operator-facing live weekly survivor workflow."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.config_runtime import (  # noqa: E402
    DEFAULT_RUNTIME_CONFIG,
    csv_text,
    parse_csv_option,
)
from survivor.live_week import (  # noqa: E402
    LiveWeekOptions,
    LiveWeekWorkflowError,
    run_live_week,
)
from survivor.odds_providers.the_odds_api import TheOddsAPIProvider  # noqa: E402


def main(
    argv: Sequence[str] | None = None,
    *,
    odds_provider_factory: type[TheOddsAPIProvider] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    options = _options_from_args(args)

    try:
        result = run_live_week(
            options,
            odds_provider_factory=odds_provider_factory,
        )
    except LiveWeekWorkflowError as exc:
        print("FAIL live weekly workflow", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print(result.summary_text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_RUNTIME_CONFIG
    parser = argparse.ArgumentParser(
        description="Run the live weekly survivor operations workflow.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=defaults.season,
        help="NFL season year.",
    )
    parser.add_argument("--week", type=int, required=True, help="NFL week to run.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=defaults.data_dir,
        help="Raw data root or a directory containing schedule/odds/public_picks/entries CSVs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=defaults.reports_dir,
        help="Directory for generated markdown reports.",
    )
    parser.add_argument(
        "--refresh-odds",
        action="store_true",
        help="Refresh odds from The Odds API before running the workflow.",
    )
    parser.add_argument(
        "--sportsbooks",
        default=csv_text(defaults.sportsbooks),
        help="Comma-separated The Odds API bookmaker keys.",
    )
    parser.add_argument(
        "--odds-regions",
        default=csv_text(defaults.odds_regions),
        help="Comma-separated The Odds API regions.",
    )
    parser.add_argument(
        "--markets",
        default=csv_text(defaults.markets),
        help="Comma-separated The Odds API markets.",
    )
    parser.add_argument(
        "--odds-format",
        choices=("american", "decimal"),
        default="american",
        help="The Odds API odds format.",
    )
    parser.add_argument(
        "--public-picks-input",
        type=Path,
        nargs="+",
        help="Optional manual public pick CSV/JSON files to import before running.",
    )
    parser.add_argument(
        "--public-picks-format",
        choices=("csv", "json"),
        help="Format for --public-picks-input. Defaults to each file extension.",
    )
    parser.add_argument(
        "--public-picks-source",
        default="manual_import",
        help="Source label for imported public pick rows that omit source.",
    )
    parser.add_argument(
        "--no-aggregate-public-picks",
        action="store_true",
        help="Keep imported public pick source rows instead of writing consensus rows.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=defaults.simulation_count,
        help="Number of Monte Carlo simulations.",
    )
    parser.add_argument(
        "--entries",
        type=int,
        default=defaults.entry_count,
        help="Number of personal entries for simulations and portfolio allocation.",
    )
    parser.add_argument(
        "--aggression",
        choices=("conservative", "balanced", "aggressive"),
        default=defaults.aggression,
        help="Portfolio aggression mode.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=defaults.pool_size,
        help="Total pool entries used for leverage and simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--run-path-ev",
        action="store_true",
        help="Run the single-entry path EV optimizer as part of the live workflow.",
    )
    parser.add_argument(
        "--path-ev-simulations",
        type=int,
        default=None,
        help="Monte Carlo simulations for path EV. Defaults to --simulations.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=LiveWeekOptions.beam_width,
        help="Number of single-entry paths retained after each beam-search week.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=LiveWeekOptions.top_k,
        help="Weekly candidates expanded per path in the path EV optimizer.",
    )
    parser.add_argument(
        "--entry-fee",
        type=float,
        default=None,
        help="Optional entry cost for path EV multiple and edge reporting.",
    )
    parser.add_argument(
        "--prize-pool",
        type=float,
        default=None,
        help="Optional prize pool for path EV dollar reporting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validations and calculations without writing imported data or reports.",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> LiveWeekOptions:
    return LiveWeekOptions(
        season=args.season,
        week=args.week,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        simulations=args.simulations,
        entries=args.entries,
        aggression=args.aggression,
        pool_size=args.pool_size,
        seed=args.seed,
        refresh_odds=args.refresh_odds,
        sportsbooks=parse_csv_option(args.sportsbooks),
        odds_regions=parse_csv_option(args.odds_regions),
        markets=parse_csv_option(args.markets),
        odds_format=args.odds_format,
        public_pick_inputs=tuple(args.public_picks_input or ()),
        public_pick_format=args.public_picks_format,
        public_pick_source=args.public_picks_source,
        aggregate_public_picks=not args.no_aggregate_public_picks,
        run_path_ev=args.run_path_ev,
        path_ev_simulations=args.path_ev_simulations,
        beam_width=args.beam_width,
        top_k=args.top_k,
        entry_fee=args.entry_fee,
        prize_pool=args.prize_pool,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
