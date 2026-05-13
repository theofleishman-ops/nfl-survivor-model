"""Run Monte Carlo survivor simulations from local CSV inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import load_sample_data, load_season_data  # noqa: E402
from survivor.simulation_reports import write_simulation_outputs  # noqa: E402
from survivor.simulator import simulate_many_seasons  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for a survivor pool.",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=1,
        help="Starting NFL week for the simulation.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo seasons to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Sample directory when using sample data; raw data root when --season "
            "is set."
        ),
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="NFL season to load from data/raw/{season}. Omit for sample data.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use checked-in sample CSVs. This is also the default without --season.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "simulations",
        help="Directory for simulation reports and CSV summaries.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5000,
        help="Total contest entries, including personal entries.",
    )
    parser.add_argument(
        "--personal-entries",
        type=int,
        default=1,
        help="Number of controlled personal entries to track.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic random seed.",
    )
    args = parser.parse_args()

    data = _load_cli_data(args)

    result = simulate_many_seasons(
        schedule_df=data["schedule_df"],
        odds_df=data["odds_df"],
        public_picks_df=data["public_picks_df"],
        entries_df=data["entries_df"],
        start_week=args.week,
        simulations=args.simulations,
        pool_size=args.pool_size,
        personal_entry_count=args.personal_entries,
        seed=args.seed,
    )
    output_paths = write_simulation_outputs(result, output_dir=args.output_dir)
    summary = result.summary

    print(f"Monte Carlo survivor simulation from week {args.week}")
    print(f"Simulations: {summary['simulations']:,}")
    print(
        "Probability at least one personal entry survives: "
        f"{summary['probability_at_least_one_personal_survives']:.1%}"
    )
    print(
        "Expected final entries: "
        f"{summary['expected_final_total_entries']:.2f} "
        f"({summary['expected_final_public_entries']:.2f} public, "
        f"{summary['expected_final_personal_entries']:.2f} personal)"
    )
    print(f"Expected contest equity: {summary['expected_contest_equity']:.3%}")
    print(f"\nWrote markdown report: {output_paths['markdown_report']}")
    print(f"Wrote week CSV: {output_paths['week_summary']}")
    print(f"Wrote leverage CSV: {output_paths['leverage_summary']}")
    print(f"Wrote path CSV: {output_paths['path_summary']}")


def _load_cli_data(args: argparse.Namespace) -> dict[str, object]:
    if args.use_sample or args.season is None:
        sample_dir = args.data_dir or PROJECT_ROOT / "data" / "sample"
        sample_data = load_sample_data(sample_dir)
        return {
            "schedule_df": sample_data["schedule"],
            "odds_df": sample_data["odds"],
            "public_picks_df": sample_data["public_picks"],
            "entries_df": sample_data["entries"],
        }

    data_dir = args.data_dir or PROJECT_ROOT / "data" / "raw"
    return load_season_data(season=args.season, data_dir=data_dir)


if __name__ == "__main__":
    main()
