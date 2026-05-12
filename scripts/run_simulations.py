"""Run Monte Carlo survivor simulations from local CSV inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import (  # noqa: E402
    load_entries_df,
    load_odds_df,
    load_public_picks_df,
    load_schedule_df,
)
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
        default=PROJECT_ROOT / "data" / "sample",
        help="Directory containing schedule, odds, public picks, and entries CSVs.",
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

    schedule_df = load_schedule_df(args.data_dir / "schedule_sample.csv")
    odds_df = load_odds_df(args.data_dir / "odds_sample.csv")
    public_picks_df = load_public_picks_df(args.data_dir / "public_picks_sample.csv")
    entries_df = load_entries_df(args.data_dir / "entries_sample.csv")

    result = simulate_many_seasons(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        entries_df=entries_df,
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


if __name__ == "__main__":
    main()
