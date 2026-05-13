"""Run the weekly survivor portfolio optimizer from local CSV inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import load_sample_data, load_season_data  # noqa: E402
from survivor.optimizer import rank_weekly_picks  # noqa: E402
from survivor.portfolio import optimize_portfolio_for_week  # noqa: E402
from survivor.portfolio_reports import write_portfolio_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize a multi-entry survivor portfolio for one week.",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week to optimize.",
    )
    parser.add_argument(
        "--entries",
        type=int,
        default=40,
        help="Number of personal entries to allocate.",
    )
    parser.add_argument(
        "--aggression",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Portfolio risk posture.",
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
        default=PROJECT_ROOT / "outputs" / "reports",
        help="Directory for the portfolio markdown report.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5000,
        help="Contest pool size used in leverage calculations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic random seed for tie-breaking entry order.",
    )
    args = parser.parse_args()

    data = _load_cli_data(args)

    rankings = rank_weekly_picks(
        week=args.week,
        schedule_df=data["schedule_df"],
        odds_df=data["odds_df"],
        public_picks_df=data["public_picks_df"],
        entries_df=data["entries_df"],
        pool_size=args.pool_size,
    )
    allocation, exposure, metrics = optimize_portfolio_for_week(
        week=args.week,
        rankings_df=rankings,
        entries_df=data["entries_df"],
        personal_entry_count=args.entries,
        aggression=args.aggression,
        random_seed=args.seed,
    )
    report_path = write_portfolio_report(
        week=args.week,
        allocation_df=allocation,
        exposure_df=exposure,
        metrics=metrics,
        rankings_df=rankings,
        aggression=args.aggression,
        output_dir=args.output_dir,
    )

    print(
        f"Week {args.week} portfolio optimization "
        f"({args.aggression}, {metrics['active_entries']} active entries)"
    )
    print(
        exposure[
            [
                "team",
                "entries_allocated",
                "exposure_pct",
                "no_vig_win_probability",
                "public_pick_pct",
                "portfolio_score",
            ]
        ].to_string(index=False)
    )
    print(
        "\nCorrelated probability at least one entry survives: "
        f"{metrics['prob_at_least_one_survives_correlated']:.1%}"
    )
    print(f"Wrote markdown report: {report_path}")


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
