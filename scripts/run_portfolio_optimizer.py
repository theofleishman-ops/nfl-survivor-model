"""Run the weekly survivor portfolio optimizer from local CSV inputs."""

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
        default=PROJECT_ROOT / "data" / "sample",
        help="Directory containing schedule, odds, public picks, and entries CSVs.",
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

    schedule_df = load_schedule_df(args.data_dir / "schedule_sample.csv")
    odds_df = load_odds_df(args.data_dir / "odds_sample.csv")
    public_picks_df = load_public_picks_df(args.data_dir / "public_picks_sample.csv")
    entries_df = load_entries_df(args.data_dir / "entries_sample.csv")

    rankings = rank_weekly_picks(
        week=args.week,
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        entries_df=entries_df,
        pool_size=args.pool_size,
    )
    allocation, exposure, metrics = optimize_portfolio_for_week(
        week=args.week,
        rankings_df=rankings,
        entries_df=entries_df,
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


if __name__ == "__main__":
    main()
