"""Run local CSV-based weekly survivor rankings."""

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
from survivor.reports import write_weekly_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank survivor picks for a week.")
    parser.add_argument("--week", type=int, required=True, help="NFL week to rank.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample",
        help="Directory containing schedule, odds, public picks, and entries CSVs.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5000,
        help="Contest pool size used in leverage calculations.",
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
    report_path = write_weekly_report(rankings, args.week)

    print(f"Week {args.week} top survivor recommendations")
    print(
        rankings[
            [
                "rank",
                "team",
                "opponent",
                "no_vig_win_probability",
                "public_pick_pct",
                "final_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nWrote markdown report: {report_path}")


if __name__ == "__main__":
    main()
