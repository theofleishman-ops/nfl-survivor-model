"""Run local CSV-based weekly survivor rankings."""

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
from survivor.reports import write_weekly_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank survivor picks for a week.")
    parser.add_argument("--week", type=int, required=True, help="NFL week to rank.")
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
        "--pool-size",
        type=int,
        default=5000,
        help="Contest pool size used in leverage calculations.",
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
