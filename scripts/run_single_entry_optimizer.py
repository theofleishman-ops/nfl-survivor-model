"""Run the single-entry path EV optimizer from local CSV inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import load_sample_data, load_season_data  # noqa: E402
from survivor.path_ev import PATH_EV_HORIZONS, optimize_single_entry_path  # noqa: E402
from survivor.path_ev_reports import write_single_entry_path_ev_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize one survivor entry for expected contest equity.",
    )
    parser.add_argument("--season", type=int, default=None, help="NFL season to load.")
    parser.add_argument("--week", type=int, required=True, help="Start week.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Season data directory or sample directory containing CSV inputs.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=5000,
        help="Monte Carlo simulations for true path EV.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=100,
        help="Number of paths retained after each beam-search week.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Weekly candidates expanded per path.",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Optional cap on final candidate paths evaluated.",
    )
    parser.add_argument(
        "--path-ev-horizon",
        choices=PATH_EV_HORIZONS,
        default="available",
        help="Evaluation horizon. Defaults to consecutive weeks with real odds available.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5000,
        help="Total contest entries, including this single entry.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument(
        "--entry-fee",
        type=float,
        default=None,
        help="Optional entry cost for EV multiple and edge reporting.",
    )
    parser.add_argument(
        "--prize-pool",
        type=float,
        default=None,
        help="Optional prize pool for dollar EV reporting.",
    )
    parser.add_argument(
        "--used-teams",
        default=None,
        help="Semicolon- or comma-separated teams already used by this entry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports",
        help="Directory for the Markdown report.",
    )
    args = parser.parse_args()

    data = _load_cli_data(args)
    used_teams = _parse_used_teams(args.used_teams) or _used_teams_from_entries(
        data.get("entries_df"),
        args.week,
    )

    result = optimize_single_entry_path(
        schedule_df=data["schedule_df"],
        odds_df=data["odds_df"],
        public_picks_df=data["public_picks_df"],
        start_week=args.week,
        used_teams=used_teams,
        pool_size=args.pool_size,
        max_paths=args.max_paths,
        simulations=args.simulations,
        random_seed=args.seed,
        top_k=args.top_k,
        beam_width=args.beam_width,
        entry_fee=args.entry_fee,
        prize_pool=args.prize_pool,
        path_ev_horizon=args.path_ev_horizon,
    )
    report_path = write_single_entry_path_ev_report(
        result,
        week=args.week,
        output_dir=args.output_dir,
    )

    current_pick = result.best_path[
        result.best_path["week"].astype(int) == int(args.week)
    ].iloc[0]
    print(f"Week {args.week} single-entry path EV optimizer")
    print(
        "Best current-week pick: "
        f"{current_pick['team']} over {current_pick['opponent']}"
    )
    print(f"Path EV: {_format_equity_pct(result.best_path_ev)}")
    print(f"Baseline fair value: {_format_money_or_not_provided(result.baseline_value)}")
    if result.ev_dollars is not None:
        print(f"EV dollars: ${result.ev_dollars:,.2f}")
    print(
        "EV multiple vs entry fee: "
        f"{_format_multiple_or_not_provided(result.ev_multiple_vs_entry_fee)}"
    )
    print(
        "EV multiple vs baseline fair value: "
        f"{_format_multiple_or_not_provided(result.ev_multiple_vs_baseline)}"
    )
    print(
        "Expected edge vs baseline: "
        f"{_format_edge_or_not_provided(result.expected_edge_vs_baseline)}"
    )
    print(f"Path survival probability: {_format_rate_pct(result.path_survival_probability)}")
    print(
        "Expected final survivors if alive: "
        f"{result.expected_survivors_if_alive:.2f}"
    )
    print("")
    print("Best path:")
    print(
        result.best_path[
            [
                "week",
                "team",
                "opponent",
                "win_probability",
                "projected_public_pick_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote markdown report: {report_path}")


def _load_cli_data(args: argparse.Namespace) -> dict[str, pd.DataFrame | None]:
    if args.season is None:
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


def _parse_used_teams(value: str | None) -> list[str]:
    if value is None or not value.strip() or value.strip().lower() in {"nan", "<na>"}:
        return []
    return [
        team.strip()
        for part in value.split(";")
        for team in part.split(",")
        if team.strip()
    ]


def _used_teams_from_entries(entries_df: pd.DataFrame | None, week: int) -> list[str]:
    if entries_df is None or entries_df.empty:
        return []

    entries = entries_df.copy()
    if "active" in entries.columns:
        active = entries[entries["active"].astype(bool)].copy()
        if not active.empty:
            entries = active

    first = entries.iloc[0]
    if "used_teams" in entries.columns:
        return _parse_used_teams(str(first.get("used_teams", "")))

    if {"entry_id", "week", "team_picked"}.issubset(entries.columns):
        entry_id = first["entry_id"]
        prior = entries[
            (entries["entry_id"] == entry_id)
            & (pd.to_numeric(entries["week"], errors="coerce") < int(week))
        ]
        return prior["team_picked"].dropna().astype(str).tolist()

    return []


def _format_equity_pct(value: float) -> str:
    numeric = float(value)
    if numeric == 0:
        return "0%"
    percentage = numeric * 100
    abs_percentage = abs(percentage)
    if abs_percentage >= 0.01:
        return f"{percentage:.4f}%"
    if abs_percentage >= 0.0001:
        return f"{percentage:.6f}%"
    return f"{percentage:.8f}%"


def _format_rate_pct(value: float) -> str:
    numeric = float(value)
    if numeric == 0:
        return "0%"
    percentage = numeric * 100
    abs_percentage = abs(percentage)
    if abs_percentage >= 1:
        return f"{percentage:.1f}%"
    if abs_percentage >= 0.01:
        return f"{percentage:.3f}%"
    return f"{percentage:.6f}%"


def _format_money_or_not_provided(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"${float(value):,.2f}"


def _format_multiple_or_not_provided(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"{float(value):.3f}x"


def _format_edge_or_not_provided(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"{float(value):+.1%}"


if __name__ == "__main__":
    main()
