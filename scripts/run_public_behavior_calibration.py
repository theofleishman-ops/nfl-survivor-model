"""Run full-season path EV across public behavior calibration presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survivor.loaders import load_sample_data, load_season_data, load_team_strength_df  # noqa: E402
from survivor.path_ev import (  # noqa: E402
    DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
    DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE,
    DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    optimize_single_entry_path,
)
from survivor.public_behavior import (  # noqa: E402
    PUBLIC_BEHAVIOR_PRESET_ORDER,
    get_public_behavior_config,
)
from survivor.public_behavior_reports import (  # noqa: E402
    write_public_behavior_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate full-season path EV across public behavior presets.",
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
        "--team-strength",
        type=Path,
        default=None,
        help="Optional team_strength.csv with season, team, rating, source, notes.",
    )
    parser.add_argument(
        "--team-strength-home-field",
        type=float,
        default=DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
        help="Home-field adjustment added to the team-strength rating differential.",
    )
    parser.add_argument(
        "--team-strength-scale",
        type=float,
        default=DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
        help="Positive logistic scale for team-strength rating differentials.",
    )
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
        "--pool-size",
        type=int,
        default=5000,
        help="Total contest entries, including this single entry.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Monte Carlo simulations per behavior preset.",
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
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument(
        "--public-field-sample-size",
        type=int,
        default=DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE,
        help="Weighted public entries sampled per path-EV simulation.",
    )
    parser.add_argument(
        "--presets",
        default=",".join(PUBLIC_BEHAVIOR_PRESET_ORDER),
        help="Comma-separated public behavior presets to run.",
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
    presets = _parse_presets(args.presets)

    calibration = run_public_behavior_calibration(
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
        team_strength_df=data.get("team_strength_df"),
        team_strength_home_field_adjustment=args.team_strength_home_field,
        team_strength_scale=args.team_strength_scale,
        public_field_sample_size=args.public_field_sample_size,
        presets=presets,
    )
    report_path = write_public_behavior_calibration_report(
        calibration,
        week=args.week,
        output_dir=args.output_dir,
    )

    print(f"Week {args.week} public behavior calibration")
    print(
        calibration[
            [
                "preset",
                "ev_dollars",
                "ev_multiple",
                "path_survival_probability",
                "expected_survivors_if_alive",
                "expected_final_field_size",
                "average_max_ownership_by_week",
                "average_chalk_concentration",
                "expected_team_exhaustion",
                "best_path",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote markdown report: {report_path}")


def run_public_behavior_calibration(
    *,
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    start_week: int,
    used_teams: list[str] | set[str] | tuple[str, ...] | None,
    pool_size: int,
    max_paths: int | None,
    simulations: int,
    random_seed: int | None,
    top_k: int,
    beam_width: int,
    entry_fee: float | None,
    prize_pool: float | None,
    team_strength_df: pd.DataFrame | None,
    team_strength_home_field_adjustment: float,
    team_strength_scale: float,
    public_field_sample_size: int | None,
    presets: list[str] | tuple[str, ...] = PUBLIC_BEHAVIOR_PRESET_ORDER,
) -> pd.DataFrame:
    """Run the optimizer once per public behavior preset."""

    rows: list[dict[str, object]] = []
    for preset_name in presets:
        config = get_public_behavior_config(preset_name)
        result = optimize_single_entry_path(
            schedule_df=schedule_df,
            odds_df=odds_df,
            public_picks_df=public_picks_df,
            start_week=start_week,
            used_teams=used_teams,
            pool_size=pool_size,
            max_paths=max_paths,
            simulations=simulations,
            random_seed=random_seed,
            top_k=top_k,
            beam_width=beam_width,
            entry_fee=entry_fee,
            prize_pool=prize_pool,
            path_ev_horizon="full-season",
            team_strength_df=team_strength_df,
            team_strength_home_field_adjustment=team_strength_home_field_adjustment,
            team_strength_scale=team_strength_scale,
            public_field_sample_size=public_field_sample_size,
            public_behavior_config=config,
        )
        rows.append(_calibration_row(config.name, config.description, result))
    return pd.DataFrame(rows)


def _calibration_row(
    preset: str,
    description: str,
    result: object,
) -> dict[str, object]:
    diagnostics = getattr(result, "diagnostics", {}) or {}
    scarcity = pd.DataFrame(diagnostics.get("public_field_scarcity_by_week", []))
    exhaustion = pd.DataFrame(diagnostics.get("expected_team_exhaustion", []))
    average_max_ownership = _mean_or_none(scarcity, "max_projected_ownership_pct")
    average_chalk = _mean_or_none(scarcity, "expected_chalk_concentration")
    expected_exhaustion = _final_week_mean_or_none(exhaustion, "burned_pct")
    config_values = diagnostics.get("public_behavior_config", {}) or {}
    return {
        "preset": preset,
        "description": description,
        "ev_dollars": getattr(result, "ev_dollars", None),
        "ev_multiple": getattr(result, "ev_multiple_vs_entry_fee", None),
        "path_survival_probability": getattr(result, "path_survival_probability", None),
        "expected_survivors_if_alive": getattr(result, "expected_survivors_if_alive", None),
        "expected_final_field_size": getattr(result, "expected_final_field_size", None),
        "average_max_ownership_by_week": average_max_ownership,
        "average_chalk_concentration": average_chalk,
        "expected_team_exhaustion": expected_exhaustion,
        "best_path": _format_best_path(getattr(result, "best_path", pd.DataFrame())),
        **config_values,
    }


def _load_cli_data(args: argparse.Namespace) -> dict[str, pd.DataFrame | None]:
    if args.season is None:
        sample_dir = args.data_dir or PROJECT_ROOT / "data" / "sample"
        sample_data = load_sample_data(sample_dir)
        return {
            "schedule_df": sample_data["schedule"],
            "odds_df": sample_data["odds"],
            "public_picks_df": sample_data["public_picks"],
            "entries_df": sample_data["entries"],
            "team_strength_df": _load_cli_team_strength(args),
        }

    data_dir = args.data_dir or PROJECT_ROOT / "data" / "raw"
    data = load_season_data(season=args.season, data_dir=data_dir)
    explicit_strength = _load_cli_team_strength(args)
    if explicit_strength is not None:
        data["team_strength_df"] = explicit_strength
    return data


def _load_cli_team_strength(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.team_strength is None:
        return None
    strength = load_team_strength_df(args.team_strength)
    if args.season is not None and "season" in strength.columns:
        strength = strength[strength["season"].astype(int) == int(args.season)].copy()
    return strength


def _parse_presets(value: str) -> list[str]:
    presets = [part.strip() for part in str(value).split(",") if part.strip()]
    if not presets:
        raise ValueError("--presets must include at least one preset.")
    for preset in presets:
        get_public_behavior_config(preset)
    return presets


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


def _mean_or_none(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _final_week_mean_or_none(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns or "week" not in df.columns:
        return None
    final_week = int(pd.to_numeric(df["week"], errors="coerce").max())
    final = df[pd.to_numeric(df["week"], errors="coerce").astype("Int64") == final_week]
    return _mean_or_none(final, column)


def _format_best_path(path: pd.DataFrame) -> str:
    if path.empty:
        return ""
    ordered = path.sort_values("week")
    return " > ".join(
        f"W{int(row['week'])}:{row['team']}" for row in ordered.to_dict("records")
    )


if __name__ == "__main__":
    main()
