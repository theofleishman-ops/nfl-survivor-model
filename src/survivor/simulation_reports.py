"""Reporting helpers for Monte Carlo simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from survivor.config import OUTPUTS_DIR
from survivor.simulator import SimulationResult


def build_simulation_markdown_report(result: SimulationResult) -> str:
    """Build a compact Markdown summary from a Monte Carlo result."""
    summary = result.summary
    lines = [
        f"# Survivor Simulation Report: Week {summary['start_week']} Start",
        "",
        "## Summary",
        "",
        f"- Simulations: {int(summary['simulations']):,}",
        f"- Pool size: {int(summary['pool_size']):,}",
        f"- Personal entries: {int(summary['personal_entry_count']):,}",
        (
            "- Probability at least one personal entry survives: "
            f"{summary['probability_at_least_one_personal_survives']:.1%}"
        ),
        (
            "- Expected final public entries: "
            f"{summary['expected_final_public_entries']:.2f}"
        ),
        (
            "- Expected final personal entries: "
            f"{summary['expected_final_personal_entries']:.2f}"
        ),
        f"- Expected contest equity: {summary['expected_contest_equity']:.3%}",
        "",
        "## Expected Remaining Entries By Week",
        "",
        _markdown_table(
            result.week_summary,
            [
                "week",
                "expected_public_entries",
                "expected_personal_entries",
                "expected_total_entries",
                "probability_at_least_one_personal_survives",
                "expected_contest_equity",
            ],
        ),
        "",
        "## Upset Leverage Observations",
        "",
        _leverage_observations(result.leverage_summary),
        "",
        _markdown_table(
            result.leverage_summary.head(10),
            [
                "week",
                "team",
                "public_pick_pct",
                "simulated_loss_rate",
                "avg_field_eliminated_if_team_loses",
                "avg_field_shrink_pct_if_team_loses",
                "contest_equity_lift_if_team_loses",
                "uniqueness_value",
                "expected_upset_equity_gain",
            ],
        ),
        "",
        "## Top Survivor Paths",
        "",
        _markdown_table(
            result.path_summary.head(10),
            [
                "path",
                "entries",
                "survival_rate",
                "avg_weeks_survived",
                "avg_final_contest_equity",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def write_simulation_report(
    result: SimulationResult,
    output_dir: str | Path = OUTPUTS_DIR / "simulations",
) -> Path:
    """Write the Markdown simulation report and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start_week = int(result.summary["start_week"])
    report_path = output_path / f"week_{start_week}_simulation_report.md"
    report_path.write_text(build_simulation_markdown_report(result), encoding="utf-8")
    return report_path


def write_simulation_csvs(
    result: SimulationResult,
    output_dir: str | Path = OUTPUTS_DIR / "simulations",
) -> dict[str, Path]:
    """Write compact CSV summaries for simulation review."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start_week = int(result.summary["start_week"])

    paths = {
        "season_summary": output_path / f"week_{start_week}_season_summary.csv",
        "week_summary": output_path / f"week_{start_week}_week_summary.csv",
        "leverage_summary": output_path / f"week_{start_week}_leverage_summary.csv",
        "path_summary": output_path / f"week_{start_week}_path_summary.csv",
    }
    pd.DataFrame([result.summary]).to_csv(paths["season_summary"], index=False)
    result.week_summary.to_csv(paths["week_summary"], index=False)
    result.leverage_summary.to_csv(paths["leverage_summary"], index=False)
    result.path_summary.to_csv(paths["path_summary"], index=False)
    return paths


def write_simulation_outputs(
    result: SimulationResult,
    output_dir: str | Path = OUTPUTS_DIR / "simulations",
) -> dict[str, Path]:
    """Write Markdown and CSV outputs for a simulation run."""
    paths = write_simulation_csvs(result, output_dir=output_dir)
    paths["markdown_report"] = write_simulation_report(result, output_dir=output_dir)
    return paths


def _leverage_observations(leverage_summary: pd.DataFrame) -> str:
    if leverage_summary.empty:
        return "No leverage observations were generated."

    top_field_shrink = leverage_summary.sort_values(
        "avg_field_eliminated_if_team_loses",
        ascending=False,
    ).iloc[0]
    top_equity_lift = leverage_summary.sort_values(
        "contest_equity_lift_if_team_loses",
        ascending=False,
    ).iloc[0]
    most_unique = leverage_summary.sort_values(
        "uniqueness_value",
        ascending=False,
    ).iloc[0]

    return "\n".join(
        [
            (
                f"- Largest conditional field shrink: Week {int(top_field_shrink['week'])} "
                f"{top_field_shrink['team']} losing removes about "
                f"{top_field_shrink['avg_field_eliminated_if_team_loses']:.1f} "
                "public entries."
            ),
            (
                f"- Best conditional equity lift: Week {int(top_equity_lift['week'])} "
                f"{top_equity_lift['team']} losing changes contest equity by "
                f"{top_equity_lift['contest_equity_lift_if_team_loses']:.3%}."
            ),
            (
                f"- Highest uniqueness value: Week {int(most_unique['week'])} "
                f"{most_unique['team']} at {most_unique['uniqueness_value']:.3f}."
            ),
        ]
    )


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[[column for column in columns if column in df.columns]].copy()
    if table.empty:
        return "_No rows._"

    for column in table.columns:
        if column in {
            "probability_at_least_one_personal_survives",
            "expected_contest_equity",
            "public_pick_pct",
            "simulated_loss_rate",
            "avg_field_shrink_pct_if_team_loses",
            "contest_equity_lift_if_team_loses",
            "expected_upset_equity_gain",
            "survival_rate",
            "avg_final_contest_equity",
        }:
            table[column] = table[column].map(_format_pct)
        elif column not in {"week", "team", "path", "entries"}:
            table[column] = table[column].map(_format_number)

    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.to_dict("records"):
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _format_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"
