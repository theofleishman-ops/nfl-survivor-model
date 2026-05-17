"""Markdown reports for single-entry path EV optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from survivor.config import OUTPUTS_DIR
from survivor.path_ev import SingleEntryPathOptimizationResult


def build_single_entry_path_ev_report(
    result: SingleEntryPathOptimizationResult,
    week: int,
) -> str:
    """Build a Markdown report for the single-entry path EV optimizer."""
    if result.best_path.empty:
        raise ValueError("Cannot build a path EV report without a best path.")

    current_pick = result.best_path[result.best_path["week"].astype(int) == int(week)]
    if current_pick.empty:
        current_pick = result.best_path.sort_values("week").head(1)
    pick = current_pick.iloc[0]

    alternatives = result.top_paths[result.top_paths["path_ev_rank"] > 1].head(10)
    lines = [
        f"# Week {week} Single-Entry Path EV",
        "",
        "## Current-Week Recommendation",
        "",
        (
            f"{pick['team']} over {pick['opponent']} with "
            f"{float(pick['win_probability']):.1%} win probability and "
            f"{float(pick['projected_public_pick_pct']):.1%} projected ownership."
        ),
        "",
        "## Path EV",
        "",
        f"- Entry cost / baseline value: {_format_money_or_blank(result.entry_cost)}",
        f"- Baseline contest equity: {result.baseline_contest_equity:.4%}",
        f"- Expected contest equity: {result.expected_contest_equity:.4%}",
        f"- EV dollars: {_format_money_or_blank(result.ev_dollars)}",
        f"- EV multiple: {result.ev_multiple:.3f}x",
        f"- Expected edge: {result.expected_edge:+.1%}",
        f"- Path survival probability: {result.path_survival_probability:.1%}",
        f"- Expected final field size: {result.expected_final_field_size:.2f}",
        f"- Expected survivors conditional on my survival: {result.expected_survivors_if_alive:.2f}",
        f"- Expected equity if alive: {result.expected_equity_if_alive:.4%}",
        "",
        "## Best Path",
        "",
        _markdown_table(
            result.best_path,
            [
                "week",
                "team",
                "opponent",
                "win_probability",
                "projected_public_pick_pct",
                "cumulative_survival_probability",
                "path_ev",
            ],
        ),
        "",
        "## Survival Probability By Week",
        "",
        _markdown_table(
            result.survival_probability_by_week,
            ["week", "path_survival_probability"],
        ),
        "",
        "## Expected Field Size By Week",
        "",
        _markdown_table(
            result.expected_field_size_by_week,
            [
                "week",
                "expected_public_entries",
                "path_survival_probability",
                "expected_total_entries",
            ],
        ),
        "",
        "## Top 10 Alternative Paths",
        "",
        _markdown_table(
            alternatives,
            [
                "path_ev_rank",
                "first_team",
                "path_ev",
                "ev_dollars",
                "ev_multiple",
                "path_survival_probability",
                "expected_survivors_if_alive",
                "expected_final_field_size",
                "path",
            ],
        ),
        "",
        "## Heuristic Ranking Comparison",
        "",
        _markdown_table(
            result.heuristic_comparison.head(10),
            [
                "heuristic_rank",
                "team",
                "opponent",
                "no_vig_win_probability",
                "public_pick_pct",
                "heuristic_final_score",
            "best_path_ev_for_current_pick",
            "best_path_ev_rank_for_current_pick",
            "selected_by_path_ev",
            ],
        ),
        "",
        "## Assumptions",
        "",
        *[f"- {assumption}" for assumption in result.assumptions],
        "",
    ]
    return "\n".join(lines)


def write_single_entry_path_ev_report(
    result: SingleEntryPathOptimizationResult,
    week: int,
    output_dir: str | Path = OUTPUTS_DIR / "reports",
) -> Path:
    """Write the path EV Markdown report and return its path."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"week_{week}_single_entry_path_ev.md"
    report_path.write_text(
        build_single_entry_path_ev_report(result, week),
        encoding="utf-8",
    )
    return report_path


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"

    table = df[[column for column in columns if column in df.columns]].copy()
    if table.empty:
        return "_No rows._"

    for column in table.columns:
        if column in {
            "win_probability",
            "projected_public_pick_pct",
            "cumulative_survival_probability",
            "path_ev",
            "path_survival_probability",
            "no_vig_win_probability",
            "public_pick_pct",
            "best_path_ev_for_current_pick",
        }:
            table[column] = table[column].map(_format_pct)
        elif column == "ev_dollars":
            table[column] = table[column].map(_format_money_or_blank)
        elif column == "ev_multiple":
            table[column] = table[column].map(_format_multiple)
        elif column not in {
            "week",
            "team",
            "opponent",
            "path",
            "first_team",
            "selected_by_path_ev",
        }:
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
    return f"{float(value):.3%}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.3f}"


def _format_money_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"${float(value):,.2f}"


def _format_multiple(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}x"
