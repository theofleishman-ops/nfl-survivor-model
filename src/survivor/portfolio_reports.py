"""Markdown reports for survivor portfolio allocations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from survivor.config import OUTPUTS_DIR


def build_portfolio_markdown_report(
    week: int,
    allocation_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    metrics: dict[str, Any],
    rankings_df: pd.DataFrame,
    aggression: str = "balanced",
) -> str:
    """Build a compact Markdown report for a weekly portfolio allocation."""
    lines = [
        f"# Week {week} Portfolio Report",
        "",
        "## Portfolio Metrics",
        "",
        _metrics_list(metrics),
        "",
        "## Allocation By Team",
        "",
        _markdown_table(
            exposure_df,
            [
                "team",
                "entries_allocated",
                "exposure_pct",
                "no_vig_win_probability",
                "public_pick_pct",
                "leverage_score",
                "scarcity_adjusted_future_cost",
                "portfolio_score",
            ],
        ),
        "",
        "## Entry Allocations",
        "",
        _markdown_table(
            allocation_df,
            [
                "entry_id",
                "team",
                "no_vig_win_probability",
                "public_pick_pct",
                "leverage_score",
                "scarcity_adjusted_future_cost",
                "allocation_reason",
            ],
        ),
        "",
        "## Concentration",
        "",
        (
            "The concentration index is Herfindahl-Hirschman Index: the sum of "
            "squared team exposure shares. A one-team portfolio is 1.000; lower "
            "values indicate more diversification across independent NFL games."
        ),
        (
            f"This allocation uses {int(metrics['number_of_teams_used'])} teams "
            f"with a max exposure of {metrics['max_team_exposure']:.1%} and an "
            f"HHI of {metrics['concentration_hhi']:.3f}."
        ),
        "",
        "## Top Excluded Teams",
        "",
        _markdown_table(
            _excluded_teams(rankings_df, exposure_df, aggression=aggression),
            [
                "team",
                "no_vig_win_probability",
                "public_pick_pct",
                "leverage_score",
                "scarcity_adjusted_future_cost",
                "final_score",
                "reason",
            ],
        ),
        "",
        "## Correlation Risk Notes",
        "",
        (
            "The independent at-least-one-survives estimate treats every entry "
            "as a separate trial, so duplicate teams make that number too "
            "optimistic."
        ),
        (
            "The correlated estimate groups entries by team and treats all "
            "entries on the same team as one shared outcome. That is the better "
            "weekly risk lens for same-team exposure."
        ),
        (
            "- Independent estimate: "
            f"{metrics['prob_at_least_one_survives_independent']:.1%}"
        ),
        (
            "- Correlated estimate: "
            f"{metrics['prob_at_least_one_survives_correlated']:.1%}"
        ),
        "",
    ]
    return "\n".join(lines)


def write_portfolio_report(
    week: int,
    allocation_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    metrics: dict[str, Any],
    rankings_df: pd.DataFrame,
    aggression: str = "balanced",
    output_dir: str | Path = OUTPUTS_DIR / "reports",
) -> Path:
    """Write the weekly portfolio markdown report and return its path."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"week_{week}_portfolio_report.md"
    report_path.write_text(
        build_portfolio_markdown_report(
            week=week,
            allocation_df=allocation_df,
            exposure_df=exposure_df,
            metrics=metrics,
            rankings_df=rankings_df,
            aggression=aggression,
        ),
        encoding="utf-8",
    )
    return report_path


def _metrics_list(metrics: dict[str, Any]) -> str:
    rows = [
        ("Active entries", f"{int(metrics['active_entries']):,}"),
        ("Allocated entries", f"{int(metrics['allocated_entries']):,}"),
        ("Unallocated entries", f"{int(metrics['unallocated_entries']):,}"),
        ("Teams used", f"{int(metrics['number_of_teams_used']):,}"),
        ("Max team exposure", f"{metrics['max_team_exposure']:.1%}"),
        ("Average win probability", f"{metrics['average_win_probability']:.1%}"),
        ("Weighted public ownership", f"{metrics['weighted_public_ownership']:.1%}"),
        ("Weighted leverage score", f"{metrics['weighted_leverage_score']:.3f}"),
        ("Weighted future cost", f"{metrics['weighted_future_cost']:.3f}"),
        ("Concentration HHI", f"{metrics['concentration_hhi']:.3f}"),
        (
            "At least one survives, independent",
            f"{metrics['prob_at_least_one_survives_independent']:.1%}",
        ),
        (
            "At least one survives, correlated",
            f"{metrics['prob_at_least_one_survives_correlated']:.1%}",
        ),
    ]
    return "\n".join(f"- {label}: {value}" for label, value in rows)


def _excluded_teams(
    rankings_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    aggression: str,
) -> pd.DataFrame:
    if rankings_df.empty:
        return pd.DataFrame(columns=["team", "reason"])

    rankings = rankings_df.copy()
    rankings["team"] = rankings["team"].astype(str)
    used_teams = set(exposure_df["team"].astype(str)) if not exposure_df.empty else set()
    excluded = rankings[~rankings["team"].isin(used_teams)].copy()
    if excluded.empty:
        return pd.DataFrame(columns=["team", "reason"])

    sort_columns = [
        column
        for column in ["final_score", "no_vig_win_probability"]
        if column in excluded.columns
    ]
    excluded = excluded.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    selected_public = (
        float(exposure_df["public_pick_pct"].mean()) if not exposure_df.empty else 0.0
    )
    median_future_cost = (
        float(rankings["scarcity_adjusted_future_cost"].median())
        if "scarcity_adjusted_future_cost" in rankings.columns
        else 0.0
    )

    rows = []
    for row in excluded.head(8).to_dict("records"):
        rows.append(row | {"reason": _exclusion_reason(row, aggression, selected_public, median_future_cost)})
    return pd.DataFrame(rows)


def _exclusion_reason(
    row: dict[str, Any],
    aggression: str,
    selected_public: float,
    median_future_cost: float,
) -> str:
    win_probability = float(row.get("no_vig_win_probability", 0.0))
    public_pick_pct = float(row.get("public_pick_pct", 0.0))
    future_cost = float(row.get("scarcity_adjusted_future_cost", 0.0))

    if aggression != "aggressive" and win_probability < 0.50:
        return "below the 50% win-probability floor for this mode"
    if future_cost > median_future_cost:
        return "higher future cost than the median remaining option"
    if public_pick_pct > selected_public:
        return "more public ownership than the allocated mix"
    return "lower portfolio score after exposure and eligibility constraints"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[[column for column in columns if column in df.columns]].copy()
    if table.empty:
        return "_No rows._"

    for column in table.columns:
        if column in {
            "exposure_pct",
            "no_vig_win_probability",
            "public_pick_pct",
        }:
            table[column] = table[column].map(_format_pct)
        elif column not in {"entry_id", "team", "allocation_reason", "reason"}:
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
    return f"{float(value):.1%}"


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"
