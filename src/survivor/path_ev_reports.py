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
    diagnostics = result.diagnostics or {}
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
        f"- Full-season EV reliability: {diagnostics.get('full_season_ev_reliability', 'not_applicable')}",
        f"- Full-season EV actionability: {diagnostics.get('full_season_ev_actionability', 'not_applicable')}",
        f"- Fallback coverage: {_format_optional_pct(diagnostics.get('fallback_coverage_pct'))}",
        f"- Evaluation horizon: {diagnostics.get('horizon', 'unknown')}",
        f"- Number of weeks evaluated: {diagnostics.get('number_of_weeks_evaluated', len(result.best_path))}",
        f"- Entry fee: {_format_money_or_blank(result.entry_fee)}",
        f"- Prize pool: {_format_money_or_blank(result.prize_pool)}",
        f"- Baseline contest equity: {_format_equity_pct(result.baseline_contest_equity)}",
        f"- Baseline fair value: {_format_money_or_blank(result.baseline_value)}",
        f"- Expected contest equity: {_format_equity_pct(result.expected_contest_equity)}",
        f"- EV dollars: {_format_money_or_blank(result.ev_dollars)}",
        f"- EV multiple vs entry fee: {_format_multiple_or_blank(result.ev_multiple_vs_entry_fee)}",
        f"- EV multiple vs baseline fair value: {_format_multiple_or_blank(result.ev_multiple_vs_baseline)}",
        f"- Expected edge vs baseline: {_format_edge_or_blank(result.expected_edge_vs_baseline)}",
        f"- Path survival probability: {_format_rate_pct(result.path_survival_probability)}",
        f"- Expected final field size: {result.expected_final_field_size:.2f}",
        f"- Expected survivors conditional on my survival: {result.expected_survivors_if_alive:.2f}",
        f"- Expected equity if alive: {_format_equity_pct(result.expected_equity_if_alive)}",
        f"- Cumulative path EV: {_format_equity_pct(result.best_path_ev)}",
        f"- Public field model: {diagnostics.get('public_field_model', 'unknown')}",
        (
            f"- Public field sample size: "
            f"{_format_optional_count(diagnostics.get('public_field_sample_size'))}"
        ),
        (
            f"- Path uniqueness score: "
            f"{_format_optional_pct(diagnostics.get('path_uniqueness_score'))}"
        ),
        (
            f"- Uniqueness after clustering adjustment: "
            f"{_format_optional_pct(diagnostics.get('cluster_adjusted_uniqueness_score'))}"
        ),
        (
            f"- Expected overlap with public field: "
            f"{_format_optional_count(diagnostics.get('expected_public_overlap_entries'))} "
            f"({_format_optional_pct(diagnostics.get('expected_public_overlap_pct'))})"
        ),
        (
            f"- Expected duplicate-path count: "
            f"{_format_optional_count(diagnostics.get('expected_duplicate_path_count'))}"
        ),
        (
            f"- Expected identical-path survivors: "
            f"{_format_optional_count(diagnostics.get('expected_identical_path_survivors'))}"
        ),
        (
            f"- Path clustering score: "
            f"{_format_optional_pct(diagnostics.get('path_clustering_score'))}"
        ),
        (
            f"- Late-season congestion score: "
            f"{_format_optional_pct(diagnostics.get('late_season_congestion_score'))}"
        ),
        (
            f"- Cluster penalty entries: "
            f"{_format_optional_count(diagnostics.get('cluster_penalty_entries'))}"
        ),
        (
            f"- Path clustering warnings: "
            f"{_format_warnings(diagnostics.get('path_clustering_warnings'))}"
        ),
        f"- Scarcity weeks: {_format_week_list(diagnostics.get('scarcity_weeks'))}",
        f"- Weeks with real odds: {_format_week_list(diagnostics.get('weeks_with_real_odds'))}",
        f"- Weeks with projected odds: {_format_week_list(diagnostics.get('weeks_with_projected_odds'))}",
        (
            f"- Weeks with projected team-strength probabilities: "
            f"{_format_week_list(diagnostics.get('weeks_with_projected_team_strength_probabilities'))}"
        ),
        (
            f"- Weeks using fallback probabilities: "
            f"{_format_week_list(diagnostics.get('weeks_using_fallback_probabilities'))}"
        ),
        (
            f"- Average projected win probability: "
            f"{_format_optional_rate(diagnostics.get('average_projected_win_probability'))}"
        ),
        (
            f"- Average fallback win probability: "
            f"{_format_optional_rate(diagnostics.get('average_fallback_win_probability'))}"
        ),
        (
            f"- Probability sources: "
            f"{_format_methods(diagnostics.get('probability_sources'))}"
        ),
        (
            f"- Ownership projection method: "
            f"{_format_methods(diagnostics.get('ownership_projection_methods'))}"
        ),
        "",
        "## Horizon Comparison",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("horizon_comparison", [])),
            [
                "label",
                "status",
                "weeks_evaluated",
                "fallback_coverage_pct",
                "full_season_ev_reliability",
                "path_ev",
                "ev_dollars",
                "ev_multiple_vs_entry_fee",
                "ev_multiple_vs_baseline",
                "expected_edge_vs_baseline",
                "path_survival_probability",
                "expected_survivors_if_alive",
                "note",
            ],
        ),
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
                "expected_public_entries_before_week",
                "raw_expected_public_entries",
                "expected_public_entries",
                "path_survival_probability",
                "expected_total_entries",
                "expected_public_overlap_entries",
                "expected_duplicate_path_count",
                "expected_identical_path_survivors",
                "cluster_penalty_entries",
                "avg_path_overlap_pct",
                "raw_path_uniqueness_score",
                "path_uniqueness_score",
                "cluster_adjusted_uniqueness_score",
            ],
        ),
        "",
        "## Public Field Evolution",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("expected_remaining_field_by_week", [])),
            [
                "week",
                "expected_remaining_entries",
                "average_available_teams",
                "max_projected_ownership_pct",
                "expected_chalk_concentration",
                "scarcity_index",
                "scarcity_spike",
            ],
        ),
        "",
        "## Expected Team Exhaustion",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("expected_team_exhaustion", [])),
            [
                "week",
                "team",
                "expected_entries_burned_team",
                "burned_pct",
                "expected_remaining_entries_with_team_available",
                "remaining_field_available_pct",
            ],
        ),
        "",
        "## Projected Future Ownership By Week",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("projected_public_ownership_by_week", [])),
            [
                "week",
                "team",
                "expected_public_picks",
                "projected_ownership_pct",
            ],
        ),
        "",
        "## Top Crowded Future Path Archetypes",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("top_crowded_future_path_archetypes", [])),
            [
                "cluster_id",
                "late_path_key",
                "expected_entries",
                "expected_surviving_entries",
                "member_path_count",
                "cluster_share",
                "late_season_congestion_score",
                "path_key",
            ],
        ),
        "",
        "## Path Convergence By Week",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("path_convergence_by_week", [])),
            [
                "week",
                "top_prefix",
                "expected_entries_on_top_prefix",
                "top_prefix_share",
                "distinct_prefix_count",
            ],
        ),
        "",
        "## Scarcity Weeks",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("public_field_scarcity_by_week", [])),
            [
                "week",
                "expected_remaining_entries",
                "average_available_teams",
                "max_projected_ownership_pct",
                "expected_chalk_concentration",
                "scarcity_index",
                "scarcity_spike",
            ],
        ),
        "",
        "## Path Uniqueness Diagnostics",
        "",
        _markdown_table(
            result.expected_field_size_by_week,
            [
                "week",
                "expected_public_overlap_entries",
                "avg_path_overlap_pct",
                "expected_duplicate_path_count",
                "expected_identical_path_survivors",
                "expected_near_identical_path_survivors",
                "cluster_penalty_entries",
                "path_clustering_score",
                "late_season_congestion_score",
                "raw_path_uniqueness_score",
                "path_uniqueness_score",
                "cluster_adjusted_uniqueness_score",
                "public_field_model",
            ],
        ),
        "",
        "## Path EV Creation By Week",
        "",
        _markdown_table(
            _path_ev_creation_table(result),
            [
                "week",
                "team",
                "opponent",
                "probability_source",
                "win_probability",
                "projected_public_pick_pct",
                "path_survival_probability",
                "expected_equity_if_alive",
                "cumulative_path_ev",
                "weekly_ev_delta",
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
                "ev_multiple_vs_entry_fee",
                "ev_multiple_vs_baseline",
                "path_survival_probability",
                "expected_survivors_if_alive",
                "expected_final_field_size",
                "expected_duplicate_path_count",
                "path_clustering_score",
                "late_season_congestion_score",
                "cluster_adjusted_uniqueness_score",
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


def _path_ev_creation_table(result: SingleEntryPathOptimizationResult) -> pd.DataFrame:
    if result.best_path.empty or result.expected_field_size_by_week.empty:
        return pd.DataFrame()
    return result.best_path.merge(
        result.expected_field_size_by_week[
            [
                column
                for column in [
                    "week",
                    "expected_equity_if_alive",
                    "cumulative_path_ev",
                    "weekly_ev_delta",
                ]
                if column in result.expected_field_size_by_week.columns
            ]
        ],
        on="week",
        how="left",
    )


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
            "expected_equity_if_alive",
            "cumulative_path_ev",
            "weekly_ev_delta",
            "fallback_coverage_pct",
            "avg_path_overlap_pct",
            "path_uniqueness_score",
            "raw_path_uniqueness_score",
            "cluster_adjusted_uniqueness_score",
            "path_clustering_score",
            "late_season_congestion_score",
            "cluster_share",
            "surviving_cluster_share",
            "top_prefix_share",
            "burned_pct",
            "remaining_field_available_pct",
            "remaining_field_used_pct",
            "projected_ownership_pct",
            "max_projected_ownership_pct",
            "expected_chalk_concentration",
        }:
            if column in {
                "path_ev",
                "best_path_ev_for_current_pick",
                "expected_equity_if_alive",
                "cumulative_path_ev",
                "weekly_ev_delta",
            }:
                table[column] = table[column].map(_format_equity_pct)
            else:
                table[column] = table[column].map(_format_pct)
        elif column == "ev_dollars":
            table[column] = table[column].map(_format_money_or_blank)
        elif column in {
            "ev_multiple",
            "ev_multiple_vs_entry_fee",
            "ev_multiple_vs_baseline",
        }:
            table[column] = table[column].map(_format_multiple_or_blank)
        elif column == "expected_edge_vs_baseline":
            table[column] = table[column].map(_format_edge_or_blank)
        elif column not in {
            "week",
            "team",
            "opponent",
            "path",
            "path_key",
            "late_path_key",
            "top_prefix",
            "first_team",
            "label",
            "status",
            "note",
            "selected_by_path_ev",
            "probability_source",
            "full_season_ev_reliability",
            "public_field_model",
            "scarcity_spike",
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


def _format_rate_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
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


def _format_optional_rate(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not applicable"
    return _format_rate_pct(value)


def _format_optional_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not applicable"
    return _format_pct(value)


def _format_optional_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not applicable"
    return _format_number(value)


def _format_equity_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
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


def _format_multiple_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"{float(value):.3f}x"


def _format_edge_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"{float(value):+.1%}"


def _format_week_list(value: Any) -> str:
    if value is None:
        return "none"
    weeks = [int(week) for week in value] if isinstance(value, (list, tuple, set)) else []
    if not weeks:
        return "none"
    return ", ".join(f"W{week}" for week in sorted(weeks))


def _format_methods(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "unknown"
    return ", ".join(
        f"{method} ({count})"
        for method, count in sorted(value.items())
    )


def _format_warnings(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "none"
    return " ".join(str(item) for item in value)
