"""Markdown report helpers."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from survivor.config import OUTPUTS_DIR


def generate_basic_weekly_report(
    schedule_rows: Iterable[dict[str, object]],
    public_pick_rows: Iterable[dict[str, object]],
    week: int | None = None,
) -> str:
    """Generate a tiny text report from loaded schedule and public pick rows."""
    schedule = list(schedule_rows)
    public_picks = list(public_pick_rows)

    if week is not None:
        schedule = [row for row in schedule if row.get("week") == week]
        public_picks = [row for row in public_picks if row.get("week") == week]

    popular_picks = sorted(
        public_picks,
        key=lambda row: float(row.get("pick_share", 0)),
        reverse=True,
    )

    lines = [
        "Basic Weekly Survivor Report",
        f"Week: {week if week is not None else 'all'}",
        f"Games loaded: {len(schedule)}",
        f"Public pick rows loaded: {len(public_picks)}",
    ]

    if popular_picks:
        top_pick = popular_picks[0]
        lines.append(
            f"Top public pick: {top_pick['team']} "
            f"({float(top_pick['pick_share']):.1%})"
        )

    return "\n".join(lines)


def build_weekly_report(rankings_df: pd.DataFrame, week: int) -> str:
    """Build a readable markdown report from a ranked weekly DataFrame."""
    if rankings_df.empty:
        raise ValueError("Cannot build a weekly report from empty rankings.")

    rankings = rankings_df.sort_values("rank").reset_index(drop=True)
    top_pick = rankings.iloc[0]
    best_raw = rankings.sort_values(
        "no_vig_win_probability",
        ascending=False,
    ).iloc[0]
    best_leverage = rankings.sort_values("leverage_score", ascending=False).iloc[0]
    best_low_future_cost = rankings.sort_values(
        "scarcity_adjusted_future_cost",
        ascending=True,
    ).iloc[0]
    avoid = rankings.sort_values(
        "scarcity_adjusted_future_cost",
        ascending=False,
    ).head(5)

    lines = [
        f"# Week {week} Survivor Rankings",
        "",
        "## Top 10 Picks",
        "",
        _markdown_table(
            rankings.head(10),
            [
                "rank",
                "team",
                "opponent",
                "no_vig_win_probability",
                "public_pick_pct",
                "leverage_score",
                "scarcity_adjusted_future_cost",
                "final_score",
            ],
        ),
        "",
        "## Why the Top Pick Ranks First",
        "",
        (
            f"{top_pick['team']} ranks first with a "
            f"{top_pick['no_vig_win_probability']:.1%} no-vig win probability, "
            f"{top_pick['public_pick_pct']:.1%} public ownership, "
            f"{top_pick['leverage_score']:.3f} leverage score, and "
            f"{top_pick['scarcity_adjusted_future_cost']:.3f} scarcity-adjusted "
            "future cost."
        ),
        "",
        "## Best Category Picks",
        "",
        (
            f"- Best raw survival pick: {best_raw['team']} "
            f"({best_raw['no_vig_win_probability']:.1%})"
        ),
        (
            f"- Best leverage pick: {best_leverage['team']} "
            f"({best_leverage['leverage_score']:.3f})"
        ),
        (
            f"- Best low-future-cost pick: {best_low_future_cost['team']} "
            f"({best_low_future_cost['scarcity_adjusted_future_cost']:.3f})"
        ),
        "",
        "## High Future-Cost Teams To Avoid",
        "",
        _markdown_table(
            avoid,
            [
                "team",
                "best_future_week",
                "future_cost",
                "scarcity_adjusted_future_cost",
                "replacement_value_summary",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def write_weekly_report(
    rankings_df: pd.DataFrame,
    week: int,
    output_dir: str | Path = OUTPUTS_DIR / "reports",
) -> Path:
    """Write the weekly markdown report and return its path."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"week_{week}_report.md"
    report_path.write_text(build_weekly_report(rankings_df, week), encoding="utf-8")
    return report_path


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[[column for column in columns if column in df.columns]].copy()
    for column in table.columns:
        if column.endswith("probability") or column == "public_pick_pct":
            table[column] = table[column].map(lambda value: f"{float(value):.1%}")
        elif column == "best_future_week":
            table[column] = table[column].map(_format_week_value)
        elif column not in {"rank", "team", "opponent"}:
            table[column] = table[column].map(_format_table_value)

    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.to_dict("records"):
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _format_table_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_week_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(int(value))
