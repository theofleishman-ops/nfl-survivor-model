"""Markdown reports for public behavior calibration runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from survivor.config import OUTPUTS_DIR


def build_public_behavior_calibration_report(
    calibration_df: pd.DataFrame,
    *,
    week: int,
) -> str:
    """Build a Markdown public behavior sensitivity report."""

    if calibration_df.empty:
        raise ValueError("Cannot build a public behavior calibration report without rows.")

    table = calibration_df.copy()
    warning = _sensitivity_warning(table)
    conservative = _interpretation_row(table, "conservative")
    base = _interpretation_row(table, "base")
    aggressive = _interpretation_row(table, "aggressive")
    drivers = _assumption_driver_table(table)

    lines = [
        f"# Week {week} Public Behavior Calibration",
        "",
        "## Sensitivity Table",
        "",
        _markdown_table(
            table,
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
            ],
        ),
        "",
        "## What Drives EV",
        "",
        _markdown_table(
            drivers,
            ["assumption", "correlation_to_ev_multiple", "low_preset", "high_preset"],
        ),
        "",
        "## Sensitivity Warning",
        "",
        warning,
        "",
        "## Interpretation",
        "",
        f"- Conservative: {_format_interpretation(conservative)}",
        f"- Base: {_format_interpretation(base)}",
        f"- Aggressive: {_format_interpretation(aggressive)}",
        "",
        "Use this as an EV range until your pool's historical ownership and "
        "burn patterns are calibrated. A precise point estimate can be less "
        "honest than the spread across plausible public behavior assumptions.",
        "",
        "## Preset Notes",
        "",
        _markdown_table(
            table,
            [
                "preset",
                "chalkiness",
                "randomness",
                "future_awareness",
                "popularity_weight",
                "scarcity_weight",
                "contrarian_rate",
                "max_single_team_ownership",
                "ownership_temperature",
                "description",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def write_public_behavior_calibration_report(
    calibration_df: pd.DataFrame,
    *,
    week: int,
    output_dir: str | Path = OUTPUTS_DIR / "reports",
) -> Path:
    """Write the public behavior calibration report and return its path."""

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"week_{week}_public_behavior_calibration.md"
    report_path.write_text(
        build_public_behavior_calibration_report(calibration_df, week=week),
        encoding="utf-8",
    )
    return report_path


def _sensitivity_warning(df: pd.DataFrame) -> str:
    multiples = pd.to_numeric(df.get("ev_multiple"), errors="coerce").dropna()
    dollars = pd.to_numeric(df.get("ev_dollars"), errors="coerce").dropna()
    if multiples.empty:
        return "EV multiple sensitivity could not be assessed because entry-fee EV was not provided."

    minimum = float(multiples.min())
    maximum = float(multiples.max())
    spread = maximum - minimum
    ratio = maximum / minimum if minimum > 0 else float("inf")
    dollar_spread = float(dollars.max() - dollars.min()) if not dollars.empty else None

    if ratio >= 1.50 or spread >= 0.75:
        suffix = (
            f" Dollar EV spread is {_format_money(dollar_spread)}."
            if dollar_spread is not None
            else ""
        )
        return (
            "Warning: EV is highly sensitive to public behavior assumptions "
            f"({minimum:.3f}x to {maximum:.3f}x).{suffix} Treat the output as "
            "a range, not a single decision-grade point estimate."
        )
    return (
        "EV is reasonably stable across these presets "
        f"({minimum:.3f}x to {maximum:.3f}x), but it should still be read as a range."
    )


def _interpretation_row(df: pd.DataFrame, label: str) -> pd.Series:
    sortable = df.copy()
    sortable["_ev_multiple_sort"] = pd.to_numeric(
        sortable.get("ev_multiple"),
        errors="coerce",
    )
    if sortable["_ev_multiple_sort"].isna().all():
        sortable["_ev_multiple_sort"] = pd.to_numeric(
            sortable.get("path_survival_probability"),
            errors="coerce",
        )
    sortable = sortable.sort_values("_ev_multiple_sort").reset_index(drop=True)
    if label == "conservative":
        return sortable.iloc[0]
    if label == "aggressive":
        return sortable.iloc[-1]
    balanced = sortable[sortable["preset"].astype(str) == "balanced_public"]
    if not balanced.empty:
        return balanced.iloc[0]
    return sortable.iloc[len(sortable) // 2]


def _format_interpretation(row: pd.Series) -> str:
    return (
        f"{row.get('preset', 'unknown')} at {_format_money(row.get('ev_dollars'))}, "
        f"{_format_multiple(row.get('ev_multiple'))}, survival "
        f"{_format_pct(row.get('path_survival_probability'))}."
    )


def _assumption_driver_table(df: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(df.get("ev_multiple"), errors="coerce")
    if target.isna().all():
        target = pd.to_numeric(df.get("ev_dollars"), errors="coerce")
    columns = [
        "chalkiness",
        "randomness",
        "future_awareness",
        "popularity_weight",
        "scarcity_weight",
        "contrarian_rate",
        "max_single_team_ownership",
        "ownership_temperature",
    ]
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
            correlation = 0.0
        else:
            correlation = float(values.corr(target))
            if pd.isna(correlation):
                correlation = 0.0
        low_row = df.loc[values.idxmin()] if not values.dropna().empty else pd.Series(dtype=object)
        high_row = df.loc[values.idxmax()] if not values.dropna().empty else pd.Series(dtype=object)
        rows.append(
            {
                "assumption": column,
                "correlation_to_ev_multiple": correlation,
                "low_preset": low_row.get("preset", ""),
                "high_preset": high_row.get("preset", ""),
                "_abs_correlation": abs(correlation),
            },
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("_abs_correlation", ascending=False)
        .drop(columns=["_abs_correlation"])
        .head(5)
        .reset_index(drop=True)
    )


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    table = df[[column for column in columns if column in df.columns]].copy()
    if table.empty:
        return "_No rows._"
    for column in table.columns:
        if column in {
            "path_survival_probability",
            "average_max_ownership_by_week",
            "average_chalk_concentration",
            "expected_team_exhaustion",
        }:
            table[column] = table[column].map(_format_pct)
        elif column == "ev_dollars":
            table[column] = table[column].map(_format_money)
        elif column == "ev_multiple":
            table[column] = table[column].map(_format_multiple)
        elif column == "correlation_to_ev_multiple":
            table[column] = table[column].map(_format_signed_number)
        elif column in {
            "chalkiness",
            "randomness",
            "future_awareness",
            "popularity_weight",
            "scarcity_weight",
            "contrarian_rate",
            "max_single_team_ownership",
            "ownership_temperature",
            "expected_survivors_if_alive",
            "expected_final_field_size",
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


def _format_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"${float(value):,.2f}"


def _format_multiple(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not provided"
    return f"{float(value):.3f}x"


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _format_signed_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+.3f}"
