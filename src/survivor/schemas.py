"""CSV schema definitions and validation helpers for survivor data imports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CsvSchema:
    """Readable schema metadata for a real-data CSV."""

    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    column_types: dict[str, str] | None = None


PUBLIC_PICK_ALIASES = (
    "public_pick_pct",
    "pick_share",
    "pick_pct",
    "ownership",
    "ownership_pct",
)

BOOL_TRUE_VALUES = {"true", "t", "1", "yes", "y", "active", "alive"}
BOOL_FALSE_VALUES = {"false", "f", "0", "no", "n", "inactive", "dead"}

SCHEDULE_SCHEMA = CsvSchema(
    required_columns=("week", "game_id", "home_team", "away_team"),
    optional_columns=("kickoff",),
    column_types={
        "week": "positive integer",
        "game_id": "string",
        "home_team": "string",
        "away_team": "string",
        "kickoff": "optional string or timestamp",
    },
)

ODDS_SCHEMA = CsvSchema(
    required_columns=(
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_moneyline",
        "away_moneyline",
    ),
    optional_columns=("sportsbook", "market_timestamp"),
    column_types={
        "week": "positive integer",
        "game_id": "string",
        "home_team": "string",
        "away_team": "string",
        "home_moneyline": "American moneyline integer, e.g. -350 or +220",
        "away_moneyline": "American moneyline integer, e.g. -350 or +220",
    },
)

PUBLIC_PICKS_SCHEMA = CsvSchema(
    required_columns=("week", "team", "public_pick_pct"),
    optional_columns=("source",),
    column_types={
        "week": "positive integer",
        "team": "string",
        "public_pick_pct": "decimal from 0 to 1, e.g. 0.38 for 38%",
    },
)

ENTRIES_SCHEMA = CsvSchema(
    required_columns=("entry_id", "active", "used_teams"),
    optional_columns=("owner_name",),
    column_types={
        "entry_id": "string",
        "owner_name": "optional string",
        "active": "boolean-ish value such as true, false, yes, no, active, inactive",
        "used_teams": "semicolon-separated team list, blank when no teams are used",
    },
)

ENTRIES_HISTORY_SCHEMA = CsvSchema(
    required_columns=("entry_id", "week", "team_picked", "is_alive"),
    optional_columns=("owner_name",),
    column_types={
        "entry_id": "string",
        "owner_name": "optional string",
        "week": "positive integer",
        "team_picked": "string",
        "is_alive": "boolean-ish value such as true, false, yes, no",
    },
)

POOL_HISTORY_SCHEMA = CsvSchema(
    required_columns=(
        "season",
        "week",
        "entries_start",
        "entries_survived",
        "most_popular_pick",
        "most_popular_pick_share",
    ),
    column_types={
        "season": "positive integer",
        "week": "positive integer",
        "entries_start": "non-negative integer",
        "entries_survived": "non-negative integer",
        "most_popular_pick": "string",
        "most_popular_pick_share": "decimal from 0 to 1",
    },
)

DOUBLE_PICK_WEEKS_SCHEMA = CsvSchema(
    required_columns=("week", "required_picks"),
    optional_columns=("note",),
    column_types={
        "week": "positive integer",
        "required_picks": "integer of 2 or greater",
        "note": "optional string",
    },
)

DATASET_SCHEMAS = {
    "schedule": SCHEDULE_SCHEMA,
    "odds": ODDS_SCHEMA,
    "public_picks": PUBLIC_PICKS_SCHEMA,
    "entries": ENTRIES_SCHEMA,
    "entries_history": ENTRIES_HISTORY_SCHEMA,
    "pool_history": POOL_HISTORY_SCHEMA,
    "double_pick_weeks": DOUBLE_PICK_WEEKS_SCHEMA,
}


def validate_schedule_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for a schedule CSV DataFrame."""
    errors: list[str] = []
    _validate_required_columns(df, SCHEDULE_SCHEMA.required_columns, "schedule", errors)
    _validate_required_values(df, SCHEDULE_SCHEMA.required_columns, "schedule", errors)
    _validate_positive_integer_column(df, "week", "schedule", errors)
    _validate_different_columns(df, "home_team", "away_team", "schedule", errors)
    _validate_duplicates(df, ["week", "game_id"], "schedule", errors)
    _validate_team_week_uniqueness(
        df,
        team_columns=["home_team", "away_team"],
        label="schedule",
        errors=errors,
    )
    return errors


def validate_odds_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for a game-level odds CSV DataFrame."""
    errors: list[str] = []
    _validate_required_columns(df, ODDS_SCHEMA.required_columns, "odds", errors)
    _validate_required_values(df, ODDS_SCHEMA.required_columns, "odds", errors)
    _validate_positive_integer_column(df, "week", "odds", errors)
    _validate_moneyline_column(df, "home_moneyline", "odds", errors)
    _validate_moneyline_column(df, "away_moneyline", "odds", errors)
    _validate_different_columns(df, "home_team", "away_team", "odds", errors)
    _validate_probability_columns_if_present(df, "odds", errors)
    _validate_duplicates(df, ["week", "game_id"], "odds", errors)
    _validate_team_week_uniqueness(
        df,
        team_columns=["home_team", "away_team"],
        label="odds",
        errors=errors,
    )
    return errors


def validate_public_picks_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for public pick ownership by week/team."""
    errors: list[str] = []
    _validate_required_columns(df, ("week", "team"), "public_picks", errors)
    pick_column = _public_pick_column(df)
    if pick_column is None:
        aliases = ", ".join(PUBLIC_PICK_ALIASES)
        errors.append(
            "public_picks needs one public pick percentage column. "
            f"Use public_pick_pct; accepted aliases are: {aliases}."
        )
    else:
        _validate_probability_column(
            df,
            pick_column,
            "public_picks",
            errors,
            display_name="public_pick_pct",
        )

    _validate_required_values(df, ("week", "team"), "public_picks", errors)
    if pick_column is not None:
        _validate_required_values(df, (pick_column,), "public_picks", errors)
    _validate_positive_integer_column(df, "week", "public_picks", errors)
    _validate_duplicates(df, ["week", "team"], "public_picks", errors)
    return errors


def validate_entries_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for supported entries CSV formats.

    Phase 4 real-data workspaces use the state format:
    ``entry_id, owner_name, active, used_teams``.

    The original sample workflow still supports historical pick rows:
    ``entry_id, owner_name, week, team_picked, is_alive``.
    """
    columns = set(df.columns)
    if set(ENTRIES_SCHEMA.required_columns).issubset(columns):
        return _validate_entry_state_df(df)
    if set(ENTRIES_HISTORY_SCHEMA.required_columns).issubset(columns):
        return _validate_entry_history_df(df)

    state_missing = sorted(set(ENTRIES_SCHEMA.required_columns) - columns)
    history_missing = sorted(set(ENTRIES_HISTORY_SCHEMA.required_columns) - columns)
    return [
        "entries must match one supported schema: "
        "state columns "
        f"({', '.join(ENTRIES_SCHEMA.required_columns)}) or historical pick columns "
        f"({', '.join(ENTRIES_HISTORY_SCHEMA.required_columns)}). "
        f"Missing for state format: {', '.join(state_missing) or 'none'}; "
        f"missing for historical format: {', '.join(history_missing) or 'none'}."
    ]


def validate_pool_history_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for historical pool-size summaries."""
    errors: list[str] = []
    required = POOL_HISTORY_SCHEMA.required_columns
    _validate_required_columns(df, required, "pool_history", errors)
    _validate_required_values(df, required, "pool_history", errors)
    _validate_positive_integer_column(df, "season", "pool_history", errors)
    _validate_positive_integer_column(df, "week", "pool_history", errors)
    _validate_non_negative_integer_column(df, "entries_start", "pool_history", errors)
    _validate_non_negative_integer_column(
        df,
        "entries_survived",
        "pool_history",
        errors,
    )
    _validate_probability_column(
        df,
        "most_popular_pick_share",
        "pool_history",
        errors,
    )
    _validate_entries_survived_not_greater_than_start(df, errors)
    _validate_duplicates(df, ["season", "week"], "pool_history", errors)
    return errors


def validate_double_pick_weeks_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for pool weeks requiring multiple picks."""
    errors: list[str] = []
    required = DOUBLE_PICK_WEEKS_SCHEMA.required_columns
    _validate_required_columns(df, required, "double_pick_weeks", errors)
    _validate_required_values(df, required, "double_pick_weeks", errors)
    _validate_positive_integer_column(df, "week", "double_pick_weeks", errors)
    _validate_min_integer_column(
        df,
        "required_picks",
        minimum=2,
        label="double_pick_weeks",
        errors=errors,
    )
    _validate_duplicates(df, ["week"], "double_pick_weeks", errors)
    return errors


def validate_or_raise(
    dataset_name: str,
    df: pd.DataFrame,
    errors: list[str],
    path: str | None = None,
) -> None:
    """Raise ``ValueError`` with actionable validation messages when needed."""
    if not errors:
        return

    location = f" in {path}" if path else ""
    details = "\n".join(f"- {error}" for error in errors)
    raise ValueError(f"{dataset_name} validation failed{location}:\n{details}")


def is_boolish(value: Any) -> bool:
    """Return whether ``value`` can be parsed as a user-entered boolean."""
    if isinstance(value, bool):
        return True
    if _is_blank(value):
        return False
    normalized = str(value).strip().lower()
    return normalized in BOOL_TRUE_VALUES or normalized in BOOL_FALSE_VALUES


def parse_boolish(value: Any) -> bool:
    """Parse a boolean-ish CSV value or raise a clear ``ValueError``."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in BOOL_TRUE_VALUES:
        return True
    if normalized in BOOL_FALSE_VALUES:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")


def _validate_entry_state_df(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    _validate_required_columns(df, ENTRIES_SCHEMA.required_columns, "entries", errors)
    _validate_required_values(df, ("entry_id", "active"), "entries", errors)
    _validate_boolish_column(df, "active", "entries", errors)
    _validate_duplicates(df, ["entry_id"], "entries", errors)
    return errors


def _validate_entry_history_df(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = ENTRIES_HISTORY_SCHEMA.required_columns
    _validate_required_columns(df, required, "entries", errors)
    _validate_required_values(df, required, "entries", errors)
    _validate_positive_integer_column(df, "week", "entries", errors)
    _validate_boolish_column(df, "is_alive", "entries", errors)
    _validate_duplicates(df, ["entry_id", "week"], "entries", errors)
    return errors


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    label: str,
    errors: list[str],
) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        errors.append(f"{label} missing required columns: {', '.join(missing)}.")


def _validate_required_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    errors: list[str],
) -> None:
    for column in columns:
        if column not in df.columns:
            continue
        missing_mask = df[column].map(_is_blank)
        if bool(missing_mask.any()):
            errors.append(
                f"{label} column '{column}' has blank required values on rows "
                f"{_format_rows(missing_mask)}."
            )


def _validate_positive_integer_column(
    df: pd.DataFrame,
    column: str,
    label: str,
    errors: list[str],
) -> None:
    _validate_min_integer_column(df, column, minimum=1, label=label, errors=errors)


def _validate_non_negative_integer_column(
    df: pd.DataFrame,
    column: str,
    label: str,
    errors: list[str],
) -> None:
    _validate_min_integer_column(df, column, minimum=0, label=label, errors=errors)


def _validate_min_integer_column(
    df: pd.DataFrame,
    column: str,
    minimum: int,
    label: str,
    errors: list[str],
) -> None:
    if column not in df.columns:
        return

    values = pd.to_numeric(df[column], errors="coerce")
    present = ~df[column].map(_is_blank)
    invalid_numeric = present & values.isna()
    if bool(invalid_numeric.any()):
        errors.append(
            f"{label} column '{column}' must be numeric on rows "
            f"{_format_rows(invalid_numeric)}."
        )
        return

    invalid_integer = present & values.notna() & ((values % 1) != 0)
    if bool(invalid_integer.any()):
        errors.append(
            f"{label} column '{column}' must contain whole numbers on rows "
            f"{_format_rows(invalid_integer)}."
        )

    invalid_range = present & values.notna() & (values < minimum)
    if bool(invalid_range.any()):
        comparator = "positive" if minimum == 1 else f"at least {minimum}"
        errors.append(
            f"{label} column '{column}' must be {comparator} on rows "
            f"{_format_rows(invalid_range)}."
        )


def _validate_moneyline_column(
    df: pd.DataFrame,
    column: str,
    label: str,
    errors: list[str],
) -> None:
    if column not in df.columns:
        return

    values = pd.to_numeric(df[column], errors="coerce")
    present = ~df[column].map(_is_blank)
    invalid_numeric = present & values.isna()
    if bool(invalid_numeric.any()):
        errors.append(
            f"{label} column '{column}' must be numeric American odds on rows "
            f"{_format_rows(invalid_numeric)}."
        )
        return

    invalid_integer = present & values.notna() & ((values % 1) != 0)
    if bool(invalid_integer.any()):
        errors.append(
            f"{label} column '{column}' must be whole-number American odds on rows "
            f"{_format_rows(invalid_integer)}."
        )

    invalid_range = present & values.notna() & ~((values <= -100) | (values >= 100))
    if bool(invalid_range.any()):
        errors.append(
            f"{label} column '{column}' must be American odds <= -100 or >= 100 "
            f"(for example -350 or +220) on rows {_format_rows(invalid_range)}."
        )


def _validate_probability_column(
    df: pd.DataFrame,
    column: str,
    label: str,
    errors: list[str],
    display_name: str | None = None,
) -> None:
    if column not in df.columns:
        return

    shown_column = display_name or column
    values = pd.to_numeric(df[column], errors="coerce")
    present = ~df[column].map(_is_blank)
    invalid_numeric = present & values.isna()
    if bool(invalid_numeric.any()):
        errors.append(
            f"{label} column '{shown_column}' must be numeric on rows "
            f"{_format_rows(invalid_numeric)}."
        )
        return

    invalid_range = present & values.notna() & ~values.between(0, 1)
    if bool(invalid_range.any()):
        errors.append(
            f"{label} column '{shown_column}' must be between 0 and 1 on rows "
            f"{_format_rows(invalid_range)}."
        )


def _validate_probability_columns_if_present(
    df: pd.DataFrame,
    label: str,
    errors: list[str],
) -> None:
    probability_columns = [
        column
        for column in df.columns
        if column.endswith("_probability") or column.endswith("_win_probability")
    ]
    for column in probability_columns:
        _validate_probability_column(df, column, label, errors)


def _validate_boolish_column(
    df: pd.DataFrame,
    column: str,
    label: str,
    errors: list[str],
) -> None:
    if column not in df.columns:
        return

    present = ~df[column].map(_is_blank)
    invalid = present & ~df[column].map(is_boolish)
    if bool(invalid.any()):
        errors.append(
            f"{label} column '{column}' must be boolean-ish "
            f"(true/false, yes/no, active/inactive) on rows {_format_rows(invalid)}."
        )


def _validate_different_columns(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    label: str,
    errors: list[str],
) -> None:
    if left_column not in df.columns or right_column not in df.columns:
        return

    left = df[left_column].astype("string").str.strip()
    right = df[right_column].astype("string").str.strip()
    invalid = left.notna() & right.notna() & (left == right)
    if bool(invalid.any()):
        errors.append(
            f"{label} columns '{left_column}' and '{right_column}' must be different "
            f"on rows {_format_rows(invalid)}."
        )


def _validate_duplicates(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not set(columns).issubset(df.columns):
        return

    comparable = df[columns].copy()
    for column in columns:
        comparable[column] = comparable[column].astype("string").str.strip()
    duplicate_mask = comparable.duplicated(subset=columns, keep=False)
    if bool(duplicate_mask.any()):
        errors.append(
            f"{label} has duplicate rows for {', '.join(columns)} on rows "
            f"{_format_rows(duplicate_mask)}."
        )


def _validate_team_week_uniqueness(
    df: pd.DataFrame,
    team_columns: list[str],
    label: str,
    errors: list[str],
) -> None:
    required = {"week", *team_columns}
    if not required.issubset(df.columns):
        return

    team_frames = []
    for column in team_columns:
        frame = df[["week", column]].rename(columns={column: "team"}).copy()
        frame["source_row"] = df.index
        team_frames.append(frame)

    teams = pd.concat(team_frames, ignore_index=True)
    teams["team"] = teams["team"].astype("string").str.strip()
    teams = teams[teams["team"].notna() & (teams["team"] != "")]
    teams["week_text"] = teams["week"].astype("string").str.strip()
    duplicate_mask = teams.duplicated(subset=["week_text", "team"], keep=False)
    if not bool(duplicate_mask.any()):
        return

    duplicate_rows = sorted({int(row) + 2 for row in teams.loc[duplicate_mask, "source_row"]})
    errors.append(
        f"{label} has teams appearing more than once in the same week on rows "
        f"{_format_row_numbers(duplicate_rows)}."
    )


def _validate_entries_survived_not_greater_than_start(
    df: pd.DataFrame,
    errors: list[str],
) -> None:
    columns = {"entries_start", "entries_survived"}
    if not columns.issubset(df.columns):
        return

    start = pd.to_numeric(df["entries_start"], errors="coerce")
    survived = pd.to_numeric(df["entries_survived"], errors="coerce")
    valid_numeric = start.notna() & survived.notna()
    invalid = valid_numeric & (survived > start)
    if bool(invalid.any()):
        errors.append(
            "pool_history entries_survived cannot exceed entries_start on rows "
            f"{_format_rows(invalid)}."
        )


def _public_pick_column(df: pd.DataFrame) -> str | None:
    for column in PUBLIC_PICK_ALIASES:
        if column in df.columns:
            return column
    return None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and value.strip() == ""


def _format_rows(mask: pd.Series) -> str:
    row_numbers = [int(index) + 2 for index, invalid in mask.items() if bool(invalid)]
    return _format_row_numbers(row_numbers)


def _format_row_numbers(row_numbers: list[int], limit: int = 5) -> str:
    shown = row_numbers[:limit]
    suffix = f", and {len(row_numbers) - limit} more" if len(row_numbers) > limit else ""
    return ", ".join(str(row) for row in shown) + suffix
