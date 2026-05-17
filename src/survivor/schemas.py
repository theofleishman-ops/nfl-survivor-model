"""CSV schema definitions and validation helpers for survivor data imports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from survivor.game_ids import MAX_SEASON, MIN_SEASON, build_game_id, parse_game_id
from survivor.teams import CANONICAL_TEAMS, normalize_team_name


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
    optional_columns=("kickoff", "kickoff_at", "game_window", "season"),
    column_types={
        "week": "positive integer",
        "game_id": "string",
        "home_team": "string",
        "away_team": "string",
        "kickoff": "optional string or timestamp",
        "kickoff_at": "optional kickoff timestamp or TBD marker",
        "game_window": "optional coarse game window label",
        "season": "optional NFL season year",
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

NORMALIZED_ODDS_SCHEMA = CsvSchema(
    required_columns=(
        "season",
        "week",
        "game_id",
        "sportsbook",
        "market_type",
        "team",
        "opponent",
        "home_team",
        "away_team",
        "moneyline",
        "spread",
        "spread_price",
        "total",
        "total_price",
        "implied_probability",
        "no_vig_win_probability",
        "pulled_at",
        "source",
    ),
    column_types={
        "season": "NFL season year",
        "week": "positive integer for game markets",
        "game_id": "canonical game id for game markets",
        "sportsbook": "sportsbook or provider label",
        "market_type": "h2h, spreads, totals, or supported futures market",
        "team": "canonical team or accepted alias",
        "opponent": "opponent team for game markets",
        "home_team": "home team for game markets",
        "away_team": "away team for game markets",
        "moneyline": "optional American moneyline",
        "spread": "optional spread from team's perspective",
        "spread_price": "optional American odds for the spread",
        "total": "optional game total or win total",
        "total_price": "optional American odds for the total",
        "implied_probability": "optional probability from 0 to 1",
        "no_vig_win_probability": "optional no-vig probability from 0 to 1",
        "pulled_at": "timestamp or export time",
        "source": "source/provider descriptor",
    },
)

NORMALIZED_ODDS_COLUMNS = set(NORMALIZED_ODDS_SCHEMA.required_columns)
NORMALIZED_GAME_MARKETS = {"h2h", "moneyline", "spreads", "spread", "totals", "total"}
NORMALIZED_FUTURES_MARKETS = {
    "super_bowl",
    "conference",
    "division",
    "playoff",
    "win_total",
}

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

POOL_HISTORY_CALIBRATION_SCHEMA = CsvSchema(
    required_columns=(
        "season",
        "week",
        "pool_size_start",
        "pool_size_end",
        "top_pick_team",
        "top_pick_pct",
        "second_pick_team",
        "second_pick_pct",
        "third_pick_team",
        "third_pick_pct",
        "notes",
    ),
    column_types={
        "season": "positive integer",
        "week": "positive integer",
        "pool_size_start": "non-negative integer",
        "pool_size_end": "non-negative integer",
        "top_pick_team": "most popular pick team",
        "top_pick_pct": "decimal from 0 to 1",
        "second_pick_team": "second most popular pick team",
        "second_pick_pct": "decimal from 0 to 1",
        "third_pick_team": "third most popular pick team",
        "third_pick_pct": "decimal from 0 to 1",
        "notes": "optional free-text notes",
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

TEAM_STRENGTH_SCHEMA = CsvSchema(
    required_columns=("season", "team", "rating", "source", "notes"),
    column_types={
        "season": "NFL season year",
        "team": "canonical team or accepted alias",
        "rating": "numeric team-strength rating",
        "source": "rating source descriptor",
        "notes": "optional free-text notes",
    },
)

DATASET_SCHEMAS = {
    "schedule": SCHEDULE_SCHEMA,
    "odds": ODDS_SCHEMA,
    "normalized_odds": NORMALIZED_ODDS_SCHEMA,
    "public_picks": PUBLIC_PICKS_SCHEMA,
    "entries": ENTRIES_SCHEMA,
    "entries_history": ENTRIES_HISTORY_SCHEMA,
    "pool_history": POOL_HISTORY_SCHEMA,
    "pool_history_calibration": POOL_HISTORY_CALIBRATION_SCHEMA,
    "double_pick_weeks": DOUBLE_PICK_WEEKS_SCHEMA,
    "team_strength": TEAM_STRENGTH_SCHEMA,
}


def validate_schedule_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for a schedule CSV DataFrame."""
    errors: list[str] = []
    _validate_required_columns(df, SCHEDULE_SCHEMA.required_columns, "schedule", errors)
    _validate_required_values(df, SCHEDULE_SCHEMA.required_columns, "schedule", errors)
    _validate_positive_integer_column(df, "week", "schedule", errors)
    _validate_season_column_if_present(df, "schedule", errors)
    _validate_team_alias_columns(df, ["home_team", "away_team"], "schedule", errors)
    _validate_game_id_rows(df, "schedule", errors)
    _validate_different_teams(df, "home_team", "away_team", "schedule", errors)
    _validate_duplicates(df, ["game_id"], "schedule", errors)
    _validate_duplicates(df, ["week", "game_id"], "schedule", errors)
    _validate_team_week_uniqueness(
        df,
        team_columns=["home_team", "away_team"],
        label="schedule",
        errors=errors,
    )
    return errors


def validate_odds_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for legacy or normalized odds CSV data."""
    if NORMALIZED_ODDS_COLUMNS.issubset(df.columns):
        return _validate_normalized_odds_df(df)
    return _validate_legacy_odds_df(df)


def _validate_legacy_odds_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for a game-level odds CSV DataFrame."""
    errors: list[str] = []
    _validate_required_columns(df, ODDS_SCHEMA.required_columns, "odds", errors)
    _validate_required_values(df, ODDS_SCHEMA.required_columns, "odds", errors)
    _validate_positive_integer_column(df, "week", "odds", errors)
    _validate_moneyline_column(df, "home_moneyline", "odds", errors)
    _validate_moneyline_column(df, "away_moneyline", "odds", errors)
    _validate_team_alias_columns(df, ["home_team", "away_team"], "odds", errors)
    _validate_game_id_rows(df, "odds", errors)
    _validate_different_teams(df, "home_team", "away_team", "odds", errors)
    _validate_probability_columns_if_present(df, "odds", errors)
    _validate_duplicates(df, ["game_id"], "odds", errors)
    _validate_duplicates(df, ["week", "game_id"], "odds", errors)
    _validate_team_week_uniqueness(
        df,
        team_columns=["home_team", "away_team"],
        label="odds",
        errors=errors,
    )
    return errors


def _validate_normalized_odds_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for normalized team-level odds."""
    errors: list[str] = []
    _validate_required_columns(
        df,
        NORMALIZED_ODDS_SCHEMA.required_columns,
        "odds",
        errors,
    )
    _validate_required_values(
        df,
        ("season", "sportsbook", "market_type", "team", "pulled_at", "source"),
        "odds",
        errors,
    )
    _validate_positive_integer_column(df, "season", "odds", errors)
    _validate_positive_integer_column(df, "week", "odds", errors)
    _validate_team_alias_columns(df, ["team", "opponent", "home_team", "away_team"], "odds", errors)
    _validate_moneyline_column(df, "moneyline", "odds", errors)
    _validate_moneyline_column(df, "spread_price", "odds", errors)
    _validate_moneyline_column(df, "total_price", "odds", errors)
    _validate_probability_columns_if_present(df, "odds", errors)
    _validate_normalized_market_types(df, errors)
    _validate_normalized_game_rows(df, errors)
    _validate_duplicates(
        df,
        ["season", "week", "game_id", "sportsbook", "market_type", "team"],
        "odds",
        errors,
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
    _validate_team_alias_columns(df, ["team"], "public_picks", errors)
    _validate_duplicates(df, ["week", "team"], "public_picks", errors)
    if pick_column is not None:
        _validate_public_pick_week_totals(df, pick_column, errors)
    return errors


def validate_season_data_relationships(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame | None = None,
    public_picks_df: pd.DataFrame | None = None,
) -> list[str]:
    """Validate cross-file relationships that single-file schemas cannot prove."""
    errors: list[str] = []
    if odds_df is not None:
        errors.extend(validate_odds_schedule_relationship(odds_df, schedule_df))
    if public_picks_df is not None:
        errors.extend(
            validate_public_picks_schedule_relationship(public_picks_df, schedule_df),
        )
    return errors


def validate_odds_schedule_relationship(
    odds_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> list[str]:
    """Return errors for odds rows that do not join to the schedule."""
    required_schedule = {"week", "game_id", "home_team", "away_team"}
    required_odds = {"week", "game_id", "home_team", "away_team"}
    if not required_schedule.issubset(schedule_df.columns) or not required_odds.issubset(
        odds_df.columns,
    ):
        return []

    schedule_lookup = _schedule_game_lookup(schedule_df)
    missing_rows: list[int] = []
    mismatch_rows: list[int] = []
    missing_examples: list[str] = []

    for index, row in odds_df.iterrows():
        if _is_blank(row.get("game_id")):
            continue
        game_id = str(row["game_id"]).strip()
        schedule_row = schedule_lookup.get(game_id)
        if schedule_row is None:
            missing_rows.append(int(index) + 2)
            missing_examples.append(game_id)
            continue

        try:
            odds_key = (
                int(pd.to_numeric(row["week"], errors="raise")),
                normalize_team_name(row["away_team"]),
                normalize_team_name(row["home_team"]),
            )
        except (TypeError, ValueError):
            continue
        if odds_key != schedule_row:
            mismatch_rows.append(int(index) + 2)

    errors: list[str] = []
    if missing_rows:
        errors.append(
            "odds game_id values must exist in schedule; missing schedule games on "
            f"rows {_format_row_numbers(missing_rows)}"
            f"{_format_examples(missing_examples)}."
        )
    if mismatch_rows:
        errors.append(
            "odds rows must match the schedule week, away_team, and home_team for "
            f"their game_id on rows {_format_row_numbers(mismatch_rows)}."
        )
    return errors


def _validate_normalized_market_types(
    df: pd.DataFrame,
    errors: list[str],
) -> None:
    if "market_type" not in df.columns:
        return

    allowed = NORMALIZED_GAME_MARKETS | NORMALIZED_FUTURES_MARKETS
    invalid_rows = []
    for index, value in df["market_type"].items():
        if _is_blank(value):
            continue
        if str(value).strip().lower() not in allowed:
            invalid_rows.append(int(index) + 2)

    if invalid_rows:
        errors.append(
            "odds column 'market_type' has unsupported values on rows "
            f"{_format_row_numbers(invalid_rows)}. Supported values: "
            f"{', '.join(sorted(allowed))}."
        )


def _validate_normalized_game_rows(
    df: pd.DataFrame,
    errors: list[str],
) -> None:
    required = {"market_type", "week", "game_id", "team", "opponent", "home_team", "away_team"}
    if not required.issubset(df.columns):
        return

    game_market = df["market_type"].astype("string").str.strip().str.lower().isin(
        NORMALIZED_GAME_MARKETS,
    )
    game_rows = df[game_market]
    if game_rows.empty:
        return

    _validate_required_values(
        game_rows,
        ("week", "game_id", "team", "opponent", "home_team", "away_team"),
        "odds",
        errors,
    )
    _validate_game_id_rows(game_rows, "odds", errors)
    _validate_different_teams(game_rows, "team", "opponent", "odds", errors)
    _validate_normalized_team_pairs(game_rows, errors)


def _validate_normalized_team_pairs(
    df: pd.DataFrame,
    errors: list[str],
) -> None:
    invalid_rows: list[int] = []
    for index, row in df.iterrows():
        if any(
            _is_blank(row.get(column))
            for column in ("team", "opponent", "home_team", "away_team")
        ):
            continue
        try:
            team = normalize_team_name(row["team"])
            opponent = normalize_team_name(row["opponent"])
            home = normalize_team_name(row["home_team"])
            away = normalize_team_name(row["away_team"])
        except ValueError:
            continue
        if {team, opponent} != {home, away}:
            invalid_rows.append(int(index) + 2)

    if invalid_rows:
        errors.append(
            "odds team/opponent must match home_team/away_team on rows "
            f"{_format_row_numbers(invalid_rows)}."
        )


def validate_public_picks_schedule_relationship(
    public_picks_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> list[str]:
    """Return errors for public pick teams that are not scheduled that week."""
    required_schedule = {"week", "home_team", "away_team"}
    required_public = {"week", "team"}
    if not required_schedule.issubset(schedule_df.columns) or not required_public.issubset(
        public_picks_df.columns,
    ):
        return []

    scheduled_teams = _scheduled_teams_by_week(schedule_df)
    unscheduled_rows: list[int] = []
    unscheduled_examples: list[str] = []
    for index, row in public_picks_df.iterrows():
        if _is_blank(row.get("week")) or _is_blank(row.get("team")):
            continue
        try:
            week = int(pd.to_numeric(row["week"], errors="raise"))
            team = normalize_team_name(row["team"])
        except (TypeError, ValueError):
            continue
        if team not in scheduled_teams.get(week, set()):
            unscheduled_rows.append(int(index) + 2)
            unscheduled_examples.append(f"W{week}:{team}")

    if not unscheduled_rows:
        return []
    return [
        "public_picks teams must appear in the schedule for the same week; "
        f"unscheduled picks on rows {_format_row_numbers(unscheduled_rows)}"
        f"{_format_examples(unscheduled_examples)}."
    ]


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
    _validate_team_alias_columns(df, ["most_popular_pick"], "pool_history", errors)
    _validate_probability_column(
        df,
        "most_popular_pick_share",
        "pool_history",
        errors,
    )
    _validate_entries_survived_not_greater_than_start(df, errors)
    _validate_duplicates(df, ["season", "week"], "pool_history", errors)
    return errors


def validate_pool_history_calibration_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for future public-behavior calibration history."""

    errors: list[str] = []
    required = POOL_HISTORY_CALIBRATION_SCHEMA.required_columns
    _validate_required_columns(df, required, "pool_history_calibration", errors)
    _validate_required_values(
        df,
        (
            "season",
            "week",
            "pool_size_start",
            "pool_size_end",
            "top_pick_team",
            "top_pick_pct",
            "second_pick_team",
            "second_pick_pct",
            "third_pick_team",
            "third_pick_pct",
        ),
        "pool_history_calibration",
        errors,
    )
    _validate_positive_integer_column(df, "season", "pool_history_calibration", errors)
    _validate_positive_integer_column(df, "week", "pool_history_calibration", errors)
    _validate_non_negative_integer_column(
        df,
        "pool_size_start",
        "pool_history_calibration",
        errors,
    )
    _validate_non_negative_integer_column(
        df,
        "pool_size_end",
        "pool_history_calibration",
        errors,
    )
    _validate_team_alias_columns(
        df,
        ["top_pick_team", "second_pick_team", "third_pick_team"],
        "pool_history_calibration",
        errors,
    )
    for column in ("top_pick_pct", "second_pick_pct", "third_pick_pct"):
        _validate_probability_column(df, column, "pool_history_calibration", errors)
    _validate_pool_history_calibration_counts(df, errors)
    _validate_duplicates(df, ["season", "week"], "pool_history_calibration", errors)
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


def validate_team_strength_df(df: pd.DataFrame) -> list[str]:
    """Return validation errors for season-level team-strength ratings."""
    errors: list[str] = []
    required = TEAM_STRENGTH_SCHEMA.required_columns
    _validate_required_columns(df, required, "team_strength", errors)
    _validate_required_values(df, ("season", "team", "rating", "source"), "team_strength", errors)
    _validate_season_column_if_present(df, "team_strength", errors)
    _validate_team_alias_columns(df, ["team"], "team_strength", errors)
    _validate_numeric_column(df, "rating", "team_strength", errors)
    _validate_team_strength_roster(df, errors)
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
    _validate_semicolon_team_list_column(df, "used_teams", "entries", errors)
    _validate_duplicates(df, ["entry_id"], "entries", errors)
    return errors


def _validate_entry_history_df(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = ENTRIES_HISTORY_SCHEMA.required_columns
    _validate_required_columns(df, required, "entries", errors)
    _validate_required_values(df, required, "entries", errors)
    _validate_positive_integer_column(df, "week", "entries", errors)
    _validate_team_alias_columns(df, ["team_picked"], "entries", errors)
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


def _validate_season_column_if_present(
    df: pd.DataFrame,
    label: str,
    errors: list[str],
) -> None:
    if "season" not in df.columns:
        return

    _validate_required_values(df, ("season",), label, errors)
    values = pd.to_numeric(df["season"], errors="coerce")
    present = ~df["season"].map(_is_blank)
    invalid_numeric = present & values.isna()
    if bool(invalid_numeric.any()):
        errors.append(
            f"{label} column 'season' must be numeric on rows "
            f"{_format_rows(invalid_numeric)}."
        )
        return

    invalid_integer = present & values.notna() & ((values % 1) != 0)
    if bool(invalid_integer.any()):
        errors.append(
            f"{label} column 'season' must contain whole numbers on rows "
            f"{_format_rows(invalid_integer)}."
        )

    invalid_range = present & values.notna() & ~values.between(MIN_SEASON, MAX_SEASON)
    if bool(invalid_range.any()):
        errors.append(
            f"{label} column 'season' must be between {MIN_SEASON} and "
            f"{MAX_SEASON} on rows {_format_rows(invalid_range)}."
        )


def _validate_team_alias_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
    errors: list[str],
) -> None:
    for column in columns:
        if column not in df.columns:
            continue

        present = ~df[column].map(_is_blank)
        invalid_rows = []
        for index, value in df.loc[present, column].items():
            try:
                normalize_team_name(value)
            except ValueError:
                invalid_rows.append(int(index) + 2)

        if invalid_rows:
            errors.append(
                f"{label} column '{column}' has invalid team aliases on rows "
                f"{_format_row_numbers(invalid_rows)}."
            )


def _validate_semicolon_team_list_column(
    df: pd.DataFrame,
    column: str,
    label: str,
    errors: list[str],
) -> None:
    if column not in df.columns:
        return

    invalid_rows = []
    for index, value in df[column].items():
        if _is_blank(value):
            continue
        teams = [team.strip() for team in str(value).split(";") if team.strip()]
        try:
            for team in teams:
                normalize_team_name(team)
        except ValueError:
            invalid_rows.append(int(index) + 2)

    if invalid_rows:
        errors.append(
            f"{label} column '{column}' has invalid team aliases on rows "
            f"{_format_row_numbers(invalid_rows)}."
        )


def _validate_game_id_rows(
    df: pd.DataFrame,
    label: str,
    errors: list[str],
) -> None:
    if "game_id" not in df.columns:
        return

    malformed_rows: list[int] = []
    mismatch_rows: list[int] = []
    season_mismatch_rows: list[int] = []

    for index, row in df.iterrows():
        if _is_blank(row.get("game_id")):
            continue

        row_number = int(index) + 2
        try:
            parsed = parse_game_id(row["game_id"])
        except ValueError:
            malformed_rows.append(row_number)
            continue

        if {"week", "home_team", "away_team"}.issubset(df.columns):
            try:
                expected = build_game_id(
                    season=int(parsed["season"]),
                    week=int(pd.to_numeric(row["week"], errors="raise")),
                    away_team=normalize_team_name(row["away_team"]),
                    home_team=normalize_team_name(row["home_team"]),
                )
            except (TypeError, ValueError):
                continue

            if str(row["game_id"]).strip() != expected:
                mismatch_rows.append(row_number)

        if "season" in df.columns and not _is_blank(row.get("season")):
            try:
                season = int(pd.to_numeric(row["season"], errors="raise"))
            except (TypeError, ValueError):
                continue
            if season != int(parsed["season"]):
                season_mismatch_rows.append(row_number)

    if malformed_rows:
        errors.append(
            f"{label} column 'game_id' has malformed canonical game IDs on rows "
            f"{_format_row_numbers(malformed_rows)}. Expected format "
            "2026_W01_BAL_AT_KC."
        )
    if mismatch_rows:
        errors.append(
            f"{label} column 'game_id' must match week, away_team, and home_team "
            f"on rows {_format_row_numbers(mismatch_rows)}."
        )
    if season_mismatch_rows:
        errors.append(
            f"{label} column 'game_id' season conflicts with column 'season' on rows "
            f"{_format_row_numbers(season_mismatch_rows)}."
        )


def _validate_different_teams(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    label: str,
    errors: list[str],
) -> None:
    if left_column not in df.columns or right_column not in df.columns:
        return

    invalid_rows: list[int] = []
    for index, row in df.iterrows():
        if _is_blank(row[left_column]) or _is_blank(row[right_column]):
            continue
        try:
            left = normalize_team_name(row[left_column])
            right = normalize_team_name(row[right_column])
        except ValueError:
            continue
        if left == right:
            invalid_rows.append(int(index) + 2)

    if invalid_rows:
        errors.append(
            f"{label} columns '{left_column}' and '{right_column}' must be "
            f"different teams on rows {_format_row_numbers(invalid_rows)}."
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


def _validate_numeric_column(
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
            f"{label} column '{column}' must be numeric on rows "
            f"{_format_rows(invalid_numeric)}."
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


def _validate_public_pick_week_totals(
    df: pd.DataFrame,
    pick_column: str,
    errors: list[str],
) -> None:
    required = {"week", "team", pick_column}
    if not required.issubset(df.columns):
        return

    values = pd.to_numeric(df[pick_column], errors="coerce")
    weeks = pd.to_numeric(df["week"], errors="coerce")
    valid = values.notna() & weeks.notna()
    if not bool(valid.any()):
        return

    totals = values.loc[valid].groupby(weeks.loc[valid].astype(int)).sum()
    invalid_weeks = [int(week) for week, total in totals.items() if float(total) > 1.000001]
    if invalid_weeks:
        errors.append(
            "public_picks public_pick_pct totals cannot exceed 1.0 within a week; "
            f"invalid weeks: {', '.join(str(week) for week in invalid_weeks[:5])}"
            f"{', and ' + str(len(invalid_weeks) - 5) + ' more' if len(invalid_weeks) > 5 else ''}."
        )


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
    teams["team"] = teams["team"].map(_normalize_team_or_original)
    teams["week_text"] = teams["week"].astype("string").str.strip()
    duplicate_mask = teams.duplicated(subset=["week_text", "team"], keep=False)
    if not bool(duplicate_mask.any()):
        return

    duplicate_rows = sorted({int(row) + 2 for row in teams.loc[duplicate_mask, "source_row"]})
    errors.append(
        f"{label} has teams appearing more than once in the same week on rows "
        f"{_format_row_numbers(duplicate_rows)}."
    )


def _validate_team_strength_roster(
    df: pd.DataFrame,
    errors: list[str],
) -> None:
    required = {"season", "team"}
    if not required.issubset(df.columns):
        return

    rows: list[dict[str, int | str]] = []
    for index, row in df.iterrows():
        if _is_blank(row.get("season")) or _is_blank(row.get("team")):
            continue
        try:
            season = int(pd.to_numeric(row["season"], errors="raise"))
            team = normalize_team_name(row["team"])
        except (TypeError, ValueError):
            continue
        rows.append({"season": season, "team": team, "source_row": int(index)})

    if not rows:
        return

    roster = pd.DataFrame(rows)
    duplicate_mask = roster.duplicated(subset=["season", "team"], keep=False)
    if bool(duplicate_mask.any()):
        duplicate_rows = sorted(
            {int(row) + 2 for row in roster.loc[duplicate_mask, "source_row"]},
        )
        errors.append(
            "team_strength must have exactly one row per team per season; "
            f"duplicate rows {_format_row_numbers(duplicate_rows)}."
        )

    expected = set(CANONICAL_TEAMS)
    for season, season_df in roster.groupby("season", sort=True):
        present = set(season_df["team"].astype(str))
        missing = sorted(expected - present)
        if missing:
            shown = ", ".join(missing[:10])
            suffix = f", and {len(missing) - 10} more" if len(missing) > 10 else ""
            errors.append(
                "team_strength must include exactly one row for every NFL team "
                f"in season {int(season)}; missing: {shown}{suffix}."
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


def _validate_pool_history_calibration_counts(
    df: pd.DataFrame,
    errors: list[str],
) -> None:
    columns = {"pool_size_start", "pool_size_end"}
    if not columns.issubset(df.columns):
        return

    start = pd.to_numeric(df["pool_size_start"], errors="coerce")
    end = pd.to_numeric(df["pool_size_end"], errors="coerce")
    valid_numeric = start.notna() & end.notna()
    invalid = valid_numeric & (end > start)
    if bool(invalid.any()):
        errors.append(
            "pool_history_calibration pool_size_end cannot exceed pool_size_start "
            f"on rows {_format_rows(invalid)}."
        )


def _schedule_game_lookup(df: pd.DataFrame) -> dict[str, tuple[int, str, str]]:
    lookup: dict[str, tuple[int, str, str]] = {}
    for _, row in df.iterrows():
        if _is_blank(row.get("game_id")):
            continue
        try:
            lookup[str(row["game_id"]).strip()] = (
                int(pd.to_numeric(row["week"], errors="raise")),
                normalize_team_name(row["away_team"]),
                normalize_team_name(row["home_team"]),
            )
        except (TypeError, ValueError):
            continue
    return lookup


def _scheduled_teams_by_week(df: pd.DataFrame) -> dict[int, set[str]]:
    teams_by_week: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        try:
            week = int(pd.to_numeric(row["week"], errors="raise"))
            home = normalize_team_name(row["home_team"])
            away = normalize_team_name(row["away_team"])
        except (TypeError, ValueError):
            continue
        teams_by_week.setdefault(week, set()).update({home, away})
    return teams_by_week


def _public_pick_column(df: pd.DataFrame) -> str | None:
    for column in PUBLIC_PICK_ALIASES:
        if column in df.columns:
            return column
    return None


def _normalize_team_or_original(value: object) -> str:
    try:
        return normalize_team_name(value)
    except ValueError:
        return str(value).strip()


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


def _format_examples(values: list[str], limit: int = 3) -> str:
    if not values:
        return ""
    unique_values = list(dict.fromkeys(values))
    shown = ", ".join(unique_values[:limit])
    suffix = (
        f", and {len(unique_values) - limit} more"
        if len(unique_values) > limit
        else ""
    )
    return f" ({shown}{suffix})"
