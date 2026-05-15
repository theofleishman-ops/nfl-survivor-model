"""CSV loaders for the local Phase 1 survivor model.

The loaders read small sample CSVs today and intentionally accept the same
schema for future manually exported real CSVs.  They normalize column names and
types at the boundary so the scoring code can work with predictable
DataFrames.
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_object_dtype, is_string_dtype

from survivor.config import SAMPLE_DATA_DIR
from survivor.game_ids import build_game_id
from survivor.schemas import (
    parse_boolish,
    validate_double_pick_weeks_df,
    validate_entries_df,
    validate_odds_df,
    validate_or_raise,
    validate_pool_history_df,
    validate_public_picks_df,
    validate_schedule_df,
    validate_season_data_relationships,
)
from survivor.teams import normalize_team_name


ENTRY_STATE_COLUMNS = {"entry_id", "active", "used_teams"}
PUBLIC_PICK_ALIASES = (
    "public_pick_pct",
    "pick_share",
    "pick_pct",
    "ownership",
    "ownership_pct",
)
SEASON_FILE_NAMES = {
    "schedule": "schedule.csv",
    "odds": "odds.csv",
    "public_picks": "public_picks.csv",
    "entries": "entries.csv",
    "pool_history": "pool_history.csv",
    "double_pick_weeks": "double_pick_weeks.csv",
}


def load_schedule_df(path: str | Path) -> pd.DataFrame:
    """Load a schedule CSV with one row per game."""
    df = _read_csv(path)
    validate_or_raise("schedule", df, validate_schedule_df(df), str(path))
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    return _strip_string_columns(df).sort_values(["week", "game_id"]).reset_index(
        drop=True,
    )


def load_odds_df(path: str | Path) -> pd.DataFrame:
    """Load game-level American moneyline odds from CSV."""
    df = _read_csv(path)
    validate_or_raise("odds", df, validate_odds_df(df), str(path))
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="raise")
    df["away_moneyline"] = pd.to_numeric(df["away_moneyline"], errors="raise")
    return _strip_string_columns(df).sort_values(["week", "game_id"]).reset_index(
        drop=True,
    )


def load_public_picks_df(path: str | Path) -> pd.DataFrame:
    """Load public pick ownership by week/team.

    The percentage column may be named ``public_pick_pct`` or one of the common
    aliases in :data:`PUBLIC_PICK_ALIASES`.  Values must be stored as decimals
    (``0.24`` for 24%).
    """
    df = _read_csv(path)
    validate_or_raise("public_picks", df, validate_public_picks_df(df), str(path))
    public_col = _find_public_pick_column(df, path)

    df = df.rename(columns={public_col: "public_pick_pct"})
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["team"] = df["team"].map(normalize_team_name)
    df["public_pick_pct"] = pd.to_numeric(df["public_pick_pct"], errors="raise").astype(
        float,
    )
    return _strip_string_columns(df[["week", "team", "public_pick_pct"]]).sort_values(
        ["week", "team"],
    ).reset_index(drop=True)


def load_entries_df(path: str | Path) -> pd.DataFrame:
    """Load survivor entry data.

    The original sample files use historical pick rows.  Real-data workspaces
    may instead use current entry state rows with ``active`` and ``used_teams``.
    """
    df = _read_csv(path)
    validate_or_raise("entries", df, validate_entries_df(df), str(path))

    if ENTRY_STATE_COLUMNS.issubset(df.columns):
        df["active"] = df["active"].map(_parse_bool)
        df["used_teams"] = df["used_teams"].map(_normalize_used_teams_value)
        sort_columns = ["entry_id"]
    else:
        df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
        df["team_picked"] = df["team_picked"].map(normalize_team_name)
        df["is_alive"] = df["is_alive"].map(_parse_bool)
        sort_columns = ["entry_id", "week"]

    return _strip_string_columns(df).sort_values(sort_columns).reset_index(drop=True)


def load_pool_history_df(path: str | Path) -> pd.DataFrame:
    """Load optional pool history summaries."""
    df = _read_csv(path)
    validate_or_raise("pool_history", df, validate_pool_history_df(df), str(path))
    integer_columns = ["season", "week", "entries_start", "entries_survived"]
    for column in integer_columns:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    df["most_popular_pick_share"] = pd.to_numeric(
        df["most_popular_pick_share"],
        errors="raise",
    ).astype(float)
    df["most_popular_pick"] = df["most_popular_pick"].map(normalize_team_name)
    return _strip_string_columns(df).sort_values(["season", "week"]).reset_index(
        drop=True,
    )


def load_double_pick_weeks_df(path: str | Path) -> pd.DataFrame:
    """Load optional weeks that require more than one survivor pick."""
    df = _read_csv(path)
    validate_or_raise(
        "double_pick_weeks",
        df,
        validate_double_pick_weeks_df(df),
        str(path),
    )
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["required_picks"] = pd.to_numeric(
        df["required_picks"],
        errors="raise",
    ).astype(int)
    return _strip_string_columns(df).sort_values(["week"]).reset_index(drop=True)


def load_sample_data(sample_dir: str | Path = SAMPLE_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the checked-in sample data bundle used by tests and the CLI."""
    base = Path(sample_dir)
    return {
        "schedule": load_schedule_df(base / "schedule_sample.csv"),
        "odds": load_odds_df(base / "odds_sample.csv"),
        "public_picks": load_public_picks_df(base / "public_picks_sample.csv"),
        "entries": load_entries_df(base / "entries_sample.csv"),
    }


def load_season_data(
    season: int = 2026,
    data_dir: str | Path = "data/raw",
    validate: bool = True,
    use_sample: bool = False,
) -> dict[str, pd.DataFrame | None]:
    """Load either real season CSVs or the checked-in sample bundle.

    Real data is expected in ``{data_dir}/{season}/`` with file names such as
    ``schedule.csv`` and ``odds.csv``.  For convenience, if ``data_dir`` itself
    contains those files, it is treated as the season directory.
    """
    if use_sample:
        sample_data = load_sample_data()
        return {
            "schedule_df": sample_data["schedule"],
            "odds_df": sample_data["odds"],
            "public_picks_df": sample_data["public_picks"],
            "entries_df": sample_data["entries"],
            "pool_history_df": None,
            "double_pick_weeks_df": None,
        }

    base = _resolve_season_dir(season=season, data_dir=data_dir)
    loaders = {
        "schedule_df": (load_schedule_df, base / SEASON_FILE_NAMES["schedule"], True),
        "odds_df": (load_odds_df, base / SEASON_FILE_NAMES["odds"], True),
        "public_picks_df": (
            load_public_picks_df,
            base / SEASON_FILE_NAMES["public_picks"],
            True,
        ),
        "entries_df": (load_entries_df, base / SEASON_FILE_NAMES["entries"], True),
        "pool_history_df": (
            load_pool_history_df,
            base / SEASON_FILE_NAMES["pool_history"],
            False,
        ),
        "double_pick_weeks_df": (
            load_double_pick_weeks_df,
            base / SEASON_FILE_NAMES["double_pick_weeks"],
            False,
        ),
    }

    loaded: dict[str, pd.DataFrame | None] = {}
    for key, (loader, path, required) in loaders.items():
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Required season data file not found: {path}. "
                    "Create it from templates with "
                    f"`python scripts/create_real_data_workspace.py --season {season}`.",
                )
            loaded[key] = None
            continue

        loaded[key] = loader(path) if validate else _read_csv(path)

    if validate:
        relationship_errors = validate_season_data_relationships(
            schedule_df=loaded["schedule_df"],
            odds_df=loaded["odds_df"],
            public_picks_df=loaded["public_picks_df"],
        )
        validate_or_raise(
            "season data relationships",
            pd.DataFrame(),
            relationship_errors,
            str(base),
        )

    return loaded


def normalize_schedule_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Normalize a raw schedule DataFrame to canonical team and game IDs.

    The input may use common provider-style column names such as ``home``,
    ``homeTeam``, ``awayTeam``, or ``kickoffTime``.  The returned DataFrame
    always includes canonical ``week``, ``game_id``, ``home_team``, and
    ``away_team`` columns.  A recognized kickoff column is converted to pandas
    datetimes and retained as ``kickoff``.
    """
    if df.empty:
        raise ValueError("schedule did not contain any rows.")

    _validate_normalization_season(season)
    source = df.copy()
    errors: list[str] = []

    column_map = _resolve_schedule_columns(source, errors)
    if errors:
        raise ValueError(_format_normalization_errors(errors))

    result = pd.DataFrame(index=source.index)
    result["week"] = _normalize_week_column(source[column_map["week"]], errors)
    result["home_team"] = _normalize_schedule_team_column(
        source[column_map["home_team"]],
        "home_team",
        errors,
    )
    result["away_team"] = _normalize_schedule_team_column(
        source[column_map["away_team"]],
        "away_team",
        errors,
    )

    if "kickoff" in column_map:
        result["kickoff"] = _normalize_kickoff_column(source[column_map["kickoff"]], errors)

    if errors:
        raise ValueError(_format_normalization_errors(errors))

    game_ids = []
    for row in result.to_dict("records"):
        game_ids.append(
            build_game_id(
                season=season,
                week=int(row["week"]),
                away_team=row["away_team"],
                home_team=row["home_team"],
            ),
        )
    result.insert(1, "game_id", game_ids)

    used_columns = set(column_map.values())
    standard_columns = {"week", "game_id", "home_team", "away_team", "kickoff"}
    extra_columns = [
        column
        for column in source.columns
        if column not in used_columns and column not in standard_columns
    ]
    normalized = pd.concat([result, source[extra_columns]], axis=1)
    validation_errors = validate_schedule_df(normalized)
    validate_or_raise("schedule", normalized, validation_errors)
    return normalized.sort_values(["week", "game_id"]).reset_index(drop=True)


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} did not contain any rows.")
    return df


def _find_public_pick_column(df: pd.DataFrame, path: str | Path) -> str:
    for column in PUBLIC_PICK_ALIASES:
        if column in df.columns:
            return column

    aliases = ", ".join(PUBLIC_PICK_ALIASES)
    raise ValueError(f"{path} needs one public pick percentage column: {aliases}")


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if is_object_dtype(df[column]) or is_string_dtype(df[column]):
            present = df[column].notna()
            df.loc[present, column] = df.loc[present, column].astype(str).str.strip()
    return df


def _parse_bool(value: object) -> bool:
    return parse_boolish(value)


def _normalize_used_teams_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    teams = [normalize_team_name(team) for team in text.split(";") if team.strip()]
    return ";".join(teams)


def _resolve_season_dir(season: int, data_dir: str | Path) -> Path:
    base = Path(data_dir)
    season_dir = base / str(season)
    if season_dir.exists():
        return season_dir
    if (base / SEASON_FILE_NAMES["schedule"]).exists():
        return base
    return season_dir


_WEEK_ALIASES = (("week", "wk", "week_number", "weekNumber", "game_week"),)
_HOME_TEAM_ALIAS_GROUPS = (
    (
        "home_team",
        "home",
        "homeTeam",
        "home_team_abbr",
        "homeTeamAbbr",
        "home_abbr",
        "home_club",
    ),
    ("favorite", "fav", "favourite"),
)
_AWAY_TEAM_ALIAS_GROUPS = (
    (
        "away_team",
        "away",
        "awayTeam",
        "away_team_abbr",
        "awayTeamAbbr",
        "away_abbr",
        "road",
        "visitor",
        "visitor_team",
    ),
    ("underdog", "dog"),
)
_KICKOFF_ALIAS_GROUPS = (
    (
        "kickoff",
        "kickoff_time",
        "kickoffTime",
        "game_time",
        "gameTime",
        "start_time",
        "startTime",
        "datetime",
        "game_datetime",
        "game_date",
    ),
)


def _resolve_schedule_columns(
    df: pd.DataFrame,
    errors: list[str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    specs = {
        "week": (_WEEK_ALIASES, True),
        "home_team": (_HOME_TEAM_ALIAS_GROUPS, True),
        "away_team": (_AWAY_TEAM_ALIAS_GROUPS, True),
        "kickoff": (_KICKOFF_ALIAS_GROUPS, False),
    }
    for canonical, (alias_groups, required) in specs.items():
        try:
            column = _find_column(df, alias_groups)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if column is None:
            if required:
                accepted = ", ".join(alias for group in alias_groups for alias in group)
                errors.append(
                    f"schedule missing a {canonical} column. "
                    f"Accepted aliases: {accepted}.",
                )
            continue
        resolved[canonical] = column

    if (
        "home_team" in resolved
        and "away_team" in resolved
        and resolved["home_team"] == resolved["away_team"]
    ):
        errors.append("schedule home_team and away_team resolved to the same column.")

    return resolved


def _find_column(
    df: pd.DataFrame,
    alias_groups: tuple[tuple[str, ...], ...],
) -> str | None:
    normalized_columns = {_normalize_column_name(column): column for column in df.columns}
    for aliases in alias_groups:
        matches = []
        for alias in aliases:
            column = normalized_columns.get(_normalize_column_name(alias))
            if column is not None:
                matches.append(column)
        unique_matches = sorted(set(matches))
        if len(unique_matches) > 1:
            raise ValueError(
                "schedule has multiple columns for the same field: "
                f"{', '.join(unique_matches)}.",
            )
        if unique_matches:
            return unique_matches[0]
    return None


def _normalize_column_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _normalize_week_column(values: pd.Series, errors: list[str]) -> pd.Series:
    present = ~values.map(_is_blank_value)
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_numeric = present & numeric.isna()
    invalid_integer = present & numeric.notna() & ((numeric % 1) != 0)
    invalid_range = present & numeric.notna() & (numeric < 1)

    if bool((~present).any()):
        errors.append(
            f"schedule column 'week' has blank required values on rows "
            f"{_format_rows(~present)}.",
        )
    if bool(invalid_numeric.any()):
        errors.append(
            f"schedule column 'week' must be numeric on rows "
            f"{_format_rows(invalid_numeric)}.",
        )
    if bool(invalid_integer.any()):
        errors.append(
            f"schedule column 'week' must contain whole numbers on rows "
            f"{_format_rows(invalid_integer)}.",
        )
    if bool(invalid_range.any()):
        errors.append(
            f"schedule column 'week' must be positive on rows "
            f"{_format_rows(invalid_range)}.",
        )

    return numeric.astype("Int64")


def _normalize_schedule_team_column(
    values: pd.Series,
    column: str,
    errors: list[str],
) -> pd.Series:
    normalized: list[str | None] = []
    invalid_rows: list[int] = []
    for index, value in values.items():
        try:
            normalized.append(normalize_team_name(value))
        except ValueError:
            normalized.append(None)
            invalid_rows.append(int(index) + 2)

    if invalid_rows:
        errors.append(
            f"schedule column '{column}' has invalid team aliases on rows "
            f"{_format_row_numbers(invalid_rows)}.",
        )
    return pd.Series(normalized, index=values.index, dtype="string")


def _normalize_kickoff_column(values: pd.Series, errors: list[str]) -> pd.Series:
    present = ~values.map(_is_blank_value)
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = present & parsed.isna()
    if bool(invalid.any()):
        errors.append(
            f"schedule column 'kickoff' has invalid datetimes on rows "
            f"{_format_rows(invalid)}.",
        )
    return parsed


def _validate_normalization_season(season: int) -> None:
    build_game_id(season=season, week=1, away_team="ARI", home_team="ATL")


def _is_blank_value(value: object) -> bool:
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


def _format_normalization_errors(errors: list[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors)
    return f"schedule normalization failed:\n{details}"


def schedule_df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-friendly copy of a normalized schedule DataFrame."""
    output = df.copy()
    if "kickoff" in output.columns and is_datetime64_any_dtype(output["kickoff"]):
        output["kickoff"] = output["kickoff"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    preferred_columns = (
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        "kickoff_at",
        "game_window",
        "kickoff",
    )
    ordered_columns = [column for column in preferred_columns if column in output.columns]
    ordered_columns.extend(column for column in output.columns if column not in ordered_columns)
    return output[ordered_columns]
