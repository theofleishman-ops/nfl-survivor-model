"""CSV loaders for the local Phase 1 survivor model.

The loaders read small sample CSVs today and intentionally accept the same
schema for future manually exported real CSVs.  They normalize column names and
types at the boundary so the scoring code can work with predictable
DataFrames.
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from survivor.config import SAMPLE_DATA_DIR
from survivor.schemas import (
    parse_boolish,
    validate_double_pick_weeks_df,
    validate_entries_df,
    validate_odds_df,
    validate_or_raise,
    validate_pool_history_df,
    validate_public_picks_df,
    validate_schedule_df,
)


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
    return _strip_string_columns(df).sort_values(["week", "game_id"]).reset_index(
        drop=True,
    )


def load_odds_df(path: str | Path) -> pd.DataFrame:
    """Load game-level American moneyline odds from CSV."""
    df = _read_csv(path)
    validate_or_raise("odds", df, validate_odds_df(df), str(path))
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
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
        sort_columns = ["entry_id"]
    else:
        df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
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

    return loaded


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


def _resolve_season_dir(season: int, data_dir: str | Path) -> Path:
    base = Path(data_dir)
    season_dir = base / str(season)
    if season_dir.exists():
        return season_dir
    if (base / SEASON_FILE_NAMES["schedule"]).exists():
        return base
    return season_dir
