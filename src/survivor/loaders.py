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


SCHEDULE_COLUMNS = {"week", "game_id", "home_team", "away_team"}
ODDS_COLUMNS = {
    "week",
    "game_id",
    "home_team",
    "away_team",
    "home_moneyline",
    "away_moneyline",
}
ENTRIES_COLUMNS = {"entry_id", "week", "team_picked", "is_alive"}
PUBLIC_PICK_ALIASES = (
    "public_pick_pct",
    "pick_share",
    "pick_pct",
    "ownership",
    "ownership_pct",
)


def load_schedule_df(path: str | Path) -> pd.DataFrame:
    """Load a schedule CSV with one row per game."""
    df = _read_csv(path)
    _require_columns(df, SCHEDULE_COLUMNS, path)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    return _strip_string_columns(df).sort_values(["week", "game_id"]).reset_index(
        drop=True,
    )


def load_odds_df(path: str | Path) -> pd.DataFrame:
    """Load game-level American moneyline odds from CSV."""
    df = _read_csv(path)
    _require_columns(df, ODDS_COLUMNS, path)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="raise")
    df["away_moneyline"] = pd.to_numeric(df["away_moneyline"], errors="raise")
    return _strip_string_columns(df).sort_values(["week", "game_id"]).reset_index(
        drop=True,
    )


def load_public_picks_df(path: str | Path) -> pd.DataFrame:
    """Load public pick ownership by week/team.

    The percentage column may be named ``public_pick_pct`` or one of the common
    aliases in :data:`PUBLIC_PICK_ALIASES`.  Values may be stored as decimals
    (``0.24``) or whole percentages (``24``); the loader returns decimals.
    """
    df = _read_csv(path)
    _require_columns(df, {"week", "team"}, path)
    public_col = _find_public_pick_column(df, path)

    df = df.rename(columns={public_col: "public_pick_pct"})
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["public_pick_pct"] = _normalize_percentage_column(df["public_pick_pct"])
    return _strip_string_columns(df[["week", "team", "public_pick_pct"]]).sort_values(
        ["week", "team"],
    ).reset_index(drop=True)


def load_entries_df(path: str | Path) -> pd.DataFrame:
    """Load historical survivor entry picks.

    Each row represents a past pick for an entry.  Phase 1 uses this only to
    flag teams already used by a specific entry when an entry id is provided.
    """
    df = _read_csv(path)
    _require_columns(df, ENTRIES_COLUMNS, path)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["is_alive"] = df["is_alive"].map(_parse_bool)
    return _strip_string_columns(df).sort_values(["entry_id", "week"]).reset_index(
        drop=True,
    )


def load_sample_data(sample_dir: str | Path = SAMPLE_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the checked-in sample data bundle used by tests and the CLI."""
    base = Path(sample_dir)
    return {
        "schedule": load_schedule_df(base / "schedule_sample.csv"),
        "odds": load_odds_df(base / "odds_sample.csv"),
        "public_picks": load_public_picks_df(base / "public_picks_sample.csv"),
        "entries": load_entries_df(base / "entries_sample.csv"),
    }


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} did not contain any rows.")
    return df


def _require_columns(df: pd.DataFrame, required_columns: set[str], path: str | Path) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")


def _find_public_pick_column(df: pd.DataFrame, path: str | Path) -> str:
    for column in PUBLIC_PICK_ALIASES:
        if column in df.columns:
            return column

    aliases = ", ".join(PUBLIC_PICK_ALIASES)
    raise ValueError(f"{path} needs one public pick percentage column: {aliases}")


def _normalize_percentage_column(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype(float)
    if values.max() > 1:
        values = values / 100
    return values


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if is_object_dtype(df[column]) or is_string_dtype(df[column]):
            df[column] = df[column].astype(str).str.strip()
    return df


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")
