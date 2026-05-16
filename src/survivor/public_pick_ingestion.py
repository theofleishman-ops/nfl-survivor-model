"""Provider-agnostic public survivor pick ingestion.

This module accepts manual CSV/JSON exports from public ownership sources and
normalizes them into a source-aware team/game schema.  It intentionally does not
scrape websites or call public pick APIs; future providers should adapt their
raw records into the same shape.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from survivor.game_ids import parse_game_id
from survivor.teams import normalize_team_name


NORMALIZED_PUBLIC_PICK_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team",
    "opponent",
    "game_id",
    "source",
    "public_pick_pct",
    "sample_size",
    "pulled_at",
    "notes",
)

AGGREGATED_PUBLIC_PICK_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team",
    "opponent",
    "game_id",
    "source",
    "public_pick_pct",
    "consensus_public_pick_pct",
    "source_count",
    "sources",
    "sample_size",
    "pulled_at",
    "notes",
)

SOURCE_TOTAL_TOLERANCE = 1e-6

_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "season": ("season", "year"),
    "week": ("week", "wk", "week_number", "weekNumber"),
    "team": (
        "team",
        "team_name",
        "teamName",
        "selection",
        "pick",
        "club",
        "franchise",
    ),
    "opponent": ("opponent", "opponent_team", "opp"),
    "game_id": ("game_id", "gameId", "event_id", "eventId", "id"),
    "source": ("source", "provider", "site", "pool", "origin"),
    "public_pick_pct": (
        "public_pick_pct",
        "pick_share",
        "pick_pct",
        "pick_percent",
        "pick_percentage",
        "public_pick_percent",
        "public_pick_percentage",
        "ownership",
        "ownership_pct",
        "ownership_percent",
        "selected_pct",
        "selected_percent",
        "percent",
        "percentage",
        "pct",
    ),
    "sample_size": (
        "sample_size",
        "samples",
        "sample",
        "entries",
        "entry_count",
        "pool_size",
        "pick_count",
        "n",
    ),
    "pulled_at": (
        "pulled_at",
        "timestamp",
        "exported_at",
        "last_update",
        "lastUpdated",
        "updated_at",
    ),
    "notes": ("notes", "note", "comment", "comments"),
}


def load_public_picks_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load raw public pick records from a CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Public picks CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} did not contain any rows.")
    return df.to_dict("records")


def load_public_picks_json(path: str | Path) -> list[dict[str, Any]]:
    """Load raw public pick records from a JSON file.

    Accepted shapes are a top-level list of records or an object containing a
    list under ``records``, ``data``, ``public_picks``, or ``picks``.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Public picks JSON file not found: {json_path}")

    with json_path.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)

    records = _extract_json_records(payload)
    if not records:
        raise ValueError(f"{json_path} did not contain any public pick records.")
    return records


def normalize_public_pick_records(
    records: list[dict[str, Any]] | pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize public pick records and join them to canonical schedule games."""
    raw_df = _records_to_frame(records)
    schedule = _prepare_schedule(schedule_df)

    rows: list[dict[str, Any]] = []
    for index, raw_record in raw_df.iterrows():
        row_number = int(index) + 2
        record = _canonicalize_record(raw_record.to_dict())
        game = _resolve_schedule_game(record, schedule, row_number)

        rows.append(
            {
                "season": game["season"],
                "week": game["week"],
                "team": game["team"],
                "opponent": game["opponent"],
                "game_id": game["game_id"],
                "source": _text_or_default(record.get("source"), "manual_import"),
                "public_pick_pct": _parse_public_pick_pct(
                    record.get("public_pick_pct"),
                    row_number,
                ),
                "sample_size": _non_negative_int_or_none(
                    record.get("sample_size"),
                    "sample_size",
                    row_number,
                ),
                "pulled_at": _text_or_none(record.get("pulled_at")),
                "notes": _text_or_none(record.get("notes")),
            },
        )

    normalized = pd.DataFrame(rows, columns=NORMALIZED_PUBLIC_PICK_COLUMNS)
    normalized = _coerce_normalized_types(normalized)
    validate_public_picks_against_schedule(normalized, schedule, raise_on_error=True)
    return _sort_public_pick_rows(normalized, include_source=True)


def aggregate_public_pick_sources(public_picks_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized source rows into consensus public ownership.

    Groups are averaged by source initially.  If every row in a team/game group
    has a positive ``sample_size``, the consensus uses sample-size weighting.
    The returned frame keeps ``public_pick_pct`` as a compatibility mirror of
    ``consensus_public_pick_pct`` for existing ranking and simulation code.
    """
    if public_picks_df.empty:
        return pd.DataFrame(columns=AGGREGATED_PUBLIC_PICK_COLUMNS)

    missing = {
        "season",
        "week",
        "team",
        "opponent",
        "game_id",
        "source",
        "public_pick_pct",
    } - set(public_picks_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"Cannot aggregate public picks missing columns: {missing_text}",
        )

    df = _coerce_normalized_types(_ensure_optional_columns(public_picks_df.copy()))
    rows: list[dict[str, Any]] = []
    group_columns = ["season", "week", "team", "opponent", "game_id"]

    for group_key, group in df.groupby(group_columns, dropna=False, sort=True):
        season, week, team, opponent, game_id = group_key
        pick_values = pd.to_numeric(group["public_pick_pct"], errors="raise")
        sample_sizes = pd.to_numeric(group["sample_size"], errors="coerce")
        use_weighted = bool(sample_sizes.notna().all() and (sample_sizes > 0).all())

        if use_weighted:
            consensus = float((pick_values * sample_sizes).sum() / sample_sizes.sum())
            notes = "sample_size_weighted"
        else:
            consensus = float(pick_values.mean())
            notes = "simple_average"

        pulled_values = [
            value for value in group["pulled_at"].map(_text_or_none) if value is not None
        ]
        source_values = sorted(
            {
                str(value).strip()
                for value in group["source"]
                if not _is_blank(value)
            },
        )
        total_sample_size = (
            int(sample_sizes.dropna().sum()) if bool(sample_sizes.notna().any()) else None
        )

        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": str(team),
                "opponent": str(opponent),
                "game_id": str(game_id),
                "source": "CONSENSUS",
                "public_pick_pct": consensus,
                "consensus_public_pick_pct": consensus,
                "source_count": len(source_values),
                "sources": ";".join(source_values),
                "sample_size": total_sample_size,
                "pulled_at": max(pulled_values) if pulled_values else _utc_now_text(),
                "notes": notes,
            },
        )

    aggregated = pd.DataFrame(rows, columns=AGGREGATED_PUBLIC_PICK_COLUMNS)
    return _sort_public_pick_rows(aggregated, include_source=False)


def validate_public_picks_against_schedule(
    public_picks_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    *,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate normalized or consensus public pick rows against the schedule."""
    errors: list[str] = []
    pick_column = _pick_value_column(public_picks_df)
    required_columns = {"season", "week", "team", "opponent", "game_id", "source"}
    missing = required_columns - set(public_picks_df.columns)
    if missing:
        errors.append(
            "public picks missing normalized columns: "
            + ", ".join(sorted(missing))
            + ".",
        )
    if pick_column is None:
        errors.append(
            "public picks need public_pick_pct or consensus_public_pick_pct.",
        )

    if not missing and pick_column is not None:
        _validate_public_pick_rows(public_picks_df, schedule_df, pick_column, errors)

    if errors and raise_on_error:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"public pick validation failed:\n{details}")
    return errors


def _extract_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_ensure_mapping(record) for record in payload]
    if isinstance(payload, dict):
        for key in ("records", "data", "public_picks", "picks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_ensure_mapping(record) for record in value]
    raise ValueError(
        "Public picks JSON must be a list or contain records/data/public_picks/picks.",
    )


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Public pick record must be a JSON object: {value!r}")
    return value


def _records_to_frame(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if df.empty:
        raise ValueError("public pick records did not contain any rows.")
    return df


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized_columns = {_normalize_column_name(column): column for column in record}
    canonical: dict[str, Any] = {}
    for field, aliases in _CANONICAL_ALIASES.items():
        canonical[field] = _first_present_value(record, normalized_columns, aliases)
    return canonical


def _first_present_value(
    record: dict[str, Any],
    normalized_columns: dict[str, str],
    aliases: tuple[str, ...],
) -> Any:
    for alias in aliases:
        column = normalized_columns.get(_normalize_column_name(alias))
        if column is not None and not _is_blank(record.get(column)):
            return record[column]
    return None


def _prepare_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    required = {"week", "game_id", "home_team", "away_team"}
    missing = required - set(schedule_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"schedule data is missing required columns: {missing_text}")

    schedule = schedule_df.copy()
    schedule["week"] = pd.to_numeric(schedule["week"], errors="raise").astype(int)
    schedule["home_team"] = schedule["home_team"].map(normalize_team_name)
    schedule["away_team"] = schedule["away_team"].map(normalize_team_name)
    if "season" in schedule.columns:
        schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce").astype(
            "Int64",
        )
    else:
        schedule["season"] = schedule["game_id"].map(_season_from_game_id).astype("Int64")
    return schedule


def _resolve_schedule_game(
    record: dict[str, Any],
    schedule: pd.DataFrame,
    row_number: int,
) -> dict[str, Any]:
    game_id = _text_or_none(record.get("game_id"))
    season = _int_or_none(record.get("season"))
    week = _int_or_none(record.get("week"))
    team = _normalize_team_or_none(record.get("team"))
    opponent = _normalize_team_or_none(record.get("opponent"))

    if team is None:
        raise ValueError(f"public pick row {row_number} is missing team.")

    if game_id is not None:
        candidates = schedule[schedule["game_id"].astype(str) == game_id]
        if candidates.empty:
            raise ValueError(
                f"public pick row {row_number} game_id {game_id!r} does not exist "
                "in schedule.",
            )
    elif week is not None:
        candidates = schedule[
            (schedule["week"] == week)
            & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
        ]
        if season is not None:
            candidates = candidates[candidates["season"] == season]
    else:
        raise ValueError(
            f"public pick row {row_number} cannot join to schedule; provide "
            "game_id or week with team.",
        )

    if candidates.empty:
        if week is not None:
            raise ValueError(
                f"public pick row {row_number} team {team!r} is not playing in "
                f"week {week}.",
            )
        raise ValueError(f"public pick row {row_number} does not match schedule.")
    if len(candidates) > 1:
        raise ValueError(
            f"public pick row {row_number} matches multiple scheduled games; "
            "include season or game_id.",
        )

    row = candidates.iloc[0]
    scheduled_teams = {row["home_team"], row["away_team"]}
    if team not in scheduled_teams:
        raise ValueError(
            f"public pick row {row_number} team {team!r} is not in scheduled game "
            f"{row['game_id']}.",
        )

    expected_opponent = (
        row["away_team"] if team == row["home_team"] else row["home_team"]
    )
    if opponent is not None and opponent != expected_opponent:
        raise ValueError(
            f"public pick row {row_number} opponent {opponent!r} does not match "
            f"scheduled game {row['game_id']}.",
        )

    schedule_season = _int_or_none(row.get("season"))
    if season is not None and schedule_season is not None and season != schedule_season:
        raise ValueError(
            f"public pick row {row_number} season {season} does not match "
            f"schedule season {schedule_season} for {row['game_id']}.",
        )
    if week is not None and int(row["week"]) != week:
        raise ValueError(
            f"public pick row {row_number} week {week} does not match schedule "
            f"week {int(row['week'])} for {row['game_id']}.",
        )

    return {
        "season": season if season is not None else schedule_season,
        "week": int(row["week"]),
        "team": team,
        "opponent": expected_opponent,
        "game_id": str(row["game_id"]).strip(),
    }


def _validate_public_pick_rows(
    public_picks_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    pick_column: str,
    errors: list[str],
) -> None:
    schedule = _prepare_schedule(schedule_df)
    schedule_by_id = {
        str(row["game_id"]).strip(): row
        for _, row in schedule.iterrows()
        if not _is_blank(row.get("game_id"))
    }

    parsed_rows: list[dict[str, Any]] = []
    duplicate_keys: list[tuple[int, str]] = []
    seen_keys: dict[tuple[int, int, str, str], int] = {}

    for index, row in public_picks_df.iterrows():
        row_number = int(index) + 2
        required_blank = [
            column
            for column in ("season", "week", "team", "opponent", "game_id", "source")
            if _is_blank(row.get(column))
        ]
        if required_blank:
            errors.append(
                f"row {row_number} is missing fields: {', '.join(required_blank)}.",
            )
            continue

        try:
            season = _int_or_none(row["season"])
            week = _int_or_none(row["week"])
            team = normalize_team_name(row["team"])
            opponent = normalize_team_name(row["opponent"])
            pct = _parse_public_pick_pct(row[pick_column], row_number)
        except ValueError as exc:
            errors.append(f"row {row_number} has invalid value: {exc}")
            continue

        source = str(row["source"]).strip()
        game_id = str(row["game_id"]).strip()
        schedule_row = schedule_by_id.get(game_id)
        if schedule_row is None:
            errors.append(
                f"row {row_number} game_id {game_id!r} does not exist in schedule.",
            )
            continue

        schedule_season = _int_or_none(schedule_row.get("season"))
        if schedule_season is not None and season != schedule_season:
            errors.append(
                f"row {row_number} season {season} does not match schedule "
                f"season {schedule_season}.",
            )
        if week != int(schedule_row["week"]):
            errors.append(
                f"row {row_number} week {week} does not match schedule week "
                f"{int(schedule_row['week'])}.",
            )
        if {team, opponent} != {schedule_row["home_team"], schedule_row["away_team"]}:
            errors.append(
                f"row {row_number} team/opponent do not match schedule game "
                f"{game_id}.",
            )

        key = (int(season), int(week), team, source)
        if key in seen_keys:
            duplicate_keys.append((row_number, f"W{week}:{team}:{source}"))
        else:
            seen_keys[key] = row_number

        parsed_rows.append(
            {
                "season": int(season),
                "week": int(week),
                "source": source,
                "public_pick_pct": pct,
            },
        )

    if duplicate_keys:
        rows = ", ".join(str(row) for row, _ in duplicate_keys[:5])
        examples = ", ".join(key for _, key in duplicate_keys[:3])
        errors.append(
            "public picks cannot contain duplicate source/team rows; duplicates on "
            f"rows {rows} ({examples}).",
        )

    _validate_source_weekly_totals(parsed_rows, errors)


def _validate_source_weekly_totals(
    parsed_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not parsed_rows:
        return

    totals = (
        pd.DataFrame(parsed_rows)
        .groupby(["season", "week", "source"], as_index=False)["public_pick_pct"]
        .sum()
    )
    invalid = totals[
        totals["public_pick_pct"] > (1.0 + SOURCE_TOTAL_TOLERANCE)
    ]
    if invalid.empty:
        return

    examples = [
        f"{row.source} season {int(row.season)} week {int(row.week)} "
        f"total {float(row.public_pick_pct):.6f}"
        for row in invalid.itertuples(index=False)
    ]
    errors.append(
        "source weekly public_pick_pct totals cannot exceed 1.0 materially; "
        + "; ".join(examples[:5])
        + (f"; and {len(examples) - 5} more" if len(examples) > 5 else "")
        + ".",
    )


def _coerce_normalized_types(df: pd.DataFrame) -> pd.DataFrame:
    output = _ensure_optional_columns(df.copy())
    output["season"] = pd.to_numeric(output["season"], errors="raise").astype("Int64")
    output["week"] = pd.to_numeric(output["week"], errors="raise").astype("Int64")
    output["team"] = output["team"].map(normalize_team_name)
    output["opponent"] = output["opponent"].map(normalize_team_name)
    output["public_pick_pct"] = pd.to_numeric(
        output["public_pick_pct"],
        errors="raise",
    ).astype(float)
    output["sample_size"] = pd.to_numeric(output["sample_size"], errors="coerce").astype(
        "Int64",
    )
    for column in ("source", "game_id", "pulled_at", "notes"):
        present = output[column].notna()
        output.loc[present, column] = output.loc[present, column].astype(str).str.strip()
    return output


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NORMALIZED_PUBLIC_PICK_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df


def _season_from_game_id(game_id: Any) -> int | None:
    try:
        return int(parse_game_id(game_id)["season"])
    except (TypeError, ValueError):
        return None


def _pick_value_column(df: pd.DataFrame) -> str | None:
    for column in ("public_pick_pct", "consensus_public_pick_pct"):
        if column in df.columns:
            return column
    return None


def _parse_public_pick_pct(value: Any, row_number: int) -> float:
    if _is_blank(value):
        raise ValueError(f"public pick row {row_number} is missing public_pick_pct.")

    if isinstance(value, str) and value.strip().endswith("%"):
        text = value.strip()[:-1].strip()
        numeric = pd.to_numeric(text, errors="raise") / 100
    else:
        numeric = pd.to_numeric(value, errors="raise")

    pct = float(numeric)
    if pct < 0 or pct > 1:
        raise ValueError(
            f"public pick row {row_number} public_pick_pct must be between 0 and 1.",
        )
    return pct


def _normalize_team_or_none(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return normalize_team_name(value)


def _int_or_none(value: Any) -> int | None:
    if _is_blank(value):
        return None
    numeric = pd.to_numeric(value, errors="raise")
    if float(numeric) % 1 != 0:
        raise ValueError(f"Expected an integer value, got {value!r}.")
    return int(numeric)


def _non_negative_int_or_none(value: Any, field: str, row_number: int) -> int | None:
    integer = _int_or_none(value)
    if integer is None:
        return None
    if integer < 0:
        raise ValueError(f"public pick row {row_number} {field} cannot be negative.")
    return integer


def _text_or_none(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return str(value).strip()


def _text_or_default(value: Any, default: str) -> str:
    return _text_or_none(value) or default


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and value.strip() == ""


def _normalize_column_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _sort_public_pick_rows(df: pd.DataFrame, *, include_source: bool) -> pd.DataFrame:
    sort_columns = ["season", "week", "game_id", "team"]
    if include_source and "source" in df.columns:
        sort_columns.append("source")
    return df.sort_values(sort_columns).reset_index(drop=True)


def _utc_now_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
