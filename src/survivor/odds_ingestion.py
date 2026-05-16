"""Provider-agnostic odds ingestion and normalization.

The ingestion layer accepts manual CSV/JSON exports and provider-shaped records,
then writes one normalized team/market row.  It deliberately avoids sportsbook
scraping and secrets; provider-specific code should adapt into this schema.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from survivor.game_ids import parse_game_id
from survivor.odds import moneyline_to_implied_probability
from survivor.teams import normalize_team_name


NORMALIZED_ODDS_COLUMNS: tuple[str, ...] = (
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
)

GAME_MARKET_TYPES = {"h2h", "moneyline", "spreads", "spread", "totals", "total"}
FUTURES_MARKET_TYPES = {
    "super_bowl",
    "conference",
    "division",
    "playoff",
    "win_total",
}
MARKET_TYPE_ALIASES = {
    "moneyline": "h2h",
    "ml": "h2h",
    "headtohead": "h2h",
    "head_to_head": "h2h",
    "h2h": "h2h",
    "spread": "spreads",
    "spreads": "spreads",
    "pointspread": "spreads",
    "point_spread": "spreads",
    "total": "totals",
    "totals": "totals",
    "overunder": "totals",
    "over_under": "totals",
    "superbowl": "super_bowl",
    "super_bowl": "super_bowl",
    "conference": "conference",
    "division": "division",
    "playoff": "playoff",
    "makeplayoffs": "playoff",
    "make_playoffs": "playoff",
    "wintotal": "win_total",
    "win_total": "win_total",
    "seasonwins": "win_total",
    "season_wins": "win_total",
}

_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "season": ("season", "year"),
    "week": ("week", "wk", "week_number", "weekNumber"),
    "game_id": ("game_id", "gameId", "event_id", "eventId", "id"),
    "sportsbook": ("sportsbook", "book", "bookmaker", "bookmaker_key", "site"),
    "market_type": ("market_type", "market", "market_key", "marketKey", "key"),
    "team": (
        "team",
        "team_name",
        "selection",
        "outcome",
        "outcome_name",
        "name",
        "participant",
    ),
    "opponent": ("opponent", "opponent_team", "opp"),
    "home_team": ("home_team", "home", "homeTeam", "home_team_abbr", "homeTeamAbbr"),
    "away_team": (
        "away_team",
        "away",
        "awayTeam",
        "visitor",
        "visitor_team",
        "awayTeamAbbr",
    ),
    "moneyline": ("moneyline", "american_odds", "price", "odds"),
    "home_moneyline": ("home_moneyline", "home_ml", "home_price", "home_odds"),
    "away_moneyline": ("away_moneyline", "away_ml", "away_price", "away_odds"),
    "spread": ("spread", "point", "points", "line", "handicap"),
    "home_spread": ("home_spread", "home_line", "home_point", "home_points"),
    "away_spread": ("away_spread", "away_line", "away_point", "away_points"),
    "spread_price": (
        "spread_price",
        "spread_odds",
        "spread_american_odds",
        "spreadPrice",
    ),
    "home_spread_price": (
        "home_spread_price",
        "home_spread_odds",
        "home_spread_price_american",
    ),
    "away_spread_price": (
        "away_spread_price",
        "away_spread_odds",
        "away_spread_price_american",
    ),
    "total": ("total", "game_total", "over_under", "overUnder", "ou"),
    "total_price": ("total_price", "total_odds", "over_price", "under_price"),
    "implied_probability": ("implied_probability", "implied_prob", "implied"),
    "no_vig_win_probability": (
        "no_vig_win_probability",
        "novig_probability",
        "no_vig_probability",
        "fair_probability",
    ),
    "pulled_at": (
        "pulled_at",
        "market_timestamp",
        "timestamp",
        "last_update",
        "lastUpdate",
        "commence_time",
    ),
    "source": ("source", "provider", "file_source"),
}


def load_odds_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load raw odds records from a CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Odds CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} did not contain any rows.")
    return df.to_dict("records")


def load_odds_json(path: str | Path) -> list[dict[str, Any]]:
    """Load raw odds records from a JSON file.

    Accepted shapes are a top-level list of records or an object with one of
    ``records``, ``data``, ``odds``, or ``events`` containing a list.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Odds JSON file not found: {json_path}")

    with json_path.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)

    records = _extract_json_records(payload)
    if not records:
        raise ValueError(f"{json_path} did not contain any odds records.")
    return records


def normalize_odds_records(
    records: list[dict[str, Any]] | pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize raw odds into the provider-agnostic team-level schema."""
    raw_df = _records_to_frame(records)
    schedule = _prepare_schedule(schedule_df)

    rows: list[dict[str, Any]] = []
    for index, raw_record in raw_df.iterrows():
        record = _canonicalize_record(raw_record.to_dict())
        market_type = _infer_market_type(record)
        row_number = int(index) + 2

        if market_type in FUTURES_MARKET_TYPES:
            rows.append(_normalize_futures_record(record, market_type, row_number))
            continue

        if _should_expand_game_sides(record):
            rows.extend(_expand_game_record(record, market_type, schedule, row_number))
        else:
            rows.append(_normalize_team_game_record(record, market_type, schedule, row_number))

    normalized = pd.DataFrame(rows, columns=NORMALIZED_ODDS_COLUMNS)
    normalized = _coerce_normalized_types(normalized)
    normalized = _add_implied_probabilities(normalized)
    normalized = _add_no_vig_probabilities(normalized)
    validate_odds_against_schedule(normalized, schedule, raise_on_error=True)
    return _sort_normalized_odds(normalized)


def validate_odds_against_schedule(
    odds_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    *,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate normalized odds rows against the schedule.

    Game markets must join to an existing schedule game.  Futures markets are
    validated for canonical team identity but do not require a game join.
    """
    errors: list[str] = []
    missing = set(NORMALIZED_ODDS_COLUMNS) - set(odds_df.columns)
    if missing:
        errors.append(
            "odds missing normalized columns: " + ", ".join(sorted(missing)) + ".",
        )

    if not missing:
        _validate_normalized_rows(odds_df, _prepare_schedule(schedule_df), errors)

    if errors and raise_on_error:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"odds validation failed:\n{details}")
    return errors


def aggregate_book_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple sportsbook rows into a consensus odds DataFrame."""
    if odds_df.empty:
        return odds_df.copy()

    df = odds_df.copy()
    required = set(NORMALIZED_ODDS_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Cannot aggregate odds missing columns: {missing_text}")

    group_columns = [
        "season",
        "week",
        "game_id",
        "market_type",
        "team",
        "opponent",
        "home_team",
        "away_team",
    ]
    numeric_columns = [
        "moneyline",
        "spread",
        "spread_price",
        "total",
        "total_price",
        "implied_probability",
        "no_vig_win_probability",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    aggregated = (
        df.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            {
                **{column: "mean" for column in numeric_columns},
                "pulled_at": "max",
            },
        )
        .reset_index(drop=True)
    )
    aggregated["sportsbook"] = "CONSENSUS"
    aggregated["source"] = "aggregate_book_odds"
    return _sort_normalized_odds(aggregated[list(NORMALIZED_ODDS_COLUMNS)])


def _extract_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_ensure_mapping(record) for record in payload]
    if isinstance(payload, dict):
        for key in ("records", "data", "odds", "events"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_ensure_mapping(record) for record in value]
    raise ValueError("Odds JSON must be a list or contain records/data/odds/events.")


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Odds record must be a JSON object: {value!r}")
    return value


def _records_to_frame(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if df.empty:
        raise ValueError("odds records did not contain any rows.")
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


def _infer_market_type(record: dict[str, Any]) -> str:
    raw_market = record.get("market_type")
    if not _is_blank(raw_market):
        normalized = _normalize_market_type(raw_market)
        if normalized is not None:
            return normalized
        return str(raw_market).strip().lower()

    if any(not _is_blank(record.get(field)) for field in ("moneyline", "home_moneyline", "away_moneyline")):
        return "h2h"
    if any(not _is_blank(record.get(field)) for field in ("spread", "home_spread", "away_spread")):
        return "spreads"
    if not _is_blank(record.get("total")):
        return "totals"
    return "h2h"


def _normalize_market_type(value: Any) -> str | None:
    key = _normalize_column_name(value)
    return MARKET_TYPE_ALIASES.get(key)


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


def _season_from_game_id(game_id: Any) -> int | None:
    try:
        return int(parse_game_id(game_id)["season"])
    except (TypeError, ValueError):
        return None


def _should_expand_game_sides(record: dict[str, Any]) -> bool:
    if not _is_blank(record.get("team")):
        return False
    return not _is_blank(record.get("home_team")) and not _is_blank(record.get("away_team"))


def _expand_game_record(
    record: dict[str, Any],
    market_type: str,
    schedule: pd.DataFrame,
    row_number: int,
) -> list[dict[str, Any]]:
    game = _resolve_schedule_game(record, schedule, row_number)
    home_record = dict(record)
    away_record = dict(record)
    home_record["team"] = game["home_team"]
    home_record["opponent"] = game["away_team"]
    home_record["moneyline"] = record.get("home_moneyline") or record.get("moneyline")
    home_record["spread"] = record.get("home_spread") or record.get("spread")
    home_record["spread_price"] = record.get("home_spread_price") or record.get(
        "spread_price",
    )

    away_record["team"] = game["away_team"]
    away_record["opponent"] = game["home_team"]
    away_record["moneyline"] = record.get("away_moneyline") or record.get("moneyline")
    away_record["spread"] = record.get("away_spread")
    if _is_blank(away_record["spread"]) and not _is_blank(home_record["spread"]):
        away_record["spread"] = -float(home_record["spread"])
    away_record["spread_price"] = record.get("away_spread_price") or record.get(
        "spread_price",
    )

    return [
        _normalized_game_row(home_record, market_type, game),
        _normalized_game_row(away_record, market_type, game),
    ]


def _normalize_team_game_record(
    record: dict[str, Any],
    market_type: str,
    schedule: pd.DataFrame,
    row_number: int,
) -> dict[str, Any]:
    game = _resolve_schedule_game(record, schedule, row_number)
    return _normalized_game_row(record, market_type, game)


def _resolve_schedule_game(
    record: dict[str, Any],
    schedule: pd.DataFrame,
    row_number: int,
) -> dict[str, Any]:
    game_id = _text_or_none(record.get("game_id"))
    week = _int_or_none(record.get("week"))
    home_team = _normalize_team_or_none(record.get("home_team"))
    away_team = _normalize_team_or_none(record.get("away_team"))
    team = _normalize_team_or_none(record.get("team"))
    opponent = _normalize_team_or_none(record.get("opponent"))

    candidates: pd.DataFrame
    if game_id is not None:
        candidates = schedule[schedule["game_id"].astype(str) == game_id]
        if candidates.empty:
            raise ValueError(
                f"odds row {row_number} game_id {game_id!r} does not exist in schedule.",
            )
    elif week is not None and home_team is not None and away_team is not None:
        candidates = schedule[
            (schedule["week"] == week)
            & (schedule["home_team"] == home_team)
            & (schedule["away_team"] == away_team)
        ]
    elif week is not None and team is not None and opponent is not None:
        candidates = schedule[
            (schedule["week"] == week)
            & (
                (
                    (schedule["home_team"] == team)
                    & (schedule["away_team"] == opponent)
                )
                | (
                    (schedule["home_team"] == opponent)
                    & (schedule["away_team"] == team)
                )
            )
        ]
    else:
        raise ValueError(
            f"odds row {row_number} cannot join to schedule; provide game_id or "
            "week with home/away teams or team/opponent.",
        )

    if candidates.empty:
        raise ValueError(
            f"odds row {row_number} does not match a scheduled game.",
        )
    if len(candidates) > 1:
        raise ValueError(
            f"odds row {row_number} matches multiple scheduled games; include game_id.",
        )

    row = candidates.iloc[0]
    if team is not None and team not in {row["home_team"], row["away_team"]}:
        raise ValueError(
            f"odds row {row_number} team {team!r} is not in scheduled game "
            f"{row['game_id']}.",
        )
    if opponent is not None and opponent not in {row["home_team"], row["away_team"]}:
        raise ValueError(
            f"odds row {row_number} opponent {opponent!r} is not in scheduled game "
            f"{row['game_id']}.",
        )

    raw_season = _int_or_none(record.get("season"))
    schedule_season = _int_or_none(row.get("season"))
    if raw_season is not None and schedule_season is not None and raw_season != schedule_season:
        raise ValueError(
            f"odds row {row_number} season {raw_season} does not match schedule "
            f"season {schedule_season} for {row['game_id']}.",
        )

    return {
        "season": raw_season if raw_season is not None else schedule_season,
        "week": int(row["week"]),
        "game_id": str(row["game_id"]).strip(),
        "home_team": str(row["home_team"]),
        "away_team": str(row["away_team"]),
    }


def _normalized_game_row(
    record: dict[str, Any],
    market_type: str,
    game: dict[str, Any],
) -> dict[str, Any]:
    team = _normalize_team_or_none(record.get("team"))
    if team is None:
        raise ValueError(f"odds row for game {game['game_id']} is missing team.")
    opponent = _normalize_team_or_none(record.get("opponent"))
    if opponent is None:
        opponent = game["away_team"] if team == game["home_team"] else game["home_team"]

    return {
        "season": game["season"],
        "week": game["week"],
        "game_id": game["game_id"],
        "sportsbook": _text_or_default(record.get("sportsbook"), "UNKNOWN"),
        "market_type": market_type,
        "team": team,
        "opponent": opponent,
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "moneyline": _float_or_none(record.get("moneyline")),
        "spread": _float_or_none(record.get("spread")),
        "spread_price": _float_or_none(record.get("spread_price")),
        "total": _float_or_none(record.get("total")),
        "total_price": _float_or_none(record.get("total_price")),
        "implied_probability": _float_or_none(record.get("implied_probability")),
        "no_vig_win_probability": _float_or_none(record.get("no_vig_win_probability")),
        "pulled_at": _text_or_default(record.get("pulled_at"), _utc_now_text()),
        "source": _text_or_default(record.get("source"), "manual_import"),
    }


def _normalize_futures_record(
    record: dict[str, Any],
    market_type: str,
    row_number: int,
) -> dict[str, Any]:
    team = _normalize_team_or_none(record.get("team"))
    if team is None:
        raise ValueError(f"futures odds row {row_number} is missing team.")
    season = _int_or_none(record.get("season"))
    if season is None:
        raise ValueError(f"futures odds row {row_number} is missing season.")

    return {
        "season": season,
        "week": _int_or_none(record.get("week")),
        "game_id": None,
        "sportsbook": _text_or_default(record.get("sportsbook"), "UNKNOWN"),
        "market_type": market_type,
        "team": team,
        "opponent": None,
        "home_team": None,
        "away_team": None,
        "moneyline": _float_or_none(record.get("moneyline")),
        "spread": None,
        "spread_price": None,
        "total": _float_or_none(record.get("total")),
        "total_price": _float_or_none(record.get("total_price")),
        "implied_probability": _float_or_none(record.get("implied_probability")),
        "no_vig_win_probability": _float_or_none(record.get("no_vig_win_probability")),
        "pulled_at": _text_or_default(record.get("pulled_at"), _utc_now_text()),
        "source": _text_or_default(record.get("source"), "manual_import"),
    }


def _coerce_normalized_types(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    nullable_int_columns = ["season", "week"]
    numeric_columns = [
        "moneyline",
        "spread",
        "spread_price",
        "total",
        "total_price",
        "implied_probability",
        "no_vig_win_probability",
    ]
    for column in nullable_int_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    missing_implied = output["implied_probability"].isna() & output["moneyline"].notna()
    output.loc[missing_implied, "implied_probability"] = output.loc[
        missing_implied,
        "moneyline",
    ].map(moneyline_to_implied_probability)
    return output


def _add_no_vig_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    game_h2h = output["market_type"].eq("h2h") & output["game_id"].notna()
    group_columns = ["season", "week", "game_id", "sportsbook", "market_type"]
    for _, group in output.loc[game_h2h].groupby(group_columns, dropna=False):
        valid = group["implied_probability"].notna()
        if int(valid.sum()) != 2:
            continue
        total = float(group.loc[valid, "implied_probability"].sum())
        if total <= 0:
            continue
        output.loc[group.loc[valid].index, "no_vig_win_probability"] = (
            group.loc[valid, "implied_probability"].astype(float) / total
        )
    return output


def _validate_normalized_rows(
    odds_df: pd.DataFrame,
    schedule: pd.DataFrame,
    errors: list[str],
) -> None:
    schedule_by_id = {
        str(row["game_id"]).strip(): row
        for _, row in schedule.iterrows()
        if not _is_blank(row.get("game_id"))
    }

    for index, row in odds_df.iterrows():
        row_number = int(index) + 2
        market_type = str(row.get("market_type", "")).strip()
        canonical_market = _normalize_market_type(market_type) or market_type
        if canonical_market not in GAME_MARKET_TYPES | FUTURES_MARKET_TYPES:
            errors.append(
                f"row {row_number} has unsupported market_type {market_type!r}.",
            )

        _validate_team_value(row.get("team"), "team", row_number, errors)
        if canonical_market in FUTURES_MARKET_TYPES:
            if _is_blank(row.get("season")):
                errors.append(f"row {row_number} futures market is missing season.")
            continue

        required_game_fields = ("week", "game_id", "team", "opponent", "home_team", "away_team")
        missing_game_fields = [
            field for field in required_game_fields if _is_blank(row.get(field))
        ]
        if missing_game_fields:
            errors.append(
                f"row {row_number} game market is missing fields: "
                f"{', '.join(missing_game_fields)}.",
            )
            continue

        game_id = str(row["game_id"]).strip()
        schedule_row = schedule_by_id.get(game_id)
        if schedule_row is None:
            errors.append(
                f"row {row_number} game_id {game_id!r} does not exist in schedule.",
            )
            continue

        try:
            team = normalize_team_name(row["team"])
            opponent = normalize_team_name(row["opponent"])
            home = normalize_team_name(row["home_team"])
            away = normalize_team_name(row["away_team"])
        except ValueError as exc:
            errors.append(f"row {row_number} has invalid team alias: {exc}")
            continue

        if {team, opponent} != {home, away}:
            errors.append(
                f"row {row_number} team/opponent do not match home/away teams.",
            )
        if home != schedule_row["home_team"] or away != schedule_row["away_team"]:
            errors.append(
                f"row {row_number} home/away teams do not match schedule game {game_id}.",
            )

    for column in ("implied_probability", "no_vig_win_probability"):
        if column in odds_df.columns:
            values = pd.to_numeric(odds_df[column], errors="coerce")
            present = odds_df[column].notna()
            invalid = present & ~values.between(0, 1)
            if bool(invalid.any()):
                errors.append(
                    f"{column} must be between 0 and 1 on rows "
                    f"{_format_rows(invalid)}.",
                )


def _validate_team_value(
    value: Any,
    column: str,
    row_number: int,
    errors: list[str],
) -> None:
    if _is_blank(value):
        errors.append(f"row {row_number} is missing {column}.")
        return
    try:
        normalize_team_name(value)
    except ValueError as exc:
        errors.append(f"row {row_number} has invalid {column}: {exc}")


def _sort_normalized_odds(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["season", "week", "game_id", "sportsbook", "market_type", "team"]
    return df.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def _normalize_team_or_none(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return normalize_team_name(value)


def _text_or_none(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return str(value).strip()


def _text_or_default(value: Any, default: str) -> str:
    return _text_or_none(value) or default


def _int_or_none(value: Any) -> int | None:
    if _is_blank(value):
        return None
    numeric = pd.to_numeric(value, errors="raise")
    if float(numeric) % 1 != 0:
        raise ValueError(f"Expected an integer value, got {value!r}.")
    return int(numeric)


def _float_or_none(value: Any) -> float | None:
    if _is_blank(value):
        return None
    return float(pd.to_numeric(value, errors="raise"))


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


def _utc_now_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _format_rows(mask: pd.Series) -> str:
    rows = [str(int(index) + 2) for index, invalid in mask.items() if bool(invalid)]
    return ", ".join(rows[:5]) + (f", and {len(rows) - 5} more" if len(rows) > 5 else "")
