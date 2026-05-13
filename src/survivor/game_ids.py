"""Canonical survivor game ID helpers."""

from __future__ import annotations

import re

from survivor.teams import normalize_team_name


MIN_SEASON = 1920
MAX_SEASON = 2100
MIN_WEEK = 1
MAX_WEEK = 22

_GAME_ID_RE = re.compile(
    r"^(?P<season>\d{4})_W(?P<week>\d{2})_"
    r"(?P<away_team>[A-Z]{2,3})_AT_(?P<home_team>[A-Z]{2,3})$",
)


def build_game_id(
    season: int,
    week: int,
    away_team: object,
    home_team: object,
) -> str:
    """Build a deterministic canonical game ID.

    Example: ``build_game_id(2026, 1, "Baltimore", "K.C.")`` returns
    ``"2026_W01_BAL_AT_KC"``.
    """
    season_int = _validate_season(season)
    week_int = _validate_week(week)
    away = normalize_team_name(away_team)
    home = normalize_team_name(home_team)

    if away == home:
        raise ValueError("A game cannot have the same home and away team.")

    return f"{season_int}_W{week_int:02d}_{away}_AT_{home}"


def parse_game_id(game_id: object) -> dict[str, int | str]:
    """Parse a canonical game ID into season, week, away, and home fields."""
    text = str(game_id).strip()
    match = _GAME_ID_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            "Malformed game_id. Expected format like 2026_W01_BAL_AT_KC.",
        )

    season = _validate_season(int(match.group("season")))
    week = _validate_week(int(match.group("week")))
    away = normalize_team_name(match.group("away_team"))
    home = normalize_team_name(match.group("home_team"))
    expected = build_game_id(season, week, away, home)
    if text != expected:
        raise ValueError(f"Malformed game_id. Expected canonical value {expected}.")

    return {
        "season": season,
        "week": week,
        "away_team": away,
        "home_team": home,
    }


def validate_game_id(game_id: object) -> bool:
    """Return whether ``game_id`` is a valid canonical survivor game ID."""
    try:
        parse_game_id(game_id)
    except (TypeError, ValueError):
        return False
    return True


def _validate_season(season: int) -> int:
    try:
        season_int = int(season)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Season must be an integer year: {season!r}") from exc

    if season_int < MIN_SEASON or season_int > MAX_SEASON:
        raise ValueError(
            f"Season must be between {MIN_SEASON} and {MAX_SEASON}: {season_int}",
        )
    return season_int


def _validate_week(week: int) -> int:
    try:
        week_int = int(week)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Week must be an integer: {week!r}") from exc

    if week_int < MIN_WEEK or week_int > MAX_WEEK:
        raise ValueError(f"Week must be between {MIN_WEEK} and {MAX_WEEK}: {week_int}")
    return week_int
