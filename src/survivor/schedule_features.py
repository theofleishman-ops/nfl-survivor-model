"""Schedule-derived NFL game timing and bye-week helpers."""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd

from survivor.teams import CANONICAL_TEAMS, normalize_team_name


THURSDAY = "THURSDAY"
SUNDAY_EARLY = "SUNDAY_EARLY"
SUNDAY_LATE = "SUNDAY_LATE"
SUNDAY_NIGHT = "SUNDAY_NIGHT"
MONDAY = "MONDAY"
SATURDAY = "SATURDAY"
INTERNATIONAL = "INTERNATIONAL"
OTHER = "OTHER"


def classify_game_window(dt: datetime | pd.Timestamp) -> str:
    """Classify a kickoff datetime into a coarse NFL game window."""
    kickoff = pd.Timestamp(dt).to_pydatetime()
    weekday = kickoff.weekday()
    kickoff_time = kickoff.time()

    if weekday == 3:
        return THURSDAY
    if weekday == 5:
        return SATURDAY
    if weekday == 0:
        return MONDAY
    if weekday != 6:
        return OTHER

    if kickoff_time < time(12, 0):
        return INTERNATIONAL
    if kickoff_time < time(16, 0):
        return SUNDAY_EARLY
    if kickoff_time < time(20, 0):
        return SUNDAY_LATE
    return SUNDAY_NIGHT


def teams_playing_by_week(schedule_df: pd.DataFrame) -> dict[int, set[str]]:
    """Return canonical teams scheduled to play in each week."""
    _require_columns(schedule_df, {"week", "home_team", "away_team"})
    schedule = schedule_df.copy()
    schedule["week"] = pd.to_numeric(schedule["week"], errors="raise").astype(int)
    schedule["home_team"] = schedule["home_team"].map(normalize_team_name)
    schedule["away_team"] = schedule["away_team"].map(normalize_team_name)

    playing: dict[int, set[str]] = {}
    for week, group in schedule.groupby("week", sort=True):
        teams = set(group["home_team"]) | set(group["away_team"])
        playing[int(week)] = teams
    return playing


def teams_on_bye_by_week(schedule_df: pd.DataFrame) -> dict[int, set[str]]:
    """Return canonical teams not playing in each represented schedule week."""
    all_teams = set(CANONICAL_TEAMS)
    return {
        week: all_teams - playing_teams
        for week, playing_teams in teams_playing_by_week(schedule_df).items()
    }


def detect_bye_weeks(schedule_df: pd.DataFrame) -> dict[str, list[int]]:
    """Return a team-to-weeks mapping for weeks where each team is not playing."""
    by_week = teams_on_bye_by_week(schedule_df)
    by_team = {team: [] for team in CANONICAL_TEAMS}
    for week, teams in by_week.items():
        for team in teams:
            by_team[team].append(week)
    return {team: weeks for team, weeks in by_team.items() if weeks}


def _require_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Schedule data is missing required columns: {missing_text}")
