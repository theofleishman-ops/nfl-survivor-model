from pathlib import Path

import pandas as pd

from survivor.game_ids import parse_game_id
from survivor.schedule_features import (
    INTERNATIONAL,
    MONDAY,
    OTHER,
    SATURDAY,
    SUNDAY_EARLY,
    SUNDAY_LATE,
    SUNDAY_NIGHT,
    THURSDAY,
    detect_bye_weeks,
    teams_on_bye_by_week,
)
from survivor.schemas import validate_schedule_df
from survivor.teams import CANONICAL_TEAMS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = PROJECT_ROOT / "data" / "raw" / "2026" / "schedule.csv"
REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "kickoff_at",
    "game_window",
]
VALID_GAME_WINDOWS = {
    THURSDAY,
    SUNDAY_EARLY,
    SUNDAY_LATE,
    SUNDAY_NIGHT,
    MONDAY,
    SATURDAY,
    INTERNATIONAL,
    OTHER,
    "TBD",
}


def test_real_2026_schedule_file_uses_required_columns_and_validates():
    schedule = _load_schedule()

    assert list(schedule.columns) == REQUIRED_COLUMNS
    assert len(schedule) == 272
    assert set(schedule["season"]) == {2026}
    assert validate_schedule_df(schedule) == []
    assert set(schedule["game_window"]).issubset(VALID_GAME_WINDOWS)
    assert schedule.loc[schedule["kickoff_at"].eq("TBD"), "game_window"].eq("TBD").all()


def test_real_2026_schedule_has_no_duplicate_game_ids_or_team_week_conflicts():
    schedule = _load_schedule()

    assert schedule["game_id"].is_unique
    team_weeks = pd.concat(
        [
            schedule[["week", "home_team"]].rename(columns={"home_team": "team"}),
            schedule[["week", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )

    assert not team_weeks.duplicated(subset=["week", "team"]).any()


def test_real_2026_schedule_represents_all_teams_once_per_game_slot():
    schedule = _load_schedule()
    team_games = pd.concat(
        [schedule["home_team"], schedule["away_team"]],
        ignore_index=True,
    )

    assert set(team_games) == set(CANONICAL_TEAMS)
    assert team_games.value_counts().to_dict() == {team: 17 for team in CANONICAL_TEAMS}


def test_real_2026_schedule_bye_week_helpers_cover_full_season():
    schedule = _load_schedule()

    byes_by_week = teams_on_bye_by_week(schedule)
    byes_by_team = detect_bye_weeks(schedule)

    assert set(byes_by_week) == set(range(1, 19))
    assert byes_by_week[1] == set()
    assert byes_by_week[18] == set()
    assert sum(len(teams) for teams in byes_by_week.values()) == 32
    assert set(byes_by_team) == set(CANONICAL_TEAMS)
    assert all(len(weeks) == 1 for weeks in byes_by_team.values())


def test_real_2026_schedule_game_ids_parse_to_row_identity():
    schedule = _load_schedule()

    for row in schedule.itertuples(index=False):
        parsed = parse_game_id(row.game_id)

        assert parsed["season"] == row.season
        assert parsed["week"] == row.week
        assert parsed["away_team"] == row.away_team
        assert parsed["home_team"] == row.home_team


def _load_schedule() -> pd.DataFrame:
    return pd.read_csv(SCHEDULE_PATH)
