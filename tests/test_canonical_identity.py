from datetime import datetime

import pandas as pd
import pytest

from survivor.game_ids import build_game_id, parse_game_id, validate_game_id
from survivor.loaders import normalize_schedule_df
from survivor.schedule_features import (
    INTERNATIONAL,
    MONDAY,
    OTHER,
    SATURDAY,
    SUNDAY_EARLY,
    SUNDAY_LATE,
    SUNDAY_NIGHT,
    THURSDAY,
    classify_game_window,
    detect_bye_weeks,
    teams_on_bye_by_week,
    teams_playing_by_week,
)
from survivor.schemas import validate_schedule_df
from survivor.teams import get_all_team_aliases, is_valid_team, normalize_team_name


def test_team_alias_normalization_handles_names_case_and_punctuation():
    assert normalize_team_name("Chiefs") == "KC"
    assert normalize_team_name("Kansas City") == "KC"
    assert normalize_team_name("K.C.") == "KC"
    assert normalize_team_name(" green bay packers ") == "GB"
    assert normalize_team_name("LA Chargers") == "LAC"
    assert normalize_team_name("L.A. Rams") == "LAR"


def test_invalid_team_alias_is_rejected():
    with pytest.raises(ValueError):
        normalize_team_name("London Monarchs")

    assert not is_valid_team("London Monarchs")
    assert is_valid_team("Washington Commanders")


def test_get_all_team_aliases_exposes_lookup_keys():
    aliases = get_all_team_aliases()

    assert aliases["KANSAS CITY CHIEFS"] == "KC"
    assert aliases["KC"] == "KC"


def test_game_id_generation_parsing_and_validation():
    game_id = build_game_id(2026, 1, "Baltimore Ravens", "K.C.")

    assert game_id == "2026_W01_BAL_AT_KC"
    assert parse_game_id(game_id) == {
        "season": 2026,
        "week": 1,
        "away_team": "BAL",
        "home_team": "KC",
    }
    assert validate_game_id(game_id)
    assert not validate_game_id("2026-W01-BAL-KC")
    assert not validate_game_id("2026_w01_bal_at_kc")


def test_normalize_schedule_df_accepts_flexible_columns_and_aliases():
    raw = pd.DataFrame(
        [
            {
                "Week": "1",
                "awayTeam": "Baltimore Ravens",
                "home": "K.C.",
                "kickoffTime": "2026-09-10T20:20:00",
            }
        ]
    )

    normalized = normalize_schedule_df(raw, season=2026)

    assert list(normalized[["week", "game_id", "away_team", "home_team"]].iloc[0]) == [
        1,
        "2026_W01_BAL_AT_KC",
        "BAL",
        "KC",
    ]
    assert pd.api.types.is_datetime64_any_dtype(normalized["kickoff"])


def test_normalize_schedule_df_accepts_favorite_underdog_fallback_columns():
    raw = pd.DataFrame([{"wk": 1, "favorite": "Chiefs", "underdog": "Ravens"}])

    normalized = normalize_schedule_df(raw, season=2026)

    assert normalized.loc[0, "home_team"] == "KC"
    assert normalized.loc[0, "away_team"] == "BAL"
    assert normalized.loc[0, "game_id"] == "2026_W01_BAL_AT_KC"


def test_schedule_validation_detects_duplicates_and_team_week_conflicts():
    df = pd.DataFrame(
        [
            {
                "week": 1,
                "game_id": "2026_W01_BAL_AT_KC",
                "away_team": "BAL",
                "home_team": "KC",
            },
            {
                "week": 1,
                "game_id": "2026_W01_BAL_AT_KC",
                "away_team": "BAL",
                "home_team": "KC",
            },
        ]
    )

    errors = validate_schedule_df(df)

    assert any("duplicate rows for game_id" in error for error in errors)
    assert any("appearing more than once in the same week" in error for error in errors)


def test_schedule_validation_detects_alias_conflicts_and_malformed_game_ids():
    df = pd.DataFrame(
        [
            {
                "week": 1,
                "game_id": "2026-W01-KC-BAL",
                "away_team": "Chiefs",
                "home_team": "KC",
            }
        ]
    )

    errors = validate_schedule_df(df)

    assert any("malformed canonical game IDs" in error for error in errors)
    assert any("must be different teams" in error for error in errors)


def test_schedule_normalization_reports_malformed_input():
    raw = pd.DataFrame([{"week": "", "home": "KC", "away": "Not A Team"}])

    with pytest.raises(ValueError) as exc_info:
        normalize_schedule_df(raw, season=2026)

    message = str(exc_info.value)
    assert "blank required values" in message
    assert "invalid team aliases" in message


def test_bye_week_helpers_return_playing_and_bye_teams():
    schedule = pd.DataFrame(
        [
            {
                "week": 1,
                "game_id": "2026_W01_BAL_AT_KC",
                "away_team": "BAL",
                "home_team": "KC",
            },
            {
                "week": 2,
                "game_id": "2026_W02_BUF_AT_KC",
                "away_team": "BUF",
                "home_team": "KC",
            },
        ]
    )

    playing = teams_playing_by_week(schedule)
    byes_by_week = teams_on_bye_by_week(schedule)
    byes_by_team = detect_bye_weeks(schedule)

    assert playing[1] == {"BAL", "KC"}
    assert "BUF" in byes_by_week[1]
    assert byes_by_team["BUF"] == [1]
    assert byes_by_team["BAL"] == [2]


@pytest.mark.parametrize(
    ("kickoff", "expected"),
    [
        (datetime(2026, 9, 10, 20, 20), THURSDAY),
        (datetime(2026, 9, 12, 16, 30), SATURDAY),
        (datetime(2026, 9, 13, 9, 30), INTERNATIONAL),
        (datetime(2026, 9, 13, 13, 0), SUNDAY_EARLY),
        (datetime(2026, 9, 13, 16, 25), SUNDAY_LATE),
        (datetime(2026, 9, 13, 20, 20), SUNDAY_NIGHT),
        (datetime(2026, 9, 14, 20, 15), MONDAY),
        (datetime(2026, 9, 11, 20, 0), OTHER),
    ],
)
def test_classify_game_window(kickoff, expected):
    assert classify_game_window(kickoff) == expected
