import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from survivor.odds_ingestion import (
    NORMALIZED_ODDS_COLUMNS,
    aggregate_book_odds,
    load_odds_csv,
    load_odds_json,
    normalize_odds_records,
)
from survivor.team_strength import build_team_strength_priors
from survivor.win_probability import estimate_game_win_probability


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_csv_odds_normalization(tmp_path):
    raw_path = tmp_path / "odds_raw.csv"
    raw_path.write_text(
        "\n".join(
            [
                "season,week,home_team,away_team,home_moneyline,away_moneyline,sportsbook,pulled_at,source",
                "2026,1,PHI,DAL,-150,130,BookA,2026-09-01T12:00:00Z,test",
            ],
        ),
        encoding="utf-8",
    )

    result = normalize_odds_records(load_odds_csv(raw_path), _schedule())

    assert list(result["team"]) == ["DAL", "PHI"]
    assert set(result.columns) == set(NORMALIZED_ODDS_COLUMNS)
    assert result["game_id"].nunique() == 1
    assert result["implied_probability"].notna().all()


def test_json_odds_normalization(tmp_path):
    raw_path = tmp_path / "odds_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "season": 2026,
                        "week": 1,
                        "team": "Kansas City Chiefs",
                        "opponent": "Los Angeles Chargers",
                        "moneyline": -125,
                        "sportsbook": "BookA",
                        "pulled_at": "2026-09-01T12:00:00Z",
                        "source": "test",
                    },
                    {
                        "season": 2026,
                        "week": 1,
                        "team": "LA Chargers",
                        "opponent": "Chiefs",
                        "moneyline": 105,
                        "sportsbook": "BookA",
                        "pulled_at": "2026-09-01T12:00:00Z",
                        "source": "test",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    result = normalize_odds_records(load_odds_json(raw_path), _schedule())

    assert set(result["team"]) == {"KC", "LAC"}
    assert set(result["game_id"]) == {"2026_W01_KC_AT_LAC"}


def test_team_alias_normalization_and_schedule_join():
    records = [
        {
            "season": 2026,
            "week": 1,
            "home_team": "Philadelphia Eagles",
            "away_team": "Dallas Cowboys",
            "home_moneyline": -150,
            "away_moneyline": 130,
            "sportsbook": "BookA",
            "pulled_at": "2026-09-01T12:00:00Z",
            "source": "test",
        }
    ]

    result = normalize_odds_records(records, _schedule())

    assert set(result["team"]) == {"PHI", "DAL"}
    assert set(result["home_team"]) == {"PHI"}
    assert set(result["away_team"]) == {"DAL"}
    assert set(result["game_id"]) == {"2026_W01_DAL_AT_PHI"}


def test_missing_game_join_fails_clearly():
    records = [
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_W01_BAL_AT_KC",
            "team": "BAL",
            "opponent": "KC",
            "moneyline": -110,
        }
    ]

    with pytest.raises(ValueError, match="does not exist in schedule"):
        normalize_odds_records(records, _schedule())


def test_no_vig_probability_creation():
    result = normalize_odds_records(
        [
            {
                "season": 2026,
                "week": 1,
                "home_team": "PHI",
                "away_team": "DAL",
                "home_moneyline": -150,
                "away_moneyline": 130,
                "sportsbook": "BookA",
            }
        ],
        _schedule(),
    )

    assert result["no_vig_win_probability"].sum() == pytest.approx(1.0)


def test_multiple_sportsbook_aggregation():
    result = normalize_odds_records(
        [
            {
                "season": 2026,
                "week": 1,
                "home_team": "PHI",
                "away_team": "DAL",
                "home_moneyline": -150,
                "away_moneyline": 130,
                "sportsbook": "BookA",
            },
            {
                "season": 2026,
                "week": 1,
                "home_team": "PHI",
                "away_team": "DAL",
                "home_moneyline": -140,
                "away_moneyline": 120,
                "sportsbook": "BookB",
            },
        ],
        _schedule(),
    )

    aggregated = aggregate_book_odds(result)

    assert set(aggregated["sportsbook"]) == {"CONSENSUS"}
    assert len(aggregated) == 2
    assert aggregated["no_vig_win_probability"].sum() == pytest.approx(1.0)


def test_spread_fallback_win_probability():
    favorite = estimate_game_win_probability({"spread": -3.5})
    underdog = estimate_game_win_probability({"spread": 3.5})

    assert favorite > 0.5
    assert underdog < 0.5
    assert favorite == pytest.approx(1 - underdog)


def test_futures_team_strength_placeholder_behavior():
    futures = pd.DataFrame(
        [
            {"team": "PHI", "market_type": "super_bowl", "moneyline": 900},
            {"team": "DAL", "market_type": "super_bowl", "moneyline": 2500},
            {"team": "PHI", "market_type": "conference", "moneyline": 450},
            {"team": "DAL", "market_type": "conference", "moneyline": 1200},
            {"team": "PHI", "market_type": "win_total", "total": 11.5},
            {"team": "DAL", "market_type": "win_total", "total": 8.5},
        ],
    )

    strength = build_team_strength_priors(futures)

    assert set(strength["team"]) == {"PHI", "DAL"}
    assert strength.iloc[0]["team"] == "PHI"
    assert strength["team_strength_score"].between(0, 1).all()


def test_import_odds_cli_smoke(tmp_path):
    schedule_path = tmp_path / "schedule.csv"
    raw_path = tmp_path / "odds_raw.csv"
    output_path = tmp_path / "odds.csv"
    _schedule().to_csv(schedule_path, index=False)
    raw_path.write_text(
        "\n".join(
            [
                "week,home_team,away_team,home_moneyline,away_moneyline,sportsbook",
                "1,PHI,DAL,-150,130,BookA",
            ],
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_odds.py",
            "--input",
            str(raw_path),
            "--format",
            "csv",
            "--season",
            "2026",
            "--schedule",
            str(schedule_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    imported = pd.read_csv(output_path)
    assert "Wrote 2 normalized odds rows" in result.stdout
    assert set(NORMALIZED_ODDS_COLUMNS).issubset(imported.columns)
    assert imported["no_vig_win_probability"].sum() == pytest.approx(1.0)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_W01_DAL_AT_PHI",
                "away_team": "DAL",
                "home_team": "PHI",
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_W01_KC_AT_LAC",
                "away_team": "KC",
                "home_team": "LAC",
            },
        ],
    )
