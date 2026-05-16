import pytest

import pandas as pd

from scripts import import_odds
from survivor.odds_ingestion import NORMALIZED_ODDS_COLUMNS, normalize_odds_records
from survivor.odds_providers.the_odds_api import API_KEY_ENV_VAR, TheOddsAPIProvider


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    provider = TheOddsAPIProvider(http_get=lambda *_args: [])

    with pytest.raises(RuntimeError, match=API_KEY_ENV_VAR):
        list(provider.fetch_odds())


def test_mocked_response_normalizes_current_nfl_odds():
    captured = {}

    def fake_http_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _odds_payload()

    provider = TheOddsAPIProvider(
        api_key="test-key",
        season=2026,
        week=1,
        markets=("h2h", "spreads", "totals"),
        bookmakers=("draftkings", "fanduel"),
        http_get=fake_http_get,
    )

    result = provider.normalize(_schedule())

    assert captured["url"].endswith("/sports/americanfootball_nfl/odds/")
    assert captured["params"]["regions"] == "us"
    assert captured["params"]["markets"] == "h2h,spreads,totals"
    assert captured["params"]["oddsFormat"] == "american"
    assert captured["params"]["bookmakers"] == "draftkings,fanduel"
    assert set(result.columns) == set(NORMALIZED_ODDS_COLUMNS)
    assert set(result["game_id"]) == {"2026_W01_DAL_AT_PHI"}
    assert set(result["sportsbook"]) == {"draftkings"}
    assert len(result) == 6

    h2h = result[result["market_type"] == "h2h"]
    assert set(h2h["team"]) == {"DAL", "PHI"}
    assert h2h["moneyline"].notna().all()
    assert h2h["no_vig_win_probability"].sum() == pytest.approx(1.0)

    spreads = result[result["market_type"] == "spreads"]
    assert dict(zip(spreads["team"], spreads["spread"])) == {"DAL": 3.5, "PHI": -3.5}

    totals = result[result["market_type"] == "totals"]
    assert set(totals["total"]) == {47.5}
    assert totals["total_price"].notna().all()


def test_team_aliases_normalize_and_join_to_schedule():
    provider = TheOddsAPIProvider(
        api_key="test-key",
        season=2026,
        week=1,
        markets="h2h",
        http_get=lambda *_args: [
            {
                "id": "provider-game-2",
                "home_team": "LA Chargers",
                "away_team": "Kansas City Chiefs",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "last_update": "2026-09-01T12:00:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "LA Chargers", "price": 105},
                                    {"name": "Chiefs", "price": -125},
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    )

    result = provider.normalize(_schedule())

    assert set(result["team"]) == {"KC", "LAC"}
    assert set(result["home_team"]) == {"LAC"}
    assert set(result["away_team"]) == {"KC"}
    assert set(result["game_id"]) == {"2026_W01_KC_AT_LAC"}


def test_unsupported_markets_are_ignored_when_supported_markets_remain():
    captured = {}

    def fake_http_get(url, params, timeout):
        captured["params"] = params
        return _odds_payload()

    provider = TheOddsAPIProvider(
        api_key="test-key",
        season=2026,
        week=1,
        markets=("h2h", "player_pass_tds"),
        http_get=fake_http_get,
    )

    result = provider.normalize(_schedule())

    assert provider.unsupported_markets == ("player_pass_tds",)
    assert captured["params"]["markets"] == "h2h"
    assert set(result["market_type"]) == {"h2h"}


def test_only_unsupported_markets_fail_before_http_call():
    called = False

    def fake_http_get(*_args):
        nonlocal called
        called = True
        return []

    provider = TheOddsAPIProvider(
        api_key="test-key",
        season=2026,
        week=1,
        markets=("player_pass_tds",),
        http_get=fake_http_get,
    )

    with pytest.raises(ValueError, match="No supported The Odds API markets"):
        list(provider.fetch_odds())
    assert called is False


def test_import_odds_cli_the_odds_api_dry_run_with_injected_provider(tmp_path, capsys):
    schedule_path = tmp_path / "schedule.csv"
    output_path = tmp_path / "odds.csv"
    _schedule().to_csv(schedule_path, index=False)
    captured = {}

    class FakeProvider:
        unsupported_markets = ()

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def normalize(self, schedule_df):
            return normalize_odds_records(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "home_team": "PHI",
                        "away_team": "DAL",
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                        "sportsbook": "fakebook",
                    },
                ],
                schedule_df,
            )

    result = import_odds.main(
        [
            "--provider",
            "the-odds-api",
            "--season",
            "2026",
            "--week",
            "1",
            "--schedule",
            str(schedule_path),
            "--output",
            str(output_path),
            "--regions",
            "us",
            "--markets",
            "h2h,spreads",
            "--sportsbook",
            "fanduel,draftkings",
            "--dry-run",
        ],
        provider_factory=FakeProvider,
    )

    stdout = capsys.readouterr().out
    assert result == 0
    assert output_path.exists() is False
    assert "Dry run complete" in stdout
    assert "fakebook" in stdout
    assert captured["season"] == 2026
    assert captured["week"] == 1
    assert captured["regions"] == "us"
    assert captured["markets"] == "h2h,spreads"
    assert captured["bookmakers"] == "fanduel,draftkings"


def _odds_payload():
    return [
        {
            "id": "provider-game-1",
            "home_team": "Philadelphia Eagles",
            "away_team": "Dallas Cowboys",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2026-09-01T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Philadelphia Eagles", "price": -150},
                                {"name": "Dallas Cowboys", "price": 130},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {
                                    "name": "Philadelphia Eagles",
                                    "price": -110,
                                    "point": -3.5,
                                },
                                {
                                    "name": "Dallas Cowboys",
                                    "price": -110,
                                    "point": 3.5,
                                },
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -105, "point": 47.5},
                                {"name": "Under", "price": -115, "point": 47.5},
                            ],
                        },
                    ],
                },
            ],
        }
    ]


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
