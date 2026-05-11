import pytest

from survivor.odds import (
    add_no_vig_probabilities,
    moneyline_to_implied_probability,
    normalize_two_way_market,
)


def test_moneyline_to_implied_probability_for_favorite():
    assert moneyline_to_implied_probability(-150) == pytest.approx(0.6)


def test_moneyline_to_implied_probability_for_underdog():
    assert moneyline_to_implied_probability(200) == pytest.approx(1 / 3)


def test_moneyline_to_implied_probability_rejects_zero():
    with pytest.raises(ValueError):
        moneyline_to_implied_probability(0)


def test_normalize_two_way_market_removes_overround():
    home, away = normalize_two_way_market(0.6, 0.5)

    assert home == pytest.approx(6 / 11)
    assert away == pytest.approx(5 / 11)
    assert home + away == pytest.approx(1.0)


def test_add_no_vig_probabilities_returns_team_level_rows():
    odds_df = pytest.importorskip("pandas").DataFrame(
        [
            {
                "week": 1,
                "game_id": "GAME-1",
                "home_team": "Atlas",
                "away_team": "Beacon",
                "home_moneyline": -150,
                "away_moneyline": 130,
            }
        ]
    )

    result = add_no_vig_probabilities(odds_df)

    assert list(result["team"]) == ["Atlas", "Beacon"]
    assert result["implied_probability"].sum() > 1
    assert result["no_vig_win_probability"].sum() == pytest.approx(1.0)
