import pytest

from survivor.odds import (
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
