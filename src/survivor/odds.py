"""Odds conversion helpers."""

from numbers import Real


def moneyline_to_implied_probability(moneyline: Real | str) -> float:
    """Convert an American moneyline price to an implied probability."""
    price = float(moneyline)
    if price == 0:
        raise ValueError("Moneyline cannot be zero.")

    if price > 0:
        return 100 / (price + 100)

    return abs(price) / (abs(price) + 100)


def normalize_two_way_market(home_prob: Real, away_prob: Real) -> tuple[float, float]:
    """Remove the overround from a two-way market."""
    home = float(home_prob)
    away = float(away_prob)
    total = home + away

    if total <= 0:
        raise ValueError("Market probabilities must sum to a positive value.")

    return home / total, away / total
