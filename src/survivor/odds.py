"""Odds conversion helpers for two-sided NFL moneyline markets.

The public CSVs are stored one row per game.  Ranking survivor picks is easier
one row per team, so :func:`add_no_vig_probabilities` converts wide game odds
into team-level rows with implied and no-vig win probabilities.
"""

from numbers import Real

import pandas as pd


REQUIRED_ODDS_COLUMNS = {
    "week",
    "game_id",
    "home_team",
    "away_team",
    "home_moneyline",
    "away_moneyline",
}


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


def add_no_vig_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Return team-level odds rows with implied and no-vig probabilities.

    Parameters
    ----------
    df:
        A game-level DataFrame with ``home_moneyline`` and ``away_moneyline``
        American odds.  The function intentionally does not fetch or scrape
        odds; it only transforms local CSV data.

    Returns
    -------
    pandas.DataFrame
        One row per team/game with ``team``, ``opponent``, ``moneyline``,
        ``implied_probability``, and ``no_vig_win_probability`` columns.
    """
    odds_df = df.copy()
    _require_columns(odds_df, REQUIRED_ODDS_COLUMNS)

    odds_df["week"] = pd.to_numeric(odds_df["week"], errors="raise").astype(int)
    odds_df["home_moneyline"] = pd.to_numeric(
        odds_df["home_moneyline"],
        errors="raise",
    )
    odds_df["away_moneyline"] = pd.to_numeric(
        odds_df["away_moneyline"],
        errors="raise",
    )

    rows: list[dict[str, object]] = []
    for row in odds_df.to_dict("records"):
        home_implied = moneyline_to_implied_probability(row["home_moneyline"])
        away_implied = moneyline_to_implied_probability(row["away_moneyline"])
        home_no_vig, away_no_vig = normalize_two_way_market(
            home_implied,
            away_implied,
        )

        rows.append(
            {
                "week": int(row["week"]),
                "game_id": row["game_id"],
                "team": row["home_team"],
                "opponent": row["away_team"],
                "is_home": True,
                "moneyline": float(row["home_moneyline"]),
                "implied_probability": home_implied,
                "no_vig_win_probability": home_no_vig,
            }
        )
        rows.append(
            {
                "week": int(row["week"]),
                "game_id": row["game_id"],
                "team": row["away_team"],
                "opponent": row["home_team"],
                "is_home": False,
                "moneyline": float(row["away_moneyline"]),
                "implied_probability": away_implied,
                "no_vig_win_probability": away_no_vig,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["week", "game_id", "is_home"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Odds data is missing required columns: {missing_text}")
