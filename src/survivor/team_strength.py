"""Team-strength priors derived from season futures markets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from survivor.odds import moneyline_to_implied_probability
from survivor.teams import CANONICAL_TEAMS, normalize_team_name


FUTURES_PROBABILITY_MARKETS = ("super_bowl", "conference", "division", "playoff")
TEAM_STRENGTH_COLUMNS = (
    "team",
    "super_bowl_implied_probability",
    "conference_implied_probability",
    "division_implied_probability",
    "playoff_implied_probability",
    "win_total",
    "win_total_rating",
    "team_strength_score",
)


def build_team_strength_priors(futures_df: pd.DataFrame) -> pd.DataFrame:
    """Build simple team-strength priors from futures odds.

    The score is a placeholder blend of normalized futures probabilities and a
    normalized win-total rating.  It is intentionally conservative: missing
    markets are skipped instead of guessed.
    """
    if futures_df.empty:
        return _empty_strength_frame()

    futures = futures_df.copy()
    _require_columns(futures, {"team", "market_type"})
    futures["team"] = futures["team"].map(normalize_team_name)
    futures["market_type"] = futures["market_type"].astype(str).str.strip().str.lower()

    rows: list[dict[str, Any]] = []
    for team, team_df in futures.groupby("team", sort=True):
        row: dict[str, Any] = {"team": team}
        for market in FUTURES_PROBABILITY_MARKETS:
            row[f"{market}_implied_probability"] = _best_market_probability(
                team_df,
                market,
            )

        row["win_total"] = _best_win_total(team_df)
        row["win_total_rating"] = _win_total_to_rating(row["win_total"])
        rows.append(row)

    strength = pd.DataFrame(rows)
    for column in TEAM_STRENGTH_COLUMNS:
        if column not in strength.columns:
            strength[column] = pd.NA

    strength = _add_team_strength_score(strength)
    return strength[list(TEAM_STRENGTH_COLUMNS)].sort_values(
        "team_strength_score",
        ascending=False,
    ).reset_index(drop=True)


def futures_to_team_strength(futures_df: pd.DataFrame) -> pd.DataFrame:
    """Alias for callers that prefer a shorter domain phrase."""
    return build_team_strength_priors(futures_df)


def _best_market_probability(team_df: pd.DataFrame, market_type: str) -> float | None:
    market_rows = team_df[team_df["market_type"] == market_type].copy()
    if market_rows.empty:
        return None

    if "implied_probability" in market_rows.columns:
        probabilities = pd.to_numeric(
            market_rows["implied_probability"],
            errors="coerce",
        ).dropna()
        if not probabilities.empty:
            return float(probabilities.mean())

    if "moneyline" not in market_rows.columns:
        return None
    moneylines = pd.to_numeric(market_rows["moneyline"], errors="coerce").dropna()
    if moneylines.empty:
        return None
    probabilities = moneylines.map(moneyline_to_implied_probability)
    return float(probabilities.mean())


def _best_win_total(team_df: pd.DataFrame) -> float | None:
    market_rows = team_df[team_df["market_type"] == "win_total"].copy()
    if market_rows.empty:
        return None

    for column in ("total", "win_total"):
        if column in market_rows.columns:
            values = pd.to_numeric(market_rows[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return None


def _win_total_to_rating(win_total: float | None) -> float | None:
    if win_total is None or pd.isna(win_total):
        return None
    # A coarse 4-to-14 win range keeps the placeholder bounded for early priors.
    return float(max(0.0, min(1.0, (float(win_total) - 4.0) / 10.0)))


def _add_team_strength_score(strength: pd.DataFrame) -> pd.DataFrame:
    output = strength.copy()
    probability_columns = [
        "super_bowl_implied_probability",
        "conference_implied_probability",
        "division_implied_probability",
        "playoff_implied_probability",
    ]
    for column in probability_columns:
        output[f"{column}_rating"] = _normalize_positive_column(output[column])

    weights = {
        "super_bowl_implied_probability_rating": 0.30,
        "conference_implied_probability_rating": 0.25,
        "division_implied_probability_rating": 0.20,
        "playoff_implied_probability_rating": 0.10,
        "win_total_rating": 0.15,
    }

    scores: list[float] = []
    for _, row in output.iterrows():
        weighted_total = 0.0
        active_weight = 0.0
        for column, weight in weights.items():
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            weighted_total += float(value) * weight
            active_weight += weight
        scores.append(weighted_total / active_weight if active_weight else 0.5)

    output["team_strength_score"] = scores
    return output


def _normalize_positive_column(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    max_value = numeric.max(skipna=True)
    if pd.isna(max_value) or float(max_value) <= 0:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="Float64")
    return (numeric / float(max_value)).astype("Float64")


def _empty_strength_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TEAM_STRENGTH_COLUMNS)


def _require_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"futures data is missing required columns: {missing_text}")


def default_team_strength_priors() -> pd.DataFrame:
    """Return neutral priors for all teams."""
    return pd.DataFrame(
        {
            "team": list(CANONICAL_TEAMS),
            "team_strength_score": [0.5] * len(CANONICAL_TEAMS),
        },
    )
