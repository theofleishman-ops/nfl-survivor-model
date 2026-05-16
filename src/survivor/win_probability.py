"""Current-game win probability estimation helpers."""

from __future__ import annotations

from math import exp
from typing import Any

import pandas as pd


DEFAULT_SPREAD_LOGISTIC_COEFFICIENT = 0.135
DEFAULT_TEAM_STRENGTH_COEFFICIENT = 4.0


def estimate_game_win_probability(
    row: dict[str, Any] | pd.Series,
    *,
    spread_coefficient: float = DEFAULT_SPREAD_LOGISTIC_COEFFICIENT,
    team_strength_coefficient: float = DEFAULT_TEAM_STRENGTH_COEFFICIENT,
) -> float:
    """Estimate a team's win probability from the best available signal.

    Priority:
    1. no-vig moneyline probability,
    2. spread-based logistic estimate,
    3. team-strength fallback.

    Spread convention follows American betting lines: negative spread means the
    team is favored, so ``-3`` maps above 50%.
    """
    no_vig = _float_or_none(_get(row, "no_vig_win_probability"))
    if no_vig is not None:
        return _clip_probability(no_vig)

    spread = _float_or_none(_get(row, "spread"))
    if spread is not None:
        return _clip_probability(1.0 / (1.0 + exp(spread_coefficient * spread)))

    team_strength = _float_or_none(_get(row, "team_strength_score"))
    opponent_strength = _float_or_none(_get(row, "opponent_team_strength_score"))
    if team_strength is not None and opponent_strength is not None:
        diff = team_strength - opponent_strength
        return _clip_probability(
            1.0 / (1.0 + exp(-team_strength_coefficient * diff)),
        )
    if team_strength is not None:
        return _clip_probability(team_strength)

    return 0.5


def _get(row: dict[str, Any] | pd.Series, key: str) -> Any:
    return row.get(key) if hasattr(row, "get") else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def _clip_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
