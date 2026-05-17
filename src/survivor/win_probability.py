"""Current-game win probability estimation helpers."""

from __future__ import annotations

from math import exp
from typing import Any

import pandas as pd


DEFAULT_SPREAD_LOGISTIC_COEFFICIENT = 0.135
DEFAULT_TEAM_STRENGTH_COEFFICIENT = 4.0
DEFAULT_HOME_FIELD_STRENGTH_EDGE = 0.0
DEFAULT_TEAM_STRENGTH_SCALE = 0.25


def estimate_game_win_probability(
    row: dict[str, Any] | pd.Series,
    *,
    spread_coefficient: float = DEFAULT_SPREAD_LOGISTIC_COEFFICIENT,
    team_strength_coefficient: float = DEFAULT_TEAM_STRENGTH_COEFFICIENT,
    home_field_strength_edge: float = DEFAULT_HOME_FIELD_STRENGTH_EDGE,
    team_strength_scale: float | None = None,
) -> float:
    """Estimate a team's win probability from the best available signal.

    Priority:
    1. no-vig moneyline probability,
    2. spread-based logistic estimate,
    3. team-strength fallback with a small home-field adjustment.

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
        scale = (
            float(team_strength_scale)
            if team_strength_scale is not None
            else 1.0 / float(team_strength_coefficient)
        )
        return estimate_team_strength_win_probability(
            team_rating=team_strength,
            opponent_rating=opponent_strength,
            is_home=_get(row, "is_home"),
            home_field_adjustment=home_field_strength_edge,
            scale=scale,
        )
    if team_strength is not None:
        return _clip_probability(team_strength)

    return 0.5


def estimate_team_strength_win_probability(
    *,
    team_rating: float,
    opponent_rating: float,
    is_home: Any = None,
    home_field_adjustment: float = DEFAULT_HOME_FIELD_STRENGTH_EDGE,
    scale: float = DEFAULT_TEAM_STRENGTH_SCALE,
) -> float:
    """Estimate win probability from two team ratings and optional home field."""
    numeric_scale = float(scale)
    if numeric_scale <= 0:
        raise ValueError("team strength scale must be positive.")
    adjustment = _home_field_edge(is_home, home_field_adjustment)
    z = (float(team_rating) - float(opponent_rating) + adjustment) / numeric_scale
    return _clip_probability(1.0 / (1.0 + exp(-z)))


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


def _home_field_edge(value: Any, edge: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return float(edge)
        if normalized in {"false", "0", "no", "n"}:
            return -float(edge)
        return 0.0
    return float(edge) if bool(value) else -float(edge)


def _clip_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
