"""Weekly survivor pick ranking model."""

from collections.abc import Mapping

import pandas as pd

from survivor.candidates import generate_weekly_candidates
from survivor.future_value import calculate_future_costs, calculate_week_scarcity
from survivor.leverage import add_leverage_columns
from survivor.odds import add_no_vig_probabilities


DEFAULT_WEIGHTS = {
    "win_probability": 1.50,
    "leverage": 0.25,
    "future_cost": 0.40,
}

RANKING_COLUMNS = [
    "week",
    "team",
    "opponent",
    "no_vig_win_probability",
    "public_pick_pct",
    "leverage_score",
    "week_scarcity_score",
    "future_cost",
    "scarcity_adjusted_future_cost",
    "raw_survival_score",
    "contest_ev_score",
    "final_score",
    "rank",
]


def rank_weekly_picks(
    week: int,
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    entries_df: pd.DataFrame | None = None,
    pool_size: int = 5000,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank survivor pick candidates for a single week.

    The score is intentionally decomposable:

    ``final_score = win_probability_component + leverage_component - future_cost_component``

    The defaults favor survival first, add a moderate ownership/leverage term,
    and subtract the full-season cost of spending teams with valuable future
    spots.
    """
    active_weights = DEFAULT_WEIGHTS | dict(weights or {})

    schedule = _as_dataframe(schedule_df)
    odds = _as_dataframe(odds_df)
    public_picks = _as_dataframe(public_picks_df)
    entries = _as_dataframe(entries_df) if entries_df is not None else None

    odds_with_probs = add_no_vig_probabilities(odds)
    candidates = generate_weekly_candidates(
        week=week,
        schedule_df=schedule,
        odds_df=odds_with_probs,
        public_picks_df=public_picks,
        entries_df=entries,
    )
    candidates = candidates[~candidates["team_already_used"]].copy()

    scarcity = calculate_week_scarcity(odds_with_probs)
    current_scarcity = scarcity[scarcity["week"] == int(week)][
        ["week", "week_scarcity_score"]
    ]
    candidates = candidates.merge(current_scarcity, on="week", how="left")

    future_costs = calculate_future_costs(
        current_week=week,
        candidates_df=candidates,
        odds_df=odds_with_probs,
    )
    ranked = candidates.merge(future_costs, on="team", how="left")
    ranked = add_leverage_columns(ranked, pool_size=pool_size)

    ranked["raw_survival_score"] = ranked["no_vig_win_probability"]
    ranked["contest_ev_score"] = (
        ranked["raw_survival_score"] + ranked["leverage_score"]
    )

    ranked["win_probability_component"] = (
        active_weights["win_probability"] * ranked["no_vig_win_probability"]
    )
    ranked["leverage_component"] = (
        active_weights["leverage"] * ranked["leverage_score"]
    )
    ranked["future_cost_component"] = (
        active_weights["future_cost"] * ranked["scarcity_adjusted_future_cost"]
    )
    ranked["final_score"] = (
        ranked["win_probability_component"]
        + ranked["leverage_component"]
        - ranked["future_cost_component"]
    )

    ranked = ranked.sort_values(
        ["final_score", "no_vig_win_probability"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked[_ordered_columns(ranked)]


def _as_dataframe(data: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    leading_columns = [
        *RANKING_COLUMNS,
        "best_future_week",
        "replacement_value_summary",
        "win_probability_component",
        "leverage_component",
        "future_cost_component",
        "expected_survival_value",
        "fade_value_if_loses",
        "expected_field_eliminated_if_team_loses",
        "expected_field_surviving_if_team_wins",
        "moneyline",
        "implied_probability",
        "team_already_used",
        "game_id",
        "is_home",
    ]
    return [column for column in leading_columns if column in df.columns]
