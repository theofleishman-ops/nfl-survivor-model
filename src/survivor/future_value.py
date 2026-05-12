"""Numeric future scarcity and opportunity-cost calculations."""

from dataclasses import dataclass

import pandas as pd

from survivor.odds import add_no_vig_probabilities


SCARCITY_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)


@dataclass(frozen=True)
class FutureCostConfig:
    """Tunable constants for the Phase 1 opportunity-cost heuristic."""

    strong_probability_threshold: float = 0.60
    scarcity_multiplier: float = 1.0


def calculate_week_scarcity(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate numeric scarcity scores for every week in the odds data.

    Scarce weeks have fewer teams above common survivor thresholds.  The score
    sums inverse counts at each threshold, weighted slightly more for the higher
    thresholds because elite options matter most in survivor pools.
    """
    team_odds = _ensure_team_level_odds(odds_df)
    _require_columns(team_odds, {"week", "team", "no_vig_win_probability"}, "odds")

    rows: list[dict[str, float | int]] = []
    for week, week_df in team_odds.groupby("week", sort=True):
        probabilities = week_df["no_vig_win_probability"].astype(float)
        count_60 = int((probabilities >= 0.60).sum())
        count_65 = int((probabilities >= 0.65).sum())
        count_70 = int((probabilities >= 0.70).sum())
        count_75 = int((probabilities >= 0.75).sum())

        scarcity_score = (
            1.00 / (1 + count_60)
            + 1.25 / (1 + count_65)
            + 1.50 / (1 + count_70)
            + 1.75 / (1 + count_75)
        )

        rows.append(
            {
                "week": int(week),
                "count_ge_60": count_60,
                "count_ge_65": count_65,
                "count_ge_70": count_70,
                "count_ge_75": count_75,
                "week_scarcity_score": float(scarcity_score),
            }
        )

    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def calculate_future_costs(
    current_week: int,
    candidates_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    config: FutureCostConfig | None = None,
) -> pd.DataFrame:
    """Estimate full-season opportunity cost for each current-week candidate.

    For each candidate team, this looks at every remaining week where that team
    is a strong future option.  Replacement cost is the positive gap between
    that team's future win probability and the next-best option in that week if
    the team has already been spent.  Scarce future weeks receive extra weight.
    """
    cfg = config or FutureCostConfig()
    team_odds = _ensure_team_level_odds(odds_df)
    _require_columns(team_odds, {"week", "team", "no_vig_win_probability"}, "odds")
    _require_columns(candidates_df, {"team"}, "candidates")

    team_odds["week"] = pd.to_numeric(team_odds["week"], errors="raise").astype(int)
    team_odds["no_vig_win_probability"] = pd.to_numeric(
        team_odds["no_vig_win_probability"],
        errors="raise",
    ).astype(float)

    remaining = team_odds[team_odds["week"] > int(current_week)].copy()
    scarcity_by_week = calculate_week_scarcity(team_odds).set_index("week")

    rows: list[dict[str, object]] = []
    for team in candidates_df["team"].drop_duplicates():
        team_future = remaining[
            (remaining["team"] == team)
            & (
                remaining["no_vig_win_probability"]
                >= cfg.strong_probability_threshold
            )
        ]

        future_cost = 0.0
        adjusted_cost = 0.0
        best_future_week: int | None = None
        best_adjusted_piece = 0.0
        summary_parts: list[str] = []

        for future_row in team_future.to_dict("records"):
            future_week = int(future_row["week"])
            team_probability = float(future_row["no_vig_win_probability"])
            week_options = remaining[
                (remaining["week"] == future_week) & (remaining["team"] != team)
            ]
            next_best_probability = (
                float(week_options["no_vig_win_probability"].max())
                if not week_options.empty
                else 0.0
            )

            replacement_cost = max(team_probability - next_best_probability, 0.0)
            scarcity_score = float(
                scarcity_by_week.loc[future_week, "week_scarcity_score"],
            )
            adjusted_piece = replacement_cost * (
                1 + cfg.scarcity_multiplier * scarcity_score
            )

            future_cost += replacement_cost
            adjusted_cost += adjusted_piece
            if adjusted_piece > best_adjusted_piece:
                best_adjusted_piece = adjusted_piece
                best_future_week = future_week

            summary_parts.append(
                "W{week}: p={prob:.3f}, replacement={replacement:.3f}, "
                "cost={cost:.3f}, scarcity={scarcity:.3f}".format(
                    week=future_week,
                    prob=team_probability,
                    replacement=next_best_probability,
                    cost=replacement_cost,
                    scarcity=scarcity_score,
                )
            )

        rows.append(
            {
                "team": team,
                "best_future_week": best_future_week,
                "future_cost": float(future_cost),
                "scarcity_adjusted_future_cost": float(adjusted_cost),
                "replacement_value_summary": "; ".join(summary_parts)
                if summary_parts
                else "No future strong spots.",
            }
        )

    return pd.DataFrame(rows)


def _ensure_team_level_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    if {"team", "opponent", "no_vig_win_probability"}.issubset(odds_df.columns):
        return odds_df.copy()
    return add_no_vig_probabilities(odds_df)


def _require_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{label} data is missing required columns: {missing_text}")
