"""Ownership leverage calculations for survivor picks."""

import pandas as pd


DEFAULT_POOL_SIZE = 5000


def calculate_leverage_metrics(
    win_probability: float,
    public_pick_pct: float,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> dict[str, float]:
    """Calculate Phase 1 leverage metrics for a single candidate.

    The leverage score rewards survival probability that is not crowded by the
    field and subtracts a chalk penalty.  This keeps two teams with similar win
    probability from ranking the same when one is dramatically more popular.
    """
    p = float(win_probability)
    q = float(public_pick_pct)
    if not 0 <= p <= 1:
        raise ValueError("win_probability must be between 0 and 1.")
    if not 0 <= q <= 1:
        raise ValueError("public_pick_pct must be between 0 and 1.")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")

    expected_survival_value = p
    fade_value_if_loses = q * (1 - p)
    leverage_score = (p * (1 - q)) + fade_value_if_loses - q

    return {
        "expected_survival_value": expected_survival_value,
        "fade_value_if_loses": fade_value_if_loses,
        "leverage_score": leverage_score,
        "expected_field_eliminated_if_team_loses": pool_size * q * (1 - p),
        "expected_field_surviving_if_team_wins": pool_size * (1 - q * (1 - p)),
    }


def add_leverage_columns(
    candidates_df: pd.DataFrame,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> pd.DataFrame:
    """Append leverage metrics to a candidate DataFrame."""
    required_columns = {"no_vig_win_probability", "public_pick_pct"}
    missing = required_columns - set(candidates_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Candidate data is missing required columns: {missing_text}")

    rows = [
        calculate_leverage_metrics(
            win_probability=row["no_vig_win_probability"],
            public_pick_pct=row["public_pick_pct"],
            pool_size=pool_size,
        )
        for row in candidates_df.to_dict("records")
    ]

    metrics_df = pd.DataFrame(rows, index=candidates_df.index)
    return pd.concat([candidates_df.copy(), metrics_df], axis=1)
