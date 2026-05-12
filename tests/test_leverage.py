import pandas as pd
import pytest

from survivor.leverage import add_leverage_columns, calculate_leverage_metrics


def test_leverage_rewards_lower_owned_comparable_team():
    low_owned = calculate_leverage_metrics(0.70, 0.05)
    chalk = calculate_leverage_metrics(0.70, 0.35)

    assert low_owned["leverage_score"] > chalk["leverage_score"]


def test_add_leverage_columns_adds_field_metrics():
    candidates = pd.DataFrame(
        [
            {
                "team": "A",
                "no_vig_win_probability": 0.70,
                "public_pick_pct": 0.20,
            }
        ]
    )

    result = add_leverage_columns(candidates, pool_size=5000)

    assert result.loc[0, "expected_survival_value"] == 0.70
    assert result.loc[0, "fade_value_if_loses"] == pytest.approx(0.06)
    assert result.loc[0, "expected_field_eliminated_if_team_loses"] == pytest.approx(300)
