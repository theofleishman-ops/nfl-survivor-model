"""Heuristic multi-entry survivor portfolio optimizer."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd


AGGRESSION_CONFIGS = {
    "conservative": {
        "min_win_probability": 0.50,
        "low_probability_cap_pct": 0.00,
        "contrarian_cap_pct": 0.10,
        "exposure_penalty": 0.20,
        "weights": {
            "final_score": 0.35,
            "win_probability": 0.45,
            "leverage": 0.05,
            "uniqueness": 0.05,
            "future_flexibility": 0.10,
        },
    },
    "balanced": {
        "min_win_probability": 0.50,
        "low_probability_cap_pct": 0.00,
        "contrarian_cap_pct": 0.20,
        "exposure_penalty": 0.15,
        "weights": {
            "final_score": 0.40,
            "win_probability": 0.30,
            "leverage": 0.15,
            "uniqueness": 0.05,
            "future_flexibility": 0.10,
        },
    },
    "aggressive": {
        "min_win_probability": 0.40,
        "low_probability_cap_pct": 0.10,
        "contrarian_cap_pct": 0.25,
        "exposure_penalty": 0.10,
        "weights": {
            "final_score": 0.30,
            "win_probability": 0.15,
            "leverage": 0.30,
            "uniqueness": 0.20,
            "future_flexibility": 0.05,
        },
    },
}

REQUIRED_RANKING_COLUMNS = {
    "week",
    "team",
    "no_vig_win_probability",
    "public_pick_pct",
    "leverage_score",
    "future_cost",
    "scarcity_adjusted_future_cost",
    "final_score",
}


@dataclass(frozen=True)
class _Entry:
    entry_id: str
    active: bool
    used_teams: frozenset[str]


def optimize_portfolio_for_week(
    week: int,
    rankings_df: pd.DataFrame,
    entries_df: pd.DataFrame,
    max_team_exposure_pct: float = 0.40,
    min_teams: int = 3,
    personal_entry_count: int | None = None,
    aggression: str = "balanced",
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Allocate survivor picks across a personal entry portfolio.

    The optimizer is a transparent greedy heuristic. It scores teams from the
    existing weekly rankings, seeds enough distinct viable teams to encourage
    diversification, then fills remaining active entries while respecting
    per-team caps and entry-level used-team constraints.
    """
    if aggression not in AGGRESSION_CONFIGS:
        choices = ", ".join(sorted(AGGRESSION_CONFIGS))
        raise ValueError(f"aggression must be one of: {choices}")
    if not 0 < max_team_exposure_pct <= 1:
        raise ValueError("max_team_exposure_pct must be between 0 and 1.")
    if min_teams < 1:
        raise ValueError("min_teams must be at least 1.")

    rng = random.Random(random_seed)
    config = AGGRESSION_CONFIGS[aggression]
    rankings = _prepare_rankings(rankings_df, week=week, config=config)
    entries = _normalize_entries(
        entries_df=entries_df,
        week=week,
        personal_entry_count=personal_entry_count,
    )
    active_entries = [entry for entry in entries if entry.active]

    if not active_entries:
        empty_allocation = _empty_allocation_df()
        empty_exposure = _empty_exposure_df()
        return empty_allocation, empty_exposure, _calculate_metrics(
            allocation_df=empty_allocation,
            exposure_df=empty_exposure,
            active_entry_count=0,
            aggression=aggression,
            max_team_exposure_pct=max_team_exposure_pct,
            unallocated_count=0,
        )

    candidates = rankings[
        rankings["no_vig_win_probability"] >= float(config["min_win_probability"])
    ].copy()
    viable_counts = _eligible_entry_counts(candidates, active_entries)
    candidates = candidates[candidates["team"].map(viable_counts).fillna(0) > 0].copy()

    if candidates.empty:
        empty_allocation = _empty_allocation_df()
        empty_exposure = _empty_exposure_df()
        return empty_allocation, empty_exposure, _calculate_metrics(
            allocation_df=empty_allocation,
            exposure_df=empty_exposure,
            active_entry_count=len(active_entries),
            aggression=aggression,
            max_team_exposure_pct=max_team_exposure_pct,
            unallocated_count=len(active_entries),
        )

    base_cap = max(1, math.floor(len(active_entries) * max_team_exposure_pct))
    team_caps = _team_caps(candidates, active_count=len(active_entries), base_cap=base_cap, config=config)

    order_index = _entry_order_index(active_entries, rng if random_seed is not None else None)
    assigned: dict[str, dict[str, Any]] = {}
    team_counts = {team: 0 for team in candidates["team"].astype(str)}

    min_team_target = min(int(min_teams), len(active_entries), len(candidates))
    for team in candidates["team"].head(min_team_target).astype(str):
        entry = _select_entry_for_team(
            team=team,
            active_entries=active_entries,
            assigned=assigned,
            candidates=candidates,
            order_index=order_index,
        )
        if entry is None:
            continue
        _assign_entry(
            entry=entry,
            team=team,
            week=week,
            candidates=candidates,
            assigned=assigned,
            team_counts=team_counts,
            reason="minimum-team seed",
        )

    for entry in sorted(
        active_entries,
        key=lambda item: (
            _eligible_option_count(item, candidates),
            order_index[item.entry_id],
            item.entry_id,
        ),
    ):
        if entry.entry_id in assigned:
            continue

        choice = _choose_team_for_entry(
            entry=entry,
            candidates=candidates,
            team_counts=team_counts,
            team_caps=team_caps,
            config=config,
        )
        if choice is None:
            continue

        _assign_entry(
            entry=entry,
            team=choice,
            week=week,
            candidates=candidates,
            assigned=assigned,
            team_counts=team_counts,
            reason="highest adjusted portfolio score",
        )

    allocation_df = _allocation_dataframe(assigned)
    exposure_df = _exposure_summary(allocation_df, candidates, active_count=len(active_entries))
    metrics = _calculate_metrics(
        allocation_df=allocation_df,
        exposure_df=exposure_df,
        active_entry_count=len(active_entries),
        aggression=aggression,
        max_team_exposure_pct=max_team_exposure_pct,
        unallocated_count=len(active_entries) - len(allocation_df),
    )
    return allocation_df, exposure_df, metrics


def _prepare_rankings(
    rankings_df: pd.DataFrame,
    week: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    rankings = rankings_df.copy()
    _require_columns(rankings, REQUIRED_RANKING_COLUMNS, "rankings")
    rankings["week"] = pd.to_numeric(rankings["week"], errors="raise").astype(int)
    rankings = rankings[rankings["week"] == int(week)].copy()
    if rankings.empty:
        raise ValueError(f"No rankings found for week {week}.")

    numeric_columns = [
        "no_vig_win_probability",
        "public_pick_pct",
        "leverage_score",
        "future_cost",
        "scarcity_adjusted_future_cost",
        "final_score",
    ]
    for column in numeric_columns:
        rankings[column] = pd.to_numeric(rankings[column], errors="raise").astype(float)

    rankings["team"] = rankings["team"].astype(str).str.strip()
    rankings["public_pick_pct"] = rankings["public_pick_pct"].clip(lower=0.0, upper=1.0)
    rankings["no_vig_win_probability"] = rankings["no_vig_win_probability"].clip(
        lower=0.0,
        upper=1.0,
    )

    weights = config["weights"]
    # Portfolio score is an equity-oriented blend: survival matters, but lower
    # public ownership, leverage, and future flexibility can beat pure chalk.
    rankings["portfolio_score"] = (
        weights["final_score"] * _normalize(rankings["final_score"])
        + weights["win_probability"] * rankings["no_vig_win_probability"]
        + weights["leverage"] * _normalize(rankings["leverage_score"])
        + weights["uniqueness"] * (1 - rankings["public_pick_pct"])
        + weights["future_flexibility"]
        * (1 - _normalize(rankings["scarcity_adjusted_future_cost"]))
    )

    return rankings.sort_values(
        ["portfolio_score", "final_score", "no_vig_win_probability", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _normalize_entries(
    entries_df: pd.DataFrame,
    week: int,
    personal_entry_count: int | None,
) -> list[_Entry]:
    entries = entries_df.copy() if entries_df is not None else pd.DataFrame()

    if entries.empty:
        normalized: list[_Entry] = []
    elif {"active", "used_teams"}.intersection(entries.columns):
        _require_columns(entries, {"entry_id"}, "entries")
        records = []
        for entry_id, group in entries.groupby("entry_id", sort=True):
            last_row = group.iloc[-1]
            active = _parse_bool(last_row.get("active", True), default=True)
            used_teams: set[str] = set()
            for value in group.get("used_teams", pd.Series(dtype=object)):
                used_teams.update(_parse_used_teams(value))
            records.append(
                _Entry(
                    entry_id=str(entry_id),
                    active=active,
                    used_teams=frozenset(used_teams),
                ),
            )
        normalized = records
    else:
        _require_columns(entries, {"entry_id", "week", "team_picked"}, "entries")
        entries["week"] = pd.to_numeric(entries["week"], errors="raise").astype(int)
        records = []
        for entry_id, group in entries.groupby("entry_id", sort=True):
            previous = group[group["week"] < int(week)].sort_values("week")
            used_teams = {
                str(team).strip()
                for team in previous["team_picked"].dropna()
                if str(team).strip()
            }
            if "is_alive" in previous.columns and not previous.empty:
                active = _parse_bool(previous.iloc[-1]["is_alive"], default=True)
            else:
                active = True
            records.append(
                _Entry(
                    entry_id=str(entry_id),
                    active=active,
                    used_teams=frozenset(used_teams),
                ),
            )
        normalized = records

    if personal_entry_count is not None:
        if personal_entry_count <= 0:
            raise ValueError("personal_entry_count must be positive when provided.")
        normalized = sorted(normalized, key=lambda entry: entry.entry_id)[
            :personal_entry_count
        ]
        existing_ids = {entry.entry_id for entry in normalized}
        next_number = 1
        while len(normalized) < personal_entry_count:
            candidate_id = f"P{next_number:03d}"
            next_number += 1
            if candidate_id in existing_ids:
                continue
            existing_ids.add(candidate_id)
            normalized.append(
                _Entry(
                    entry_id=candidate_id,
                    active=True,
                    used_teams=frozenset(),
                ),
            )

    return sorted(normalized, key=lambda entry: entry.entry_id)


def _team_caps(
    candidates: pd.DataFrame,
    active_count: int,
    base_cap: int,
    config: dict[str, Any],
) -> dict[str, int]:
    caps: dict[str, int] = {}
    low_probability_cap_pct = float(config["low_probability_cap_pct"])
    contrarian_cap_pct = float(config["contrarian_cap_pct"])
    for row in candidates.to_dict("records"):
        cap = base_cap
        win_probability = float(row["no_vig_win_probability"])
        public_pick_pct = float(row["public_pick_pct"])

        if win_probability < 0.50 and low_probability_cap_pct > 0:
            cap = min(cap, max(1, math.floor(active_count * low_probability_cap_pct)))
        elif public_pick_pct <= 0.05 and win_probability < 0.60:
            cap = min(cap, max(1, math.floor(active_count * contrarian_cap_pct)))

        caps[str(row["team"])] = max(1, cap)
    return caps


def _choose_team_for_entry(
    entry: _Entry,
    candidates: pd.DataFrame,
    team_counts: dict[str, int],
    team_caps: dict[str, int],
    config: dict[str, Any],
) -> str | None:
    best_team: str | None = None
    best_score = -math.inf

    for row in candidates.to_dict("records"):
        team = str(row["team"])
        if team in entry.used_teams:
            continue
        if team_counts.get(team, 0) >= team_caps.get(team, 0):
            continue

        cap = max(1, team_caps[team])
        current_exposure = team_counts.get(team, 0) / cap
        adjusted_score = float(row["portfolio_score"]) - (
            float(config["exposure_penalty"]) * current_exposure
        )

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_team = team

    return best_team


def _select_entry_for_team(
    team: str,
    active_entries: list[_Entry],
    assigned: dict[str, dict[str, Any]],
    candidates: pd.DataFrame,
    order_index: dict[str, int],
) -> _Entry | None:
    eligible_entries = [
        entry
        for entry in active_entries
        if entry.entry_id not in assigned and team not in entry.used_teams
    ]
    if not eligible_entries:
        return None
    return sorted(
        eligible_entries,
        key=lambda entry: (
            _eligible_option_count(entry, candidates),
            order_index[entry.entry_id],
            entry.entry_id,
        ),
    )[0]


def _assign_entry(
    entry: _Entry,
    team: str,
    week: int,
    candidates: pd.DataFrame,
    assigned: dict[str, dict[str, Any]],
    team_counts: dict[str, int],
    reason: str,
) -> None:
    row = candidates[candidates["team"] == team].iloc[0].to_dict()
    assigned[entry.entry_id] = {
        "entry_id": entry.entry_id,
        "week": int(week),
        "team": team,
        "no_vig_win_probability": float(row["no_vig_win_probability"]),
        "public_pick_pct": float(row["public_pick_pct"]),
        "leverage_score": float(row["leverage_score"]),
        "future_cost": float(row["future_cost"]),
        "scarcity_adjusted_future_cost": float(row["scarcity_adjusted_future_cost"]),
        "final_score": float(row["final_score"]),
        "portfolio_score": float(row["portfolio_score"]),
        "used_teams": ";".join(sorted(entry.used_teams)),
        "allocation_reason": reason,
    }
    team_counts[team] = team_counts.get(team, 0) + 1


def _allocation_dataframe(assigned: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not assigned:
        return _empty_allocation_df()
    return pd.DataFrame(assigned.values()).sort_values("entry_id").reset_index(drop=True)


def _exposure_summary(
    allocation_df: pd.DataFrame,
    candidates: pd.DataFrame,
    active_count: int,
) -> pd.DataFrame:
    if allocation_df.empty:
        return _empty_exposure_df()

    counts = (
        allocation_df.groupby("team", as_index=False)
        .agg(entries_allocated=("entry_id", "count"))
        .sort_values(["entries_allocated", "team"], ascending=[False, True])
    )
    counts["exposure_pct"] = counts["entries_allocated"] / active_count

    team_columns = [
        "team",
        "no_vig_win_probability",
        "public_pick_pct",
        "leverage_score",
        "future_cost",
        "scarcity_adjusted_future_cost",
        "final_score",
        "portfolio_score",
    ]
    exposure = counts.merge(candidates[team_columns], on="team", how="left")
    return exposure.sort_values(
        ["entries_allocated", "portfolio_score", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _calculate_metrics(
    allocation_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    active_entry_count: int,
    aggression: str,
    max_team_exposure_pct: float,
    unallocated_count: int,
) -> dict[str, float | int | str]:
    if allocation_df.empty:
        return {
            "active_entries": int(active_entry_count),
            "allocated_entries": 0,
            "unallocated_entries": int(unallocated_count),
            "number_of_teams_used": 0,
            "max_team_exposure": 0.0,
            "average_win_probability": 0.0,
            "weighted_public_ownership": 0.0,
            "weighted_leverage_score": 0.0,
            "weighted_future_cost": 0.0,
            "weighted_scarcity_adjusted_future_cost": 0.0,
            "concentration_hhi": 0.0,
            "estimated_probability_at_least_one_survives_independent": 0.0,
            "prob_at_least_one_survives_independent": 0.0,
            "prob_at_least_one_survives_correlated": 0.0,
            "aggression": aggression,
            "max_team_exposure_pct": float(max_team_exposure_pct),
        }

    entry_fail_probability_product = math.prod(
        1 - float(probability)
        for probability in allocation_df["no_vig_win_probability"]
    )
    unique_team_fail_probability_product = math.prod(
        1 - float(probability)
        for probability in exposure_df["no_vig_win_probability"]
    )

    exposure_values = exposure_df["exposure_pct"].astype(float)
    independent_survival_probability = 1 - entry_fail_probability_product
    correlated_survival_probability = 1 - unique_team_fail_probability_product

    return {
        "active_entries": int(active_entry_count),
        "allocated_entries": int(len(allocation_df)),
        "unallocated_entries": int(unallocated_count),
        "number_of_teams_used": int(allocation_df["team"].nunique()),
        "max_team_exposure": float(exposure_values.max()),
        "average_win_probability": float(allocation_df["no_vig_win_probability"].mean()),
        "weighted_public_ownership": float(allocation_df["public_pick_pct"].mean()),
        "weighted_leverage_score": float(allocation_df["leverage_score"].mean()),
        "weighted_future_cost": float(allocation_df["future_cost"].mean()),
        "weighted_scarcity_adjusted_future_cost": float(
            allocation_df["scarcity_adjusted_future_cost"].mean(),
        ),
        # HHI sums squared exposure shares. It is 1.0 for one-team concentration
        # and falls as entries spread across more teams.
        "concentration_hhi": float((exposure_values**2).sum()),
        "estimated_probability_at_least_one_survives_independent": float(
            independent_survival_probability,
        ),
        "prob_at_least_one_survives_independent": float(
            independent_survival_probability,
        ),
        "prob_at_least_one_survives_correlated": float(
            correlated_survival_probability,
        ),
        "aggression": aggression,
        "max_team_exposure_pct": float(max_team_exposure_pct),
    }


def _eligible_entry_counts(
    candidates: pd.DataFrame,
    entries: list[_Entry],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for team in candidates["team"].astype(str):
        counts[team] = sum(1 for entry in entries if team not in entry.used_teams)
    return counts


def _eligible_option_count(entry: _Entry, candidates: pd.DataFrame) -> int:
    return int((~candidates["team"].astype(str).isin(entry.used_teams)).sum())


def _entry_order_index(entries: list[_Entry], rng: random.Random | None) -> dict[str, int]:
    ordered_ids = [entry.entry_id for entry in entries]
    if rng is not None:
        rng.shuffle(ordered_ids)
    return {entry_id: index for index, entry_id in enumerate(ordered_ids)}


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype(float)
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return pd.Series(0.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def _parse_used_teams(value: Any) -> set[str]:
    if isinstance(value, (set, list, tuple)):
        return {str(team).strip() for team in value if str(team).strip()}
    if value is None or pd.isna(value):
        return set()

    text = str(value).strip()
    if not text:
        return set()
    return {team.strip() for team in text.split(";") if team.strip()}


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y", "active", "alive"}:
        return True
    if normalized in {"false", "f", "0", "no", "n", "inactive", "dead"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")


def _require_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{label} data is missing required columns: {missing_text}")


def _empty_allocation_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entry_id",
            "week",
            "team",
            "no_vig_win_probability",
            "public_pick_pct",
            "leverage_score",
            "future_cost",
            "scarcity_adjusted_future_cost",
            "final_score",
            "portfolio_score",
            "used_teams",
            "allocation_reason",
        ],
    )


def _empty_exposure_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team",
            "entries_allocated",
            "exposure_pct",
            "no_vig_win_probability",
            "public_pick_pct",
            "leverage_score",
            "future_cost",
            "scarcity_adjusted_future_cost",
            "final_score",
            "portfolio_score",
        ],
    )
