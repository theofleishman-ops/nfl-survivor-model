"""Survivor pick candidate generation."""

import pandas as pd

from survivor.odds import add_no_vig_probabilities


def generate_weekly_candidates(
    week: int,
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    entries_df: pd.DataFrame | None = None,
    entry_id: str | None = None,
    include_used: bool = True,
) -> pd.DataFrame:
    """Build team-level survivor candidates for a requested week.

    ``entry_id`` is optional because Phase 1 focuses on one-week ranking.  When
    it is provided, candidates include a ``team_already_used`` flag based on
    prior weeks in ``entries_df``.
    """
    _require_columns(schedule_df, {"week", "game_id", "home_team", "away_team"}, "schedule")
    week_schedule = schedule_df[schedule_df["week"].astype(int) == int(week)]
    if week_schedule.empty:
        raise ValueError(f"No scheduled games found for week {week}.")

    odds_with_probs = _ensure_team_level_odds(odds_df)
    candidates = odds_with_probs[odds_with_probs["week"].astype(int) == int(week)].copy()
    if candidates.empty:
        raise ValueError(f"No odds found for week {week}.")

    scheduled_game_ids = set(week_schedule["game_id"])
    candidates = candidates[candidates["game_id"].isin(scheduled_game_ids)].copy()
    if candidates.empty:
        raise ValueError(f"Week {week} schedule and odds game ids do not overlap.")

    public_picks = _normalize_public_picks(public_picks_df)
    candidates = candidates.merge(
        public_picks[public_picks["week"].astype(int) == int(week)],
        on=["week", "team"],
        how="left",
    )
    candidates["public_pick_pct"] = candidates["public_pick_pct"].fillna(0.0)
    candidates["team_already_used"] = candidates["team"].isin(
        _used_teams_for_entry(entries_df, week, entry_id),
    )

    if not include_used:
        candidates = candidates[~candidates["team_already_used"]].copy()

    columns = [
        "week",
        "team",
        "opponent",
        "game_id",
        "is_home",
        "moneyline",
        "implied_probability",
        "no_vig_win_probability",
        "public_pick_pct",
        "team_already_used",
    ]
    return candidates[columns].sort_values(
        ["no_vig_win_probability", "public_pick_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _ensure_team_level_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    if {"team", "opponent", "no_vig_win_probability"}.issubset(odds_df.columns):
        return odds_df.copy()
    return add_no_vig_probabilities(odds_df)


def _normalize_public_picks(public_picks_df: pd.DataFrame) -> pd.DataFrame:
    df = public_picks_df.copy()
    if "public_pick_pct" not in df.columns:
        if "pick_share" in df.columns:
            df = df.rename(columns={"pick_share": "public_pick_pct"})
        else:
            raise ValueError(
                "Public pick data needs public_pick_pct or pick_share column.",
            )

    _require_columns(df, {"week", "team", "public_pick_pct"}, "public picks")
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["public_pick_pct"] = pd.to_numeric(
        df["public_pick_pct"],
        errors="raise",
    ).astype(float)
    if df["public_pick_pct"].max() > 1:
        df["public_pick_pct"] = df["public_pick_pct"] / 100
    return df[["week", "team", "public_pick_pct"]]


def _used_teams_for_entry(
    entries_df: pd.DataFrame | None,
    week: int,
    entry_id: str | None,
) -> set[str]:
    if entries_df is None or entries_df.empty:
        return set()

    entries = entries_df.copy()
    _require_columns(entries, {"week", "team_picked"}, "entries")
    entries["week"] = pd.to_numeric(entries["week"], errors="raise").astype(int)
    previous_picks = entries[entries["week"] < int(week)]

    if entry_id is not None:
        if "entry_id" not in previous_picks.columns:
            raise ValueError("entries data needs entry_id when entry_id is provided.")
        previous_picks = previous_picks[previous_picks["entry_id"] == entry_id]
    elif "entry_id" in previous_picks.columns and previous_picks["entry_id"].nunique() > 1:
        return set()

    return set(previous_picks["team_picked"].dropna().astype(str))


def _require_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{label} data is missing required columns: {missing_text}")
