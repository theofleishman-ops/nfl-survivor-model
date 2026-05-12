"""Monte Carlo simulation engine for survivor pool outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survivor.odds import add_no_vig_probabilities
from survivor.optimizer import rank_weekly_picks
from survivor.public_picks import load_public_picks
from survivor.schedule import load_schedule


DEFAULT_POOL_SIZE = 5000


@dataclass
class PersonalEntryState:
    """Mutable state for one controlled survivor entry during a season."""

    entry_id: str
    alive: bool = True
    used_teams: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WeekSimulationResult:
    """Detailed result for one simulated NFL week."""

    week: int
    public_entries_start: int
    public_entries_survived: int
    public_entries_eliminated: int
    personal_entries_start: int
    personal_entries_survived: int
    personal_entries_eliminated: int
    contest_equity: float
    any_personal_survives: bool
    outcomes: pd.DataFrame
    public_pick_counts: pd.DataFrame
    personal_picks: pd.DataFrame
    leverage_rows: pd.DataFrame


@dataclass(frozen=True)
class SingleSeasonResult:
    """One simulated season, including weekly and entry-level detail."""

    week_results: pd.DataFrame
    leverage_results: pd.DataFrame
    personal_paths: pd.DataFrame
    season_summary: dict[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated output from many Monte Carlo seasons."""

    season_results: pd.DataFrame
    week_results: pd.DataFrame
    week_summary: pd.DataFrame
    leverage_results: pd.DataFrame
    leverage_summary: pd.DataFrame
    personal_paths: pd.DataFrame
    path_summary: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class _PreparedSimulationData:
    weeks: list[int]
    team_odds: pd.DataFrame
    public_picks: pd.DataFrame
    rankings_by_week: dict[int, pd.DataFrame]
    weeks_data: dict[int, "_PreparedWeekData"]


@dataclass(frozen=True)
class _PreparedWeekData:
    week: int
    game_rows: list[dict[str, Any]]
    public_distribution: pd.DataFrame
    ranking_records: list[dict[str, Any]]


def run_placeholder_simulation(
    schedule_path: str | Path,
    public_picks_path: str | Path,
) -> dict[str, int | str]:
    """Prove the starter data flow works by loading sample inputs."""
    schedule_rows = load_schedule(schedule_path)
    public_pick_rows = load_public_picks(public_picks_path)

    return {
        "status": "ok",
        "games_loaded": len(schedule_rows),
        "public_pick_rows_loaded": len(public_pick_rows),
    }


def simulate_game_outcomes(
    odds_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample winners for every game in ``odds_df`` using no-vig probabilities."""
    active_rng = _get_rng(rng=rng, seed=seed)
    team_odds = _ensure_team_level_odds(_as_dataframe(odds_df))
    game_rows = _game_rows_from_team_odds(team_odds)

    sampled = active_rng.random(len(game_rows))
    outcomes: list[dict[str, Any]] = []
    for draw, game in zip(sampled, game_rows, strict=True):
        team_a_wins = draw < game["team_a_probability"]
        winning_team = game["team_a"] if team_a_wins else game["team_b"]
        losing_team = game["team_b"] if team_a_wins else game["team_a"]
        winning_probability = (
            game["team_a_probability"] if team_a_wins else game["team_b_probability"]
        )
        losing_probability = (
            game["team_b_probability"] if team_a_wins else game["team_a_probability"]
        )

        outcomes.append(
            {
                "week": game["week"],
                "game_id": game["game_id"],
                "winning_team": winning_team,
                "losing_team": losing_team,
                "winning_probability": float(winning_probability),
                "losing_probability": float(losing_probability),
                "favorite": game["favorite"],
                "favorite_probability": float(game["favorite_probability"]),
                "upset": winning_team != game["favorite"],
            }
        )

    return pd.DataFrame(outcomes).sort_values(["week", "game_id"]).reset_index(
        drop=True,
    )


def simulate_week(
    week: int,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None = None,
    public_entries: int = DEFAULT_POOL_SIZE,
    personal_states: list[PersonalEntryState] | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> WeekSimulationResult:
    """Simulate one survivor week for the public field and controlled entries."""
    if public_entries < 0:
        raise ValueError("public_entries must be non-negative.")

    active_rng = _get_rng(rng=rng, seed=seed)
    team_odds = _ensure_team_level_odds(_as_dataframe(odds_df))
    week_odds = team_odds[team_odds["week"].astype(int) == int(week)].copy()
    if week_odds.empty:
        raise ValueError(f"No odds found for week {week}.")

    public_picks = _normalize_public_picks(_as_dataframe(public_picks_df))
    rankings = (
        _sort_rankings(_as_dataframe(rankings_df))
        if rankings_df is not None
        else _fallback_rankings(week=week, week_odds=week_odds, public_picks=public_picks)
    )
    rankings = rankings[rankings["week"].astype(int) == int(week)].copy()
    if rankings.empty:
        raise ValueError(f"No rankings found for week {week}.")

    states = personal_states if personal_states is not None else _new_personal_states(1)
    week_data = _prepare_week_data(
        week=week,
        week_odds=week_odds,
        public_picks=public_picks,
        rankings=rankings,
    )
    return _simulate_prepared_week(
        week_data=week_data,
        public_entries=public_entries,
        personal_states=states,
        rng=active_rng,
    )


def simulate_single_season(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None = None,
    entries_df: pd.DataFrame | None = None,
    start_week: int | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    personal_entry_count: int = 1,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> SingleSeasonResult:
    """Simulate one full survivor season from ``start_week`` through the data."""
    active_rng = _get_rng(rng=rng, seed=seed)
    prepared = _prepare_simulation_data(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        rankings_df=rankings_df,
        entries_df=entries_df,
        start_week=start_week,
        pool_size=pool_size,
    )
    return _simulate_single_prepared(
        prepared=prepared,
        pool_size=pool_size,
        personal_entry_count=personal_entry_count,
        rng=active_rng,
    )


def simulate_many_seasons(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    simulations: int = 10000,
    rankings_df: pd.DataFrame | None = None,
    entries_df: pd.DataFrame | None = None,
    start_week: int | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    personal_entry_count: int = 1,
    seed: int | None = None,
) -> SimulationResult:
    """Run many full-season simulations and aggregate survivor equity metrics."""
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = _get_rng(seed=seed)
    prepared = _prepare_simulation_data(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        rankings_df=rankings_df,
        entries_df=entries_df,
        start_week=start_week,
        pool_size=pool_size,
    )

    season_frames: list[pd.DataFrame] = []
    week_frames: list[pd.DataFrame] = []
    leverage_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []

    for simulation_number in range(1, simulations + 1):
        season = _simulate_single_prepared(
            prepared=prepared,
            pool_size=pool_size,
            personal_entry_count=personal_entry_count,
            rng=rng,
        )

        season_row = pd.DataFrame([season.season_summary])
        season_row.insert(0, "simulation", simulation_number)
        season_frames.append(season_row)

        week_results = season.week_results.copy()
        week_results.insert(0, "simulation", simulation_number)
        week_frames.append(week_results)

        leverage_results = season.leverage_results.copy()
        leverage_results.insert(0, "simulation", simulation_number)
        leverage_frames.append(leverage_results)

        personal_paths = season.personal_paths.copy()
        personal_paths.insert(0, "simulation", simulation_number)
        path_frames.append(personal_paths)

    season_results = pd.concat(season_frames, ignore_index=True)
    week_results_all = pd.concat(week_frames, ignore_index=True)
    leverage_results_all = pd.concat(leverage_frames, ignore_index=True)
    personal_paths_all = pd.concat(path_frames, ignore_index=True)

    week_summary = _summarize_weeks(week_results_all)
    leverage_summary = _summarize_leverage(leverage_results_all, week_results_all)
    path_summary = _summarize_paths(personal_paths_all, season_results)
    summary = _summarize_seasons(
        season_results=season_results,
        simulations=simulations,
        start_week=prepared.weeks[0],
        end_week=prepared.weeks[-1],
        pool_size=pool_size,
        personal_entry_count=personal_entry_count,
    )

    return SimulationResult(
        season_results=season_results,
        week_results=week_results_all,
        week_summary=week_summary,
        leverage_results=leverage_results_all,
        leverage_summary=leverage_summary,
        personal_paths=personal_paths_all,
        path_summary=path_summary,
        summary=summary,
    )


def _simulate_single_prepared(
    prepared: _PreparedSimulationData,
    pool_size: int,
    personal_entry_count: int,
    rng: np.random.Generator,
) -> SingleSeasonResult:
    if personal_entry_count <= 0:
        raise ValueError("personal_entry_count must be positive.")
    if pool_size < personal_entry_count:
        raise ValueError("pool_size must be at least personal_entry_count.")

    public_entries = int(pool_size - personal_entry_count)
    personal_states = _new_personal_states(personal_entry_count)
    week_rows: list[dict[str, Any]] = []
    leverage_frames: list[pd.DataFrame] = []

    for week in prepared.weeks:
        result = _simulate_prepared_week(
            week_data=prepared.weeks_data[week],
            public_entries=public_entries,
            personal_states=personal_states,
            rng=rng,
        )

        public_entries = result.public_entries_survived
        total_entries_survived = (
            result.public_entries_survived + result.personal_entries_survived
        )
        week_rows.append(
            {
                "week": int(week),
                "public_entries_start": result.public_entries_start,
                "public_entries_survived": result.public_entries_survived,
                "public_entries_eliminated": result.public_entries_eliminated,
                "personal_entries_start": result.personal_entries_start,
                "personal_entries_survived": result.personal_entries_survived,
                "personal_entries_eliminated": result.personal_entries_eliminated,
                "total_entries_survived": total_entries_survived,
                "contest_equity": result.contest_equity,
                "any_personal_survives": result.any_personal_survives,
            }
        )
        leverage_frames.append(result.leverage_rows)

    week_results = pd.DataFrame(week_rows)
    leverage_results = pd.concat(leverage_frames, ignore_index=True)
    personal_paths = _personal_paths_frame(personal_states, prepared.weeks)
    final_week = week_results.iloc[-1]
    season_summary = {
        "start_week": int(prepared.weeks[0]),
        "end_week": int(prepared.weeks[-1]),
        "public_entries_final": int(final_week["public_entries_survived"]),
        "personal_entries_final": int(final_week["personal_entries_survived"]),
        "total_entries_final": int(final_week["total_entries_survived"]),
        "contest_equity_final": float(final_week["contest_equity"]),
        "at_least_one_personal_survived": bool(final_week["any_personal_survives"]),
    }
    return SingleSeasonResult(
        week_results=week_results,
        leverage_results=leverage_results,
        personal_paths=personal_paths,
        season_summary=season_summary,
    )


def _prepare_simulation_data(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None,
    entries_df: pd.DataFrame | None,
    start_week: int | None,
    pool_size: int,
) -> _PreparedSimulationData:
    schedule = _as_dataframe(schedule_df)
    odds = _as_dataframe(odds_df)
    public_picks = _normalize_public_picks(_as_dataframe(public_picks_df))
    entries = _as_dataframe(entries_df) if entries_df is not None else None
    team_odds = _ensure_team_level_odds(odds)
    weeks = _simulation_weeks(schedule, team_odds, start_week=start_week)
    rankings_by_week = _rankings_by_week(
        weeks=weeks,
        schedule_df=schedule,
        odds_df=odds,
        public_picks_df=public_picks,
        entries_df=entries,
        rankings_df=rankings_df,
        pool_size=pool_size,
    )
    filtered_team_odds = team_odds[team_odds["week"].isin(weeks)].copy()
    weeks_data = {
        week: _prepare_week_data(
            week=week,
            week_odds=filtered_team_odds[filtered_team_odds["week"] == week],
            public_picks=public_picks,
            rankings=rankings_by_week[week],
        )
        for week in weeks
    }
    return _PreparedSimulationData(
        weeks=weeks,
        team_odds=filtered_team_odds,
        public_picks=public_picks,
        rankings_by_week=rankings_by_week,
        weeks_data=weeks_data,
    )


def _rankings_by_week(
    weeks: list[int],
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    entries_df: pd.DataFrame | None,
    rankings_df: pd.DataFrame | None,
    pool_size: int,
) -> dict[int, pd.DataFrame]:
    provided_rankings = (
        _sort_rankings(_as_dataframe(rankings_df)) if rankings_df is not None else None
    )
    rankings_by_week: dict[int, pd.DataFrame] = {}
    for week in weeks:
        if provided_rankings is not None:
            week_rankings = provided_rankings[
                provided_rankings["week"].astype(int) == int(week)
            ].copy()
        else:
            week_rankings = pd.DataFrame()

        if week_rankings.empty:
            week_rankings = rank_weekly_picks(
                week=week,
                schedule_df=schedule_df,
                odds_df=odds_df,
                public_picks_df=public_picks_df,
                entries_df=entries_df,
                pool_size=pool_size,
            )

        rankings_by_week[week] = _sort_rankings(week_rankings)
    return rankings_by_week


def _prepare_week_data(
    week: int,
    week_odds: pd.DataFrame,
    public_picks: pd.DataFrame,
    rankings: pd.DataFrame,
) -> _PreparedWeekData:
    public_distribution = _public_pick_distribution(
        week=week,
        week_odds=week_odds,
        public_picks=public_picks,
    )
    return _PreparedWeekData(
        week=int(week),
        game_rows=_game_rows_from_team_odds(week_odds),
        public_distribution=public_distribution,
        ranking_records=_sort_rankings(rankings).to_dict("records"),
    )


def _simulate_prepared_week(
    week_data: _PreparedWeekData,
    public_entries: int,
    personal_states: list[PersonalEntryState],
    rng: np.random.Generator,
) -> WeekSimulationResult:
    outcomes = _sample_game_rows(week_data.game_rows, rng=rng)
    winning_teams = set(outcomes["winning_team"])

    pick_counts = (
        rng.multinomial(
            int(public_entries),
            week_data.public_distribution["pick_probability"].to_numpy(dtype=float),
        )
        if public_entries > 0
        else np.zeros(len(week_data.public_distribution), dtype=int)
    )
    public_pick_counts = week_data.public_distribution.copy()
    public_pick_counts["simulated_public_picks"] = pick_counts
    public_pick_counts["team_won"] = public_pick_counts["team"].isin(winning_teams)
    public_pick_counts["public_entries_eliminated_by_team"] = np.where(
        public_pick_counts["team_won"],
        0,
        public_pick_counts["simulated_public_picks"],
    )

    public_entries_survived = int(
        public_pick_counts.loc[
            public_pick_counts["team_won"],
            "simulated_public_picks",
        ].sum()
    )
    public_entries_eliminated = int(public_entries - public_entries_survived)

    personal_entries_start = sum(state.alive for state in personal_states)
    personal_pick_rows: list[dict[str, Any]] = []
    for state in personal_states:
        if not state.alive:
            continue

        pick_row = _choose_personal_pick_record(
            state=state,
            ranking_records=week_data.ranking_records,
        )
        if pick_row is None:
            state.alive = False
            personal_pick_rows.append(
                {
                    "entry_id": state.entry_id,
                    "week": int(week_data.week),
                    "team": None,
                    "won": False,
                    "alive_after": False,
                    "no_vig_win_probability": np.nan,
                    "public_pick_pct": np.nan,
                    "rank": np.nan,
                    "final_score": np.nan,
                }
            )
            continue

        team = str(pick_row["team"])
        won = team in winning_teams
        state.used_teams.append(team)
        state.path.append(f"W{int(week_data.week)}:{team}")
        if not won:
            state.alive = False

        rank_value = pick_row.get("rank", np.nan)
        personal_pick_rows.append(
            {
                "entry_id": state.entry_id,
                "week": int(week_data.week),
                "team": team,
                "won": bool(won),
                "alive_after": bool(state.alive),
                "no_vig_win_probability": float(pick_row["no_vig_win_probability"]),
                "public_pick_pct": float(pick_row.get("public_pick_pct", 0.0)),
                "rank": int(rank_value) if not pd.isna(rank_value) else np.nan,
                "final_score": float(pick_row.get("final_score", np.nan)),
            }
        )

    personal_entries_survived = sum(state.alive for state in personal_states)
    personal_entries_eliminated = personal_entries_start - personal_entries_survived
    total_entries_survived = public_entries_survived + personal_entries_survived
    contest_equity = (
        personal_entries_survived / total_entries_survived
        if total_entries_survived > 0
        else 0.0
    )

    public_pick_counts["public_entries_before_week"] = int(public_entries)
    public_pick_counts["public_entries_after_week"] = public_entries_survived
    public_pick_counts["contest_equity_after_week"] = float(contest_equity)
    public_pick_counts["field_shrink_pct_by_team"] = np.where(
        public_entries > 0,
        public_pick_counts["public_entries_eliminated_by_team"] / public_entries,
        0.0,
    )

    leverage_rows = public_pick_counts[
        [
            "week",
            "team",
            "no_vig_win_probability",
            "public_pick_pct",
            "pick_probability",
            "team_won",
            "simulated_public_picks",
            "public_entries_before_week",
            "public_entries_eliminated_by_team",
            "field_shrink_pct_by_team",
            "contest_equity_after_week",
        ]
    ].copy()

    return WeekSimulationResult(
        week=int(week_data.week),
        public_entries_start=int(public_entries),
        public_entries_survived=public_entries_survived,
        public_entries_eliminated=public_entries_eliminated,
        personal_entries_start=int(personal_entries_start),
        personal_entries_survived=int(personal_entries_survived),
        personal_entries_eliminated=int(personal_entries_eliminated),
        contest_equity=float(contest_equity),
        any_personal_survives=personal_entries_survived > 0,
        outcomes=outcomes,
        public_pick_counts=public_pick_counts,
        personal_picks=pd.DataFrame(personal_pick_rows),
        leverage_rows=leverage_rows,
    )


def _sample_game_rows(
    game_rows: list[dict[str, Any]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled = rng.random(len(game_rows))
    outcomes: list[dict[str, Any]] = []
    for draw, game in zip(sampled, game_rows, strict=True):
        team_a_wins = draw < game["team_a_probability"]
        winning_team = game["team_a"] if team_a_wins else game["team_b"]
        losing_team = game["team_b"] if team_a_wins else game["team_a"]
        winning_probability = (
            game["team_a_probability"] if team_a_wins else game["team_b_probability"]
        )
        losing_probability = (
            game["team_b_probability"] if team_a_wins else game["team_a_probability"]
        )
        outcomes.append(
            {
                "week": game["week"],
                "game_id": game["game_id"],
                "winning_team": winning_team,
                "losing_team": losing_team,
                "winning_probability": float(winning_probability),
                "losing_probability": float(losing_probability),
                "favorite": game["favorite"],
                "favorite_probability": float(game["favorite_probability"]),
                "upset": winning_team != game["favorite"],
            }
        )
    return pd.DataFrame(outcomes)


def _choose_personal_pick_record(
    state: PersonalEntryState,
    ranking_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    used = set(state.used_teams)
    for record in ranking_records:
        if str(record["team"]) not in used:
            return record
    return None


def _summarize_weeks(week_results: pd.DataFrame) -> pd.DataFrame:
    grouped = week_results.groupby("week", as_index=False).agg(
        expected_public_entries=("public_entries_survived", "mean"),
        expected_personal_entries=("personal_entries_survived", "mean"),
        expected_total_entries=("total_entries_survived", "mean"),
        probability_at_least_one_personal_survives=(
            "any_personal_survives",
            "mean",
        ),
        expected_contest_equity=("contest_equity", "mean"),
    )
    return grouped.sort_values("week").reset_index(drop=True)


def _summarize_leverage(
    leverage_results: pd.DataFrame,
    week_results: pd.DataFrame,
) -> pd.DataFrame:
    week_equity = week_results.groupby("week")["contest_equity"].mean()
    grouped = leverage_results.groupby(["week", "team"], as_index=False).agg(
        no_vig_win_probability=("no_vig_win_probability", "mean"),
        public_pick_pct=("public_pick_pct", "mean"),
        pick_probability=("pick_probability", "mean"),
        simulated_loss_rate=("team_won", lambda values: 1.0 - float(values.mean())),
        avg_public_picks=("simulated_public_picks", "mean"),
        avg_field_eliminated=("public_entries_eliminated_by_team", "mean"),
        avg_field_shrink_pct=("field_shrink_pct_by_team", "mean"),
    )

    conditional_rows: list[dict[str, Any]] = []
    for (week, team), team_rows in leverage_results.groupby(["week", "team"]):
        loss_rows = team_rows[~team_rows["team_won"]]
        if loss_rows.empty:
            avg_eliminated_if_loses = 0.0
            avg_shrink_if_loses = 0.0
            avg_equity_if_loses = float("nan")
        else:
            avg_eliminated_if_loses = float(
                loss_rows["public_entries_eliminated_by_team"].mean()
            )
            avg_shrink_if_loses = float(loss_rows["field_shrink_pct_by_team"].mean())
            avg_equity_if_loses = float(loss_rows["contest_equity_after_week"].mean())

        baseline_equity = float(week_equity.loc[week])
        equity_lift = (
            avg_equity_if_loses - baseline_equity
            if not pd.isna(avg_equity_if_loses)
            else 0.0
        )
        conditional_rows.append(
            {
                "week": int(week),
                "team": team,
                "avg_field_eliminated_if_team_loses": avg_eliminated_if_loses,
                "avg_field_shrink_pct_if_team_loses": avg_shrink_if_loses,
                "avg_contest_equity_if_team_loses": avg_equity_if_loses,
                "contest_equity_lift_if_team_loses": equity_lift,
            }
        )

    conditional = pd.DataFrame(conditional_rows)
    summary = grouped.merge(conditional, on=["week", "team"], how="left")
    # A simple uniqueness proxy: survival probability multiplied by the share
    # of the field not making the same pick.
    summary["uniqueness_value"] = summary["no_vig_win_probability"] * (
        1 - summary["public_pick_pct"]
    )
    summary["expected_upset_equity_gain"] = summary["simulated_loss_rate"] * summary[
        "contest_equity_lift_if_team_loses"
    ].clip(lower=0)
    return summary.sort_values(
        [
            "expected_upset_equity_gain",
            "avg_field_eliminated_if_team_loses",
            "public_pick_pct",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_paths(
    personal_paths: pd.DataFrame,
    season_results: pd.DataFrame,
) -> pd.DataFrame:
    path_with_equity = personal_paths.merge(
        season_results[["simulation", "contest_equity_final"]],
        on="simulation",
        how="left",
    )
    if path_with_equity.empty:
        return pd.DataFrame(
            columns=[
                "path",
                "entries",
                "survival_rate",
                "avg_weeks_survived",
                "avg_final_contest_equity",
            ]
        )

    summary = path_with_equity.groupby("path", as_index=False).agg(
        entries=("entry_id", "count"),
        survival_rate=("survived_full_season", "mean"),
        avg_weeks_survived=("weeks_survived", "mean"),
        avg_final_contest_equity=("contest_equity_final", "mean"),
    )
    return summary.sort_values(
        ["survival_rate", "avg_final_contest_equity", "entries"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_seasons(
    season_results: pd.DataFrame,
    simulations: int,
    start_week: int,
    end_week: int,
    pool_size: int,
    personal_entry_count: int,
) -> dict[str, Any]:
    return {
        "simulations": int(simulations),
        "start_week": int(start_week),
        "end_week": int(end_week),
        "pool_size": int(pool_size),
        "personal_entry_count": int(personal_entry_count),
        "probability_at_least_one_personal_survives": float(
            season_results["at_least_one_personal_survived"].mean()
        ),
        "expected_final_public_entries": float(
            season_results["public_entries_final"].mean()
        ),
        "expected_final_personal_entries": float(
            season_results["personal_entries_final"].mean()
        ),
        "expected_final_total_entries": float(
            season_results["total_entries_final"].mean()
        ),
        "expected_contest_equity": float(season_results["contest_equity_final"].mean()),
    }


def _personal_paths_frame(
    personal_states: list[PersonalEntryState],
    weeks: list[int],
) -> pd.DataFrame:
    final_week = max(weeks)
    rows = []
    for state in personal_states:
        weeks_survived = len(state.path) if state.alive else max(len(state.path) - 1, 0)
        rows.append(
            {
                "entry_id": state.entry_id,
                "path": " > ".join(state.path) if state.path else "No picks",
                "survived_full_season": bool(state.alive),
                "weeks_survived": int(weeks_survived),
                "final_week": int(final_week if state.alive else weeks[weeks_survived]),
            }
        )
    return pd.DataFrame(rows)


def _choose_personal_pick(
    state: PersonalEntryState,
    rankings: pd.DataFrame,
) -> pd.Series | None:
    available = rankings[~rankings["team"].astype(str).isin(state.used_teams)].copy()
    if available.empty:
        return None
    return _sort_rankings(available).iloc[0]


def _public_pick_distribution(
    week: int,
    week_odds: pd.DataFrame,
    public_picks: pd.DataFrame,
) -> pd.DataFrame:
    teams = week_odds[
        ["week", "team", "no_vig_win_probability"]
    ].drop_duplicates().copy()
    teams["week"] = teams["week"].astype(int)
    week_picks = public_picks[public_picks["week"].astype(int) == int(week)]
    distribution = teams.merge(week_picks, on=["week", "team"], how="left")
    distribution["public_pick_pct"] = distribution["public_pick_pct"].fillna(0.0)

    weights = distribution["public_pick_pct"].to_numpy(dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        probabilities = np.full(len(distribution), 1.0 / len(distribution))
    else:
        probabilities = weights / total_weight

    distribution["pick_probability"] = probabilities
    return distribution.sort_values("team").reset_index(drop=True)


def _fallback_rankings(
    week: int,
    week_odds: pd.DataFrame,
    public_picks: pd.DataFrame,
) -> pd.DataFrame:
    rankings = _public_pick_distribution(
        week=week,
        week_odds=week_odds,
        public_picks=public_picks,
    )
    rankings["final_score"] = (
        rankings["no_vig_win_probability"] * (1 - rankings["public_pick_pct"])
    )
    rankings = rankings.sort_values(
        ["final_score", "no_vig_win_probability"],
        ascending=[False, False],
    ).reset_index(drop=True)
    rankings["rank"] = range(1, len(rankings) + 1)
    return rankings


def _simulation_weeks(
    schedule_df: pd.DataFrame,
    team_odds: pd.DataFrame,
    start_week: int | None,
) -> list[int]:
    if "week" in schedule_df.columns:
        weeks = sorted(schedule_df["week"].astype(int).unique().tolist())
    else:
        weeks = sorted(team_odds["week"].astype(int).unique().tolist())

    if start_week is not None:
        weeks = [week for week in weeks if week >= int(start_week)]
    if not weeks:
        raise ValueError("No simulation weeks found.")
    return weeks


def _game_rows_from_team_odds(team_odds: pd.DataFrame) -> list[dict[str, Any]]:
    game_rows: list[dict[str, Any]] = []
    sort_columns = ["week", "game_id"]
    odds = team_odds.sort_values(sort_columns).copy()
    for (week, game_id), game_df in odds.groupby(["week", "game_id"], sort=True):
        if len(game_df) != 2:
            raise ValueError(f"Game {game_id} in week {week} needs exactly two teams.")

        if "is_home" in game_df.columns:
            game_df = game_df.sort_values("is_home", ascending=False)
        else:
            game_df = game_df.sort_values("team")

        row_a = game_df.iloc[0]
        row_b = game_df.iloc[1]
        prob_a = float(row_a["no_vig_win_probability"])
        prob_b = float(row_b["no_vig_win_probability"])
        total = prob_a + prob_b
        if total <= 0:
            raise ValueError(f"Game {game_id} probabilities must sum positive.")
        prob_a = prob_a / total
        prob_b = prob_b / total
        favorite = str(row_a["team"] if prob_a >= prob_b else row_b["team"])
        favorite_probability = max(prob_a, prob_b)
        game_rows.append(
            {
                "week": int(week),
                "game_id": game_id,
                "team_a": str(row_a["team"]),
                "team_b": str(row_b["team"]),
                "team_a_probability": prob_a,
                "team_b_probability": prob_b,
                "favorite": favorite,
                "favorite_probability": favorite_probability,
            }
        )
    return game_rows


def _ensure_team_level_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    if {"team", "opponent", "no_vig_win_probability"}.issubset(odds_df.columns):
        team_odds = odds_df.copy()
    else:
        team_odds = add_no_vig_probabilities(odds_df)

    required_columns = {"week", "game_id", "team", "no_vig_win_probability"}
    missing = required_columns - set(team_odds.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Odds data is missing required columns: {missing_text}")

    team_odds["week"] = pd.to_numeric(team_odds["week"], errors="raise").astype(int)
    team_odds["no_vig_win_probability"] = pd.to_numeric(
        team_odds["no_vig_win_probability"],
        errors="raise",
    ).astype(float)
    if not team_odds["no_vig_win_probability"].between(0, 1).all():
        raise ValueError("no_vig_win_probability values must be between 0 and 1.")
    return team_odds.sort_values(["week", "game_id", "team"]).reset_index(drop=True)


def _normalize_public_picks(public_picks_df: pd.DataFrame) -> pd.DataFrame:
    picks = public_picks_df.copy()
    if "public_pick_pct" not in picks.columns:
        if "pick_share" in picks.columns:
            picks = picks.rename(columns={"pick_share": "public_pick_pct"})
        else:
            raise ValueError(
                "Public pick data needs public_pick_pct or pick_share column."
            )

    required_columns = {"week", "team", "public_pick_pct"}
    missing = required_columns - set(picks.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Public pick data is missing required columns: {missing_text}")

    picks["week"] = pd.to_numeric(picks["week"], errors="raise").astype(int)
    picks["public_pick_pct"] = pd.to_numeric(
        picks["public_pick_pct"],
        errors="raise",
    ).astype(float)
    if picks["public_pick_pct"].max() > 1:
        picks["public_pick_pct"] = picks["public_pick_pct"] / 100
    if not picks["public_pick_pct"].between(0, 1).all():
        raise ValueError("public_pick_pct values must be between 0 and 1.")
    return picks[["week", "team", "public_pick_pct"]].copy()


def _sort_rankings(rankings_df: pd.DataFrame) -> pd.DataFrame:
    rankings = rankings_df.copy()
    if "week" not in rankings.columns or "team" not in rankings.columns:
        raise ValueError("Rankings data needs week and team columns.")

    sort_columns: list[str]
    ascending: list[bool]
    if "rank" in rankings.columns:
        sort_columns = ["rank"]
        ascending = [True]
    elif "final_score" in rankings.columns:
        sort_columns = ["final_score", "no_vig_win_probability"]
        ascending = [False, False]
    else:
        sort_columns = ["no_vig_win_probability"]
        ascending = [False]

    rankings["week"] = pd.to_numeric(rankings["week"], errors="raise").astype(int)
    if "public_pick_pct" not in rankings.columns:
        rankings["public_pick_pct"] = 0.0
    return rankings.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True,
    )


def _new_personal_states(personal_entry_count: int) -> list[PersonalEntryState]:
    return [
        PersonalEntryState(entry_id=f"MY{i:03d}")
        for i in range(1, personal_entry_count + 1)
    ]


def _as_dataframe(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def _get_rng(
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    return rng if rng is not None else np.random.default_rng(seed)
