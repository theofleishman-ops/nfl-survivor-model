"""Public-field path simulation for survivor contests.

The public field is modeled as weighted survivor entries, not as independent
weekly ownership snapshots.  Each simulated public entry has its own used-team
set and a planned pick path.  Weekly ownership and remaining-field summaries
then emerge from those paths after simulated game outcomes are applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from survivor.public_behavior import (
    PublicBehaviorConfig,
    TEAM_POPULARITY_BIAS,
    public_behavior_from_legacy,
    resolve_public_behavior_config,
)


DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE = 400
DEFAULT_PUBLIC_CHALKINESS = 3.0
DEFAULT_PUBLIC_FUTURE_AWARENESS = 0.20
DEFAULT_PUBLIC_RANDOMNESS = 0.08

PUBLIC_FIELD_PICK_SOURCE = "public_field_path_simulation"
MAX_PATH_CLUSTER_SUMMARY_ROWS = 25000


@dataclass(frozen=True)
class PublicFieldSimulationResult:
    """Pathwise public-field simulation output.

    The DataFrames are intended for reporting.  The numpy matrices are retained
    so path EV can replay the same simulated public paths under a candidate
    entry's conditional survival path without rebuilding public choices.
    """

    entry_paths: pd.DataFrame
    week_by_week_public_ownership: pd.DataFrame
    remaining_field_distribution: pd.DataFrame
    team_usage_exhaustion: pd.DataFrame
    scarcity_by_week: pd.DataFrame
    path_clusters: pd.DataFrame
    path_convergence: pd.DataFrame
    diagnostics: dict[str, Any]
    weeks: list[int]
    teams: list[str]
    team_to_code: dict[str, int]
    game_to_code: dict[str, int]
    pick_matrix: np.ndarray
    pick_game_matrix: np.ndarray
    pick_win_matrix: np.ndarray
    win_matrix: np.ndarray
    entry_weight: float
    pool_size: int
    sample_entry_count: int
    simulations: int
    week_to_index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "week_to_index",
            {int(week): index for index, week in enumerate(self.weeks)},
        )


def simulate_public_field_paths(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    projected_team_strength: pd.DataFrame | None = None,
    start_week: int = 1,
    pool_size: int = 5000,
    simulations: int = 1000,
    random_seed: int | None = None,
    *,
    public_field_sample_size: int | None = DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE,
    chalkiness: float = DEFAULT_PUBLIC_CHALKINESS,
    future_awareness: float = DEFAULT_PUBLIC_FUTURE_AWARENESS,
    randomness: float = DEFAULT_PUBLIC_RANDOMNESS,
    behavior_config: PublicBehaviorConfig | str | None = None,
    team_probabilities_df: pd.DataFrame | None = None,
    weeks: list[int] | tuple[int, ...] | None = None,
    include_entry_paths: bool = True,
) -> PublicFieldSimulationResult:
    """Simulate weighted public-entry season paths with used-team constraints.

    ``pool_size`` is the number of public entries represented by the simulation.
    For path EV integration this is usually total contest entries minus the
    controlled entry.  ``public_field_sample_size`` controls the number of
    weighted representative entries per simulation.
    """
    _validate_positive_int("pool_size", pool_size, allow_zero=True)
    _validate_positive_int("simulations", simulations)
    if public_field_sample_size is not None:
        _validate_positive_int("public_field_sample_size", public_field_sample_size)
    public_behavior = _resolve_behavior_config(
        behavior_config,
        chalkiness=chalkiness,
        future_awareness=future_awareness,
        randomness=randomness,
    )

    team_probabilities = _prepare_team_probabilities(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        projected_team_strength=projected_team_strength,
        team_probabilities_df=team_probabilities_df,
        behavior_config=public_behavior,
    )
    selected_weeks = _resolve_weeks(team_probabilities, start_week, weeks)
    team_probabilities = team_probabilities[
        team_probabilities["week"].astype(int).isin(selected_weeks)
    ].copy()
    if team_probabilities.empty or not selected_weeks:
        return _empty_public_field_result(pool_size=pool_size, simulations=simulations)

    sample_entries = _resolve_sample_entry_count(pool_size, public_field_sample_size)
    if sample_entries <= 0:
        return _empty_public_field_result(
            pool_size=pool_size,
            simulations=simulations,
            weeks=selected_weeks,
            teams=sorted(team_probabilities["team"].astype(str).unique().tolist()),
        )

    prepared = _PreparedPublicField(
        probabilities=team_probabilities,
        weeks=selected_weeks,
        teams=sorted(team_probabilities["team"].astype(str).unique().tolist()),
    )
    rng = np.random.default_rng(random_seed)
    matrices = _simulate_public_pick_matrices(
        prepared=prepared,
        simulations=simulations,
        sample_entries=sample_entries,
        rng=rng,
        behavior_config=public_behavior,
    )
    entry_weight = float(pool_size) / float(sample_entries)
    summaries = _summarize_public_field(
        prepared=prepared,
        matrices=matrices,
        entry_weight=entry_weight,
        include_entry_paths=include_entry_paths,
    )
    diagnostics = {
        "model": PUBLIC_FIELD_PICK_SOURCE,
        "pool_size": int(pool_size),
        "simulations": int(simulations),
        "sample_entry_count": int(sample_entries),
        "entry_weight": float(entry_weight),
        "start_week": int(min(selected_weeks)),
        "end_week": int(max(selected_weeks)),
        "public_behavior_preset": public_behavior.name,
        "public_behavior_description": public_behavior.description,
        "chalkiness": float(public_behavior.chalkiness),
        "future_awareness": float(public_behavior.future_awareness),
        "randomness": float(public_behavior.randomness),
        "popularity_weight": float(public_behavior.popularity_weight),
        "scarcity_weight": float(public_behavior.scarcity_weight),
        "contrarian_rate": float(public_behavior.contrarian_rate),
        "max_single_team_ownership": float(public_behavior.max_single_team_ownership),
        "ownership_temperature": float(public_behavior.ownership_temperature),
        "clustering_strength": float(public_behavior.clustering_strength),
        "elite_path_bias": float(public_behavior.elite_path_bias),
        "late_season_overlap_weight": float(
            public_behavior.late_season_overlap_weight,
        ),
        "path_convergence_temperature": float(
            public_behavior.path_convergence_temperature,
        ),
        "late_season_convergence_score": _mean_or_zero(
            summaries["path_convergence"],
            "top_prefix_share",
        ),
    }

    return PublicFieldSimulationResult(
        entry_paths=summaries["entry_paths"],
        week_by_week_public_ownership=summaries["week_by_week_public_ownership"],
        remaining_field_distribution=summaries["remaining_field_distribution"],
        team_usage_exhaustion=summaries["team_usage_exhaustion"],
        scarcity_by_week=summaries["scarcity_by_week"],
        path_clusters=summaries["path_clusters"],
        path_convergence=summaries["path_convergence"],
        diagnostics=diagnostics,
        weeks=selected_weeks,
        teams=prepared.teams,
        team_to_code=prepared.team_to_code,
        game_to_code=prepared.game_to_code,
        pick_matrix=matrices["pick_matrix"],
        pick_game_matrix=matrices["pick_game_matrix"],
        pick_win_matrix=matrices["pick_win_matrix"],
        win_matrix=matrices["win_matrix"],
        entry_weight=entry_weight,
        pool_size=int(pool_size),
        sample_entry_count=int(sample_entries),
        simulations=int(simulations),
    )


def generate_public_entry_path(
    team_probabilities_df: pd.DataFrame,
    start_week: int = 1,
    random_seed: int | None = None,
    *,
    used_teams: set[str] | list[str] | tuple[str, ...] | None = None,
    chalkiness: float = DEFAULT_PUBLIC_CHALKINESS,
    future_awareness: float = DEFAULT_PUBLIC_FUTURE_AWARENESS,
    randomness: float = DEFAULT_PUBLIC_RANDOMNESS,
    behavior_config: PublicBehaviorConfig | str | None = None,
) -> pd.DataFrame:
    """Generate one public entry's planned pick path."""
    public_behavior = _resolve_behavior_config(
        behavior_config,
        chalkiness=chalkiness,
        future_awareness=future_awareness,
        randomness=randomness,
    )
    probabilities = _normalize_team_probabilities(_as_dataframe(team_probabilities_df))
    weeks = _resolve_weeks(probabilities, start_week, None)
    rng = np.random.default_rng(random_seed)
    used = {str(team).strip() for team in used_teams or () if str(team).strip()}
    rows: list[dict[str, Any]] = []
    for week in weeks:
        week_rows = probabilities[probabilities["week"].astype(int) == int(week)].copy()
        pick = choose_public_pick(
            week_rows,
            used,
            rng=rng,
            all_probabilities=probabilities,
            behavior_config=public_behavior,
        )
        if pick is None:
            continue
        team = str(pick["team"])
        used.add(team)
        rows.append(
            {
                "week": int(week),
                "team": team,
                "opponent": str(pick.get("opponent", "")),
                "game_id": pick.get("game_id"),
                "win_probability": float(pick["win_probability"]),
                "projected_public_pick_pct": float(
                    pick.get("projected_public_pick_pct", 0.0),
                ),
            },
        )
    return pd.DataFrame(rows)


def choose_public_pick(
    week_probabilities_df: pd.DataFrame,
    used_teams: set[str] | list[str] | tuple[str, ...],
    rng: np.random.Generator | None = None,
    *,
    all_probabilities: pd.DataFrame | None = None,
    chalkiness: float = DEFAULT_PUBLIC_CHALKINESS,
    future_awareness: float = DEFAULT_PUBLIC_FUTURE_AWARENESS,
    randomness: float = DEFAULT_PUBLIC_RANDOMNESS,
    behavior_config: PublicBehaviorConfig | str | None = None,
) -> dict[str, Any] | None:
    """Choose one public pick from eligible teams using chalk-biased weights."""
    public_behavior = _resolve_behavior_config(
        behavior_config,
        chalkiness=chalkiness,
        future_awareness=future_awareness,
        randomness=randomness,
    )
    active_rng = rng or np.random.default_rng()
    week_df = _normalize_team_probabilities(_as_dataframe(week_probabilities_df))
    if week_df.empty:
        return None
    used = {str(team).strip() for team in used_teams if str(team).strip()}
    available = week_df[~week_df["team"].astype(str).isin(used)].copy()
    if available.empty:
        return None
    weights = _choice_weights(
        available,
        all_probabilities if all_probabilities is not None else week_df,
        behavior_config=public_behavior,
    )
    total = float(weights.sum())
    probabilities = (
        weights / total if total > 0 else np.full(len(available), 1.0 / len(available))
    )
    index = int(active_rng.choice(np.arange(len(available)), p=probabilities))
    return available.iloc[index].to_dict()


def update_remaining_public_entries(
    entry_paths_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    *,
    week: int | None = None,
) -> pd.DataFrame:
    """Apply game outcomes to public entry paths and return alive-state rows."""
    paths = _as_dataframe(entry_paths_df)
    outcomes = _as_dataframe(outcomes_df)
    if paths.empty:
        return pd.DataFrame()
    required = {"simulation", "entry_id", "week", "team"}
    missing = required - set(paths.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"entry paths are missing required columns: {missing_text}")
    if "winning_team" not in outcomes.columns:
        raise ValueError("outcomes data is missing required column: winning_team")

    rows = paths.copy()
    if week is not None:
        rows = rows[rows["week"].astype(int) <= int(week)].copy()
    winning = outcomes[["simulation", "week", "winning_team"]].copy()
    winning["week"] = pd.to_numeric(winning["week"], errors="raise").astype(int)
    winning["won_marker"] = True
    merged = rows.merge(
        winning,
        left_on=["simulation", "week", "team"],
        right_on=["simulation", "week", "winning_team"],
        how="left",
    )
    merged["won"] = merged["won_marker"].fillna(False).astype(bool)
    merged = merged.drop(columns=["winning_team", "won_marker"], errors="ignore")
    merged = merged.sort_values(["simulation", "entry_id", "week"]).reset_index(drop=True)
    merged["alive_after"] = merged.groupby(["simulation", "entry_id"])["won"].cummin()
    merged["alive_before"] = (
        merged.groupby(["simulation", "entry_id"])["alive_after"]
        .shift(fill_value=True)
        .astype(bool)
    )
    return merged


@dataclass(frozen=True)
class _PreparedPublicField:
    probabilities: pd.DataFrame
    weeks: list[int]
    teams: list[str]

    @property
    def team_to_code(self) -> dict[str, int]:
        return {team: index for index, team in enumerate(self.teams)}

    @property
    def game_to_code(self) -> dict[str, int]:
        game_ids = sorted(self.probabilities["game_id"].astype(str).unique().tolist())
        return {game_id: index for index, game_id in enumerate(game_ids)}


def _simulate_public_pick_matrices(
    *,
    prepared: _PreparedPublicField,
    simulations: int,
    sample_entries: int,
    rng: np.random.Generator,
    behavior_config: PublicBehaviorConfig,
) -> dict[str, np.ndarray]:
    week_count = len(prepared.weeks)
    team_count = len(prepared.teams)
    pick_matrix = np.full((simulations, sample_entries, week_count), -1, dtype=np.int16)
    pick_game_matrix = np.full(
        (simulations, sample_entries, week_count),
        -1,
        dtype=np.int16,
    )
    pick_win_matrix = np.zeros((simulations, sample_entries, week_count), dtype=bool)
    win_matrix = np.zeros((week_count, simulations, team_count), dtype=bool)

    used = np.zeros((simulations * sample_entries, team_count), dtype=bool)
    row_indices = np.arange(simulations * sample_entries)
    simulation_indices = np.repeat(np.arange(simulations), sample_entries)
    entry_indices = np.tile(np.arange(sample_entries), simulations)
    cohort_count = max(1, min(sample_entries, int(np.sqrt(max(sample_entries, 1)))))
    entry_cohorts = entry_indices % cohort_count
    clustering_strength = float(behavior_config.clustering_strength)
    clustered_entries = (
        rng.random(simulations * sample_entries) < min(max(clustering_strength, 0.0), 1.0)
        if clustering_strength > 0
        else np.zeros(simulations * sample_entries, dtype=bool)
    )
    team_to_code = prepared.team_to_code
    game_to_code = prepared.game_to_code

    for week_index, week in enumerate(prepared.weeks):
        week_df = prepared.probabilities[
            prepared.probabilities["week"].astype(int) == int(week)
        ].copy()
        week_weights = _week_code_weights(
            week_df,
            prepared.probabilities,
            prepared.teams,
            behavior_config=behavior_config,
        )
        eligible_weights = np.broadcast_to(
            week_weights,
            (simulations * sample_entries, team_count),
        ).astype(np.float64, copy=True)
        eligible_weights[used] = 0.0
        totals = eligible_weights.sum(axis=1)
        no_options = totals <= 0
        if bool(no_options.any()):
            fallback = (week_weights > 0) & ~used[no_options]
            fallback_totals = fallback.sum(axis=1)
            replacement = np.divide(
                fallback,
                fallback_totals[:, np.newaxis],
                out=np.zeros_like(fallback, dtype=float),
                where=fallback_totals[:, np.newaxis] > 0,
            )
            eligible_weights[no_options] = replacement
            totals = eligible_weights.sum(axis=1)

        eligible_weights = _apply_path_clustering_pressure(
            eligible_weights,
            week_df,
            prepared.probabilities,
            prepared.teams,
            week_index=week_index,
            week_count=week_count,
            behavior_config=behavior_config,
        )
        totals = eligible_weights.sum(axis=1)
        choices = np.full(simulations * sample_entries, -1, dtype=np.int16)
        valid_rows = totals > 0
        cumulative = np.cumsum(eligible_weights[valid_rows], axis=1)
        draw_quantiles = rng.random(int(valid_rows.sum()))
        if clustering_strength > 0 and int(valid_rows.sum()) > 0:
            cohort_draws = rng.random((simulations, cohort_count))
            common_draws = cohort_draws[
                simulation_indices[valid_rows],
                entry_cohorts[valid_rows],
            ]
            draw_quantiles = np.where(
                clustered_entries[valid_rows],
                common_draws,
                draw_quantiles,
            )
        draws = draw_quantiles * totals[valid_rows]
        choices[valid_rows] = (
            cumulative >= draws[:, np.newaxis]
        ).argmax(axis=1).astype(np.int16)
        pick_matrix[:, :, week_index] = choices.reshape(simulations, sample_entries)
        used[row_indices[valid_rows], choices[valid_rows]] = True

        game_codes_by_team = np.full(team_count, -1, dtype=np.int16)
        win_probabilities_by_team = np.full(team_count, np.nan, dtype=float)
        for row in week_df.to_dict("records"):
            team_code = team_to_code[str(row["team"])]
            game_codes_by_team[team_code] = game_to_code[str(row["game_id"])]
            win_probabilities_by_team[team_code] = float(row["win_probability"])

        valid_choices = choices >= 0
        choice_game_codes = np.full_like(choices, -1, dtype=np.int16)
        choice_game_codes[valid_choices] = game_codes_by_team[choices[valid_choices]]
        pick_game_matrix[:, :, week_index] = choice_game_codes.reshape(
            simulations,
            sample_entries,
        )
        _sample_week_winners(
            win_matrix=win_matrix,
            week_index=week_index,
            week_df=week_df,
            team_to_code=team_to_code,
            rng=rng,
            simulations=simulations,
        )
        week_wins = win_matrix[week_index]
        pick_wins = np.zeros(simulations * sample_entries, dtype=bool)
        pick_wins[valid_choices] = week_wins[
            simulation_indices[valid_choices],
            choices[valid_choices],
        ]
        pick_win_matrix[:, :, week_index] = pick_wins.reshape(
            simulations,
            sample_entries,
        )

    return {
        "pick_matrix": pick_matrix,
        "pick_game_matrix": pick_game_matrix,
        "pick_win_matrix": pick_win_matrix,
        "win_matrix": win_matrix,
    }


def _sample_week_winners(
    *,
    win_matrix: np.ndarray,
    week_index: int,
    week_df: pd.DataFrame,
    team_to_code: dict[str, int],
    rng: np.random.Generator,
    simulations: int,
) -> None:
    for _, game_df in week_df.groupby("game_id", sort=True):
        game = game_df.sort_values("team").reset_index(drop=True)
        if len(game) < 2:
            continue
        team_a = str(game.iloc[0]["team"])
        team_b = str(game.iloc[1]["team"])
        team_a_code = team_to_code[team_a]
        team_b_code = team_to_code[team_b]
        probability_a = float(game.iloc[0]["win_probability"])
        draws = rng.random(simulations)
        a_wins = draws < probability_a
        win_matrix[week_index, :, team_a_code] = a_wins
        win_matrix[week_index, :, team_b_code] = ~a_wins


def _summarize_public_field(
    *,
    prepared: _PreparedPublicField,
    matrices: dict[str, np.ndarray],
    entry_weight: float,
    include_entry_paths: bool,
) -> dict[str, pd.DataFrame]:
    pick_matrix = matrices["pick_matrix"]
    pick_win_matrix = matrices["pick_win_matrix"]
    simulations, sample_entries, week_count = pick_matrix.shape
    team_count = len(prepared.teams)

    alive_before = np.ones((simulations, sample_entries), dtype=bool)
    used_counts = np.zeros((simulations, team_count), dtype=float)
    ownership_rows: list[dict[str, Any]] = []
    remaining_rows: list[dict[str, Any]] = []
    exhaustion_rows: list[dict[str, Any]] = []
    scarcity_rows: list[dict[str, Any]] = []
    entry_path_frames: list[pd.DataFrame] = []

    for week_index, week in enumerate(prepared.weeks):
        picks = pick_matrix[:, :, week_index]
        wins = pick_win_matrix[:, :, week_index]
        alive_after = alive_before & wins

        for team_code, team in enumerate(prepared.teams):
            picked_alive = alive_before & (picks == team_code)
            alive_team_count = picked_alive.sum(axis=1).astype(float) * entry_weight
            alive_before_count = alive_before.sum(axis=1).astype(float) * entry_weight
            projected_ownership = np.divide(
                alive_team_count,
                alive_before_count,
                out=np.zeros_like(alive_team_count),
                where=alive_before_count > 0,
            )
            ownership_rows.append(
                {
                    "week": int(week),
                    "team": team,
                    "expected_public_picks": float(alive_team_count.mean()),
                    "projected_ownership_pct": float(projected_ownership.mean()),
                },
            )

            used_counts[:, team_code] += (picks == team_code).sum(axis=1) * entry_weight
            burned_pct = used_counts[:, team_code] / max(entry_weight * sample_entries, 1.0)
            remaining_used = (
                (alive_after & _team_used_through_week(pick_matrix, team_code, week_index))
                .sum(axis=1)
                .astype(float)
                * entry_weight
            )
            remaining_entries = alive_after.sum(axis=1).astype(float) * entry_weight
            expected_owned_pct = np.divide(
                remaining_used,
                remaining_entries,
                out=np.zeros_like(remaining_used),
                where=remaining_entries > 0,
            )
            remaining_rows.append(
                {
                    "week": int(week),
                    "team": team,
                    "expected_remaining_entries_with_team_used": float(
                        remaining_used.mean(),
                    ),
                    "remaining_field_used_pct": float(expected_owned_pct.mean()),
                },
            )
            exhaustion_rows.append(
                {
                    "week": int(week),
                    "team": team,
                    "expected_entries_burned_team": float(used_counts[:, team_code].mean()),
                    "burned_pct": float(burned_pct.mean()),
                    "expected_remaining_entries_with_team_available": float(
                        (remaining_entries - remaining_used).mean(),
                    ),
                    "remaining_field_available_pct": float(
                        1.0 - expected_owned_pct.mean(),
                    ),
                },
            )

        week_ownership = pd.DataFrame(
            [row for row in ownership_rows if int(row["week"]) == int(week)],
        )
        expected_remaining = (
            alive_after.sum(axis=1).astype(float) * entry_weight
        ).mean()
        available_counts = []
        for sim_index in range(simulations):
            alive_entries = alive_after[sim_index]
            if not bool(alive_entries.any()):
                available_counts.append(0.0)
                continue
            used_by_alive = _used_team_count_by_entry(
                pick_matrix[sim_index : sim_index + 1, :, : week_index + 1],
                team_count,
            )[0]
            available_counts.extend(
                (team_count - used_by_alive[alive_entries]).astype(float).tolist(),
            )
        average_available = float(np.mean(available_counts)) if available_counts else 0.0
        top_ownership = (
            float(week_ownership["projected_ownership_pct"].max())
            if not week_ownership.empty
            else 0.0
        )
        chalk_concentration = float(
            week_ownership["projected_ownership_pct"].nlargest(3).sum()
            if not week_ownership.empty
            else 0.0
        )
        scarcity_index = _scarcity_index(
            expected_remaining=expected_remaining,
            average_available_teams=average_available,
            team_count=team_count,
            top_ownership=top_ownership,
            chalk_concentration=chalk_concentration,
        )
        scarcity_rows.append(
            {
                "week": int(week),
                "expected_remaining_entries": float(expected_remaining),
                "average_available_teams": average_available,
                "max_projected_ownership_pct": top_ownership,
                "expected_chalk_concentration": chalk_concentration,
                "scarcity_index": scarcity_index,
                "scarcity_spike": bool(scarcity_index >= 0.65 or top_ownership >= 0.45),
            },
        )

        if include_entry_paths:
            entry_path_frames.append(
                _entry_paths_for_week(
                    prepared=prepared,
                    picks=picks,
                    wins=wins,
                    alive_before=alive_before,
                    alive_after=alive_after,
                    week=week,
                    week_index=week_index,
                    entry_weight=entry_weight,
                ),
            )
        alive_before = alive_after

    entry_paths = (
        pd.concat(entry_path_frames, ignore_index=True)
        if entry_path_frames
        else pd.DataFrame()
    )
    path_clusters = _path_cluster_summaries(
        prepared=prepared,
        pick_matrix=pick_matrix,
        pick_win_matrix=pick_win_matrix,
        entry_weight=entry_weight,
    )
    path_convergence = _path_convergence_by_week(
        prepared=prepared,
        pick_matrix=pick_matrix,
        entry_weight=entry_weight,
    )
    return {
        "entry_paths": entry_paths,
        "week_by_week_public_ownership": pd.DataFrame(ownership_rows).sort_values(
            ["week", "projected_ownership_pct", "team"],
            ascending=[True, False, True],
        ).reset_index(drop=True),
        "remaining_field_distribution": pd.DataFrame(remaining_rows).sort_values(
            ["week", "remaining_field_used_pct", "team"],
            ascending=[True, False, True],
        ).reset_index(drop=True),
        "team_usage_exhaustion": pd.DataFrame(exhaustion_rows).sort_values(
            ["week", "burned_pct", "team"],
            ascending=[True, False, True],
        ).reset_index(drop=True),
        "scarcity_by_week": pd.DataFrame(scarcity_rows).sort_values("week").reset_index(
            drop=True,
        ),
        "path_clusters": path_clusters,
        "path_convergence": path_convergence,
    }


def _entry_paths_for_week(
    *,
    prepared: _PreparedPublicField,
    picks: np.ndarray,
    wins: np.ndarray,
    alive_before: np.ndarray,
    alive_after: np.ndarray,
    week: int,
    week_index: int,
    entry_weight: float,
) -> pd.DataFrame:
    simulations, sample_entries = picks.shape
    simulation_values = np.repeat(np.arange(1, simulations + 1), sample_entries)
    entry_values = np.tile(np.arange(1, sample_entries + 1), simulations)
    pick_values = picks.reshape(-1)
    rows = pd.DataFrame(
        {
            "simulation": simulation_values,
            "entry_id": entry_values,
            "week": int(week),
            "team": [
                prepared.teams[int(code)] if int(code) >= 0 else None
                for code in pick_values
            ],
            "alive_before": alive_before.reshape(-1),
            "won": wins.reshape(-1),
            "alive_after": alive_after.reshape(-1),
            "entry_weight": float(entry_weight),
        },
    )
    enrich = prepared.probabilities[
        prepared.probabilities["week"].astype(int) == int(week)
    ][
        [
            "team",
            "opponent",
            "game_id",
            "win_probability",
            "projected_public_pick_pct",
        ]
    ].copy()
    rows = rows.merge(enrich, on="team", how="left")
    rows["path_week_index"] = int(week_index + 1)
    return rows


def _path_cluster_summaries(
    *,
    prepared: _PreparedPublicField,
    pick_matrix: np.ndarray,
    pick_win_matrix: np.ndarray,
    entry_weight: float,
    late_window: int = 4,
) -> pd.DataFrame:
    simulations, sample_entries, week_count = pick_matrix.shape
    if simulations <= 0 or sample_entries <= 0 or week_count <= 0:
        return pd.DataFrame()

    row_weight = float(entry_weight) / max(float(simulations), 1.0)
    flat_paths = pick_matrix.reshape(simulations * sample_entries, week_count)
    survived = pick_win_matrix.all(axis=2).reshape(simulations * sample_entries)
    if flat_paths.size == 0:
        return pd.DataFrame()
    flat_paths, survived, sample_scale = _deterministic_summary_sample(
        flat_paths,
        survived,
        max_rows=MAX_PATH_CLUSTER_SUMMARY_ROWS,
    )
    row_weight *= sample_scale
    unique_paths, inverse, counts = np.unique(
        flat_paths,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    survivor_counts = np.bincount(
        inverse,
        weights=survived.astype(float),
        minlength=len(unique_paths),
    )
    late_count = min(max(int(late_window), 1), week_count)
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for codes, count, survivor_count in zip(
        unique_paths,
        counts,
        survivor_counts,
        strict=False,
    ):
        if not _has_valid_codes(codes):
            continue
        late_codes = tuple(int(code) for code in codes[-late_count:])
        current = grouped.setdefault(
            late_codes,
            {
                "expected_entries": 0.0,
                "expected_surviving_entries": 0.0,
                "observed_path_count": 0,
                "member_path_count": 0,
                "representative_codes": codes,
                "representative_count": -1,
            },
        )
        current["expected_entries"] += float(count) * row_weight
        current["expected_surviving_entries"] += float(survivor_count) * row_weight
        current["observed_path_count"] += int(count)
        current["member_path_count"] += 1
        if int(count) > int(current["representative_count"]):
            current["representative_codes"] = codes
            current["representative_count"] = int(count)

    if not grouped:
        return pd.DataFrame()
    total_entries = max(
        float(sum(item["expected_entries"] for item in grouped.values())),
        1.0,
    )
    cluster_rows: list[dict[str, Any]] = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (item[1]["expected_entries"], item[1]["member_path_count"]),
        reverse=True,
    )
    for cluster_id, (late_codes, group) in enumerate(
        sorted_groups[:25],
        start=1,
    ):
        expected_entries = float(group["expected_entries"])
        expected_survivors = float(group["expected_surviving_entries"])
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "path_key": _path_key_from_codes(
                    np.asarray(group["representative_codes"], dtype=np.int16),
                    prepared.weeks,
                    prepared.teams,
                ),
                "late_path_key": _path_key_from_codes(
                    np.asarray(late_codes, dtype=np.int16),
                    prepared.weeks[-late_count:],
                    prepared.teams,
                ),
                "expected_entries": expected_entries,
                "expected_surviving_entries": expected_survivors,
                "observed_path_count": int(group["observed_path_count"]),
                "member_path_count": int(group["member_path_count"]),
                "cluster_share": expected_entries / total_entries,
                "surviving_cluster_share": expected_survivors / total_entries,
                "late_season_congestion_score": min(
                    max(expected_entries / total_entries, 0.0),
                    1.0,
                ),
            },
        )
    return pd.DataFrame(cluster_rows).reset_index(drop=True)


def _path_convergence_by_week(
    *,
    prepared: _PreparedPublicField,
    pick_matrix: np.ndarray,
    entry_weight: float,
) -> pd.DataFrame:
    simulations, sample_entries, week_count = pick_matrix.shape
    if simulations <= 0 or sample_entries <= 0 or week_count <= 0:
        return pd.DataFrame()
    row_weight = float(entry_weight) / max(float(simulations), 1.0)
    flat_paths = pick_matrix.reshape(simulations * sample_entries, week_count)
    flat_paths, _, sample_scale = _deterministic_summary_sample(
        flat_paths,
        None,
        max_rows=MAX_PATH_CLUSTER_SUMMARY_ROWS,
    )
    row_weight *= sample_scale
    unique_paths, counts = np.unique(flat_paths, axis=0, return_counts=True)
    valid_indices = [
        index for index, codes in enumerate(unique_paths) if _has_valid_codes(codes)
    ]
    if not valid_indices:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for week_index, week in enumerate(prepared.weeks):
        prefixes: dict[tuple[int, ...], int] = {}
        for index in valid_indices:
            prefix = tuple(int(code) for code in unique_paths[index, : week_index + 1])
            prefixes[prefix] = prefixes.get(prefix, 0) + int(counts[index])
        top_prefix_codes, top_count = max(
            prefixes.items(),
            key=lambda item: (item[1], item[0]),
        )
        top_entries = float(top_count) * row_weight
        total_entries = max(float(sum(prefixes.values())) * row_weight, 1.0)
        top_prefix = _path_key_from_codes(
            np.asarray(top_prefix_codes, dtype=np.int16),
            prepared.weeks[: week_index + 1],
            prepared.teams,
        )
        rows.append(
            {
                "week": int(week),
                "top_prefix": top_prefix,
                "expected_entries_on_top_prefix": float(top_entries),
                "top_prefix_share": float(top_entries) / float(total_entries),
                "distinct_prefix_count": int(len(valid_indices)),
            },
        )
    return pd.DataFrame(rows)


def _path_key_from_codes(
    codes: np.ndarray,
    weeks: list[int],
    teams: list[str],
) -> str:
    return " > ".join(
        f"W{int(week)}:{teams[int(code)]}"
        for week, code in zip(weeks, codes, strict=False)
        if int(code) >= 0
    )


def _has_valid_codes(codes: np.ndarray) -> bool:
    return bool(np.asarray(codes).size > 0 and (np.asarray(codes) >= 0).any())


def _deterministic_summary_sample(
    rows: np.ndarray,
    survived: np.ndarray | None,
    *,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    row_count = int(rows.shape[0])
    if row_count <= int(max_rows):
        return rows, survived, 1.0
    indices = np.linspace(0, row_count - 1, int(max_rows), dtype=int)
    sampled_survived = survived[indices] if survived is not None else None
    return rows[indices], sampled_survived, float(row_count) / float(len(indices))


def _team_used_through_week(
    pick_matrix: np.ndarray,
    team_code: int,
    week_index: int,
) -> np.ndarray:
    return (pick_matrix[:, :, : week_index + 1] == int(team_code)).any(axis=2)


def _used_team_count_by_entry(pick_subset: np.ndarray, team_count: int) -> np.ndarray:
    simulations, sample_entries, _ = pick_subset.shape
    counts = np.zeros((simulations, sample_entries), dtype=int)
    for team_code in range(team_count):
        counts += (pick_subset == team_code).any(axis=2)
    return counts


def _apply_path_clustering_pressure(
    eligible_weights: np.ndarray,
    week_df: pd.DataFrame,
    all_probabilities: pd.DataFrame,
    teams: list[str],
    *,
    week_index: int,
    week_count: int,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    strength = float(behavior_config.clustering_strength)
    if strength <= 0 or eligible_weights.size == 0:
        return eligible_weights

    convergence = _late_convergence_factor(
        week_index=week_index,
        week_count=week_count,
        late_season_overlap_weight=float(behavior_config.late_season_overlap_weight),
    )
    if convergence <= 0:
        return eligible_weights

    pressure = _elite_path_pressure(
        week_df,
        all_probabilities,
        teams,
        behavior_config=behavior_config,
    )
    shaped = eligible_weights * (
        1.0
        + strength
        * float(behavior_config.elite_path_bias)
        * convergence
        * pressure[np.newaxis, :]
    )

    target_temperature = float(behavior_config.path_convergence_temperature)
    effective_temperature = 1.0 + strength * convergence * (target_temperature - 1.0)
    effective_temperature = max(effective_temperature, 0.05)
    if effective_temperature != 1.0:
        shaped = np.where(shaped > 0, shaped ** (1.0 / effective_temperature), 0.0)
    return shaped


def _elite_path_pressure(
    week_df: pd.DataFrame,
    all_probabilities: pd.DataFrame,
    teams: list[str],
    *,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    pressure = np.zeros(len(teams), dtype=float)
    if week_df.empty:
        return pressure

    current_week = int(week_df["week"].astype(int).min())
    future = all_probabilities[
        all_probabilities["week"].astype(int) > current_week
    ].copy()
    future_best = (
        future.groupby("team")["win_probability"].max().astype(float).to_dict()
        if not future.empty
        else {}
    )
    ownership_column = (
        "projected_public_pick_pct"
        if "projected_public_pick_pct" in week_df.columns
        else None
    )
    team_positions = {team: index for index, team in enumerate(teams)}
    for row in week_df.to_dict("records"):
        team = str(row["team"])
        index = team_positions.get(team)
        if index is None:
            continue
        win_probability = float(row.get("win_probability", 0.0))
        ownership = float(row.get(ownership_column, 0.0)) if ownership_column else 0.0
        future_probability = float(future_best.get(team, 0.0))
        favorite_signal = max(win_probability - 0.55, 0.0) / 0.35
        future_signal = max(future_probability - 0.60, 0.0) / 0.30
        ownership_signal = max(ownership - 0.10, 0.0) / 0.40
        pressure[index] = min(
            1.0,
            0.50 * favorite_signal
            + 0.30 * future_signal * (1.0 + float(behavior_config.future_awareness) * 0.15)
            + 0.20 * ownership_signal,
        )
    return pressure


def _late_convergence_factor(
    *,
    week_index: int,
    week_count: int,
    late_season_overlap_weight: float,
) -> float:
    if week_count <= 1:
        return 0.0
    position = float(week_index) / float(max(week_count - 1, 1))
    late_weight = max(float(late_season_overlap_weight), 1.0)
    shaped = position ** 0.75
    return float(min(max(shaped * late_weight / 2.0, 0.0), 1.0))


def _week_code_weights(
    week_df: pd.DataFrame,
    all_probabilities: pd.DataFrame,
    teams: list[str],
    *,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    weights = np.zeros(len(teams), dtype=float)
    raw = _choice_weights(
        week_df,
        all_probabilities,
        behavior_config=behavior_config,
    )
    team_positions = {team: index for index, team in enumerate(teams)}
    for value, team in zip(raw, week_df["team"].astype(str), strict=False):
        weights[team_positions[team]] = float(value)
    total = float(weights.sum())
    return weights / total if total > 0 else weights


def _choice_weights(
    week_df: pd.DataFrame,
    all_probabilities: pd.DataFrame,
    *,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    rows = week_df.copy()
    if rows.empty:
        return np.array([], dtype=float)
    probabilities = rows["win_probability"].astype(float).clip(lower=0.01, upper=0.99)
    popularity = _popularity_weights(rows, behavior_config=behavior_config)
    future_penalty = _future_value_penalty(
        rows,
        all_probabilities,
        behavior_config=behavior_config,
    )
    scarcity = _week_alternative_scarcity(probabilities)
    favorite_boost = np.where(
        probabilities.to_numpy() >= 0.70,
        1.0 + float(behavior_config.scarcity_weight) * 0.50 * scarcity,
        1.0,
    )
    base = (
        probabilities.to_numpy() ** float(behavior_config.chalkiness)
    ) * popularity * future_penalty * favorite_boost
    contrarian = (
        probabilities.to_numpy() ** max(1.0, float(behavior_config.chalkiness) * 0.55)
    ) * (1.0 / np.maximum(popularity, 0.05)) * future_penalty
    structured = (
        (1.0 - float(behavior_config.contrarian_rate)) * base
        + float(behavior_config.contrarian_rate) * contrarian
    )
    random_component = np.ones(len(rows), dtype=float)
    raw = (
        (1.0 - float(behavior_config.randomness)) * structured
        + float(behavior_config.randomness) * random_component
    )
    return _apply_ownership_shape(raw, behavior_config)


def _popularity_weights(
    rows: pd.DataFrame,
    *,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    if "projected_public_pick_pct" in rows.columns:
        pct = rows["projected_public_pick_pct"].astype(float).clip(lower=0)
        if float(pct.sum()) > 0:
            base = 0.35 + 4.0 * pct.to_numpy()
            return base ** float(behavior_config.popularity_weight)
    base = (
        rows["team"]
        .astype(str)
        .map(TEAM_POPULARITY_BIAS)
        .fillna(1.0)
        .astype(float)
        .to_numpy()
    )
    return base ** float(behavior_config.popularity_weight)


def _future_value_penalty(
    rows: pd.DataFrame,
    all_probabilities: pd.DataFrame,
    *,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    future_awareness = float(behavior_config.future_awareness)
    if future_awareness <= 0:
        return np.ones(len(rows), dtype=float)
    current_week = int(rows["week"].astype(int).min())
    future = all_probabilities[
        all_probabilities["week"].astype(int) > int(current_week)
    ].copy()
    if future.empty:
        return np.ones(len(rows), dtype=float)
    future_best = (
        future.groupby("team")["win_probability"]
        .max()
        .astype(float)
        .to_dict()
    )
    current_probabilities = rows["win_probability"].astype(float).to_numpy()
    future_values = rows["team"].astype(str).map(future_best).fillna(0.0).to_numpy()
    future_edge = np.maximum(future_values - current_probabilities, 0.0)
    scarcity_multiplier = 2.0 + max(float(behavior_config.scarcity_weight) - 1.0, 0.0)
    return 1.0 / (1.0 + future_awareness * scarcity_multiplier * future_edge)


def _scarcity_index(
    *,
    expected_remaining: float,
    average_available_teams: float,
    team_count: int,
    top_ownership: float,
    chalk_concentration: float,
) -> float:
    if team_count <= 0:
        return 0.0
    available_pressure = 1.0 - min(max(average_available_teams / team_count, 0.0), 1.0)
    chalk_pressure = min(max((top_ownership + chalk_concentration / 3.0) / 1.25, 0.0), 1.0)
    field_pressure = 0.0 if expected_remaining <= 0 else min(1.0, 1.0 / np.sqrt(expected_remaining))
    return float(min(1.0, 0.55 * available_pressure + 0.40 * chalk_pressure + 0.05 * field_pressure))


def _apply_ownership_shape(
    weights: np.ndarray,
    behavior_config: PublicBehaviorConfig,
) -> np.ndarray:
    shaped = np.asarray(weights, dtype=float)
    if shaped.size == 0:
        return shaped
    shaped = np.where(np.isfinite(shaped) & (shaped > 0), shaped, 0.0)
    total = float(shaped.sum())
    probabilities = (
        shaped / total
        if total > 0
        else np.full(shaped.size, 1.0 / float(shaped.size), dtype=float)
    )

    temperature = float(behavior_config.ownership_temperature)
    if temperature != 1.0:
        probabilities = probabilities ** (1.0 / temperature)
        temperature_total = float(probabilities.sum())
        probabilities = (
            probabilities / temperature_total
            if temperature_total > 0
            else np.full(shaped.size, 1.0 / float(shaped.size), dtype=float)
        )

    cap = float(behavior_config.max_single_team_ownership)
    if cap < 1.0:
        probabilities = _cap_probabilities(probabilities, cap)
    return probabilities


def _cap_probabilities(probabilities: np.ndarray, cap: float) -> np.ndarray:
    if probabilities.size == 0:
        return probabilities
    effective_cap = max(float(cap), 1.0 / float(probabilities.size))
    capped = np.asarray(probabilities, dtype=float).copy()
    for _ in range(probabilities.size + 1):
        over = capped > effective_cap
        if not bool(over.any()):
            break
        excess = float((capped[over] - effective_cap).sum())
        capped[over] = effective_cap
        under = ~over
        if not bool(under.any()) or excess <= 0:
            break
        capacity = np.maximum(effective_cap - capped[under], 0.0)
        capacity_total = float(capacity.sum())
        if capacity_total <= 0:
            break
        capped[under] = capped[under] + excess * capacity / capacity_total

    total = float(capped.sum())
    return (
        capped / total
        if total > 0
        else np.full(probabilities.size, 1.0 / float(probabilities.size), dtype=float)
    )


def _week_alternative_scarcity(win_probabilities: pd.Series) -> float:
    probabilities = pd.to_numeric(win_probabilities, errors="coerce").dropna().astype(float)
    if probabilities.empty:
        return 0.0
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
    return float(max(0.0, min(1.0, scarcity_score / 5.50)))


def _prepare_team_probabilities(
    *,
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    projected_team_strength: pd.DataFrame | None,
    team_probabilities_df: pd.DataFrame | None,
    behavior_config: PublicBehaviorConfig,
) -> pd.DataFrame:
    if team_probabilities_df is not None:
        return _normalize_team_probabilities(_as_dataframe(team_probabilities_df))

    from survivor.path_ev import build_forward_game_probabilities

    return _normalize_team_probabilities(
        build_forward_game_probabilities(
            schedule_df=schedule_df,
            odds_df=odds_df,
            public_picks_df=public_picks_df,
            team_strength_df=projected_team_strength,
            public_behavior_config=behavior_config,
        ),
    )


def _normalize_team_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    probabilities = df.copy()
    required = {"week", "game_id", "team", "opponent", "win_probability"}
    missing = required - set(probabilities.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"team probabilities are missing required columns: {missing_text}")
    probabilities["week"] = pd.to_numeric(probabilities["week"], errors="raise").astype(int)
    probabilities["team"] = probabilities["team"].astype(str)
    probabilities["opponent"] = probabilities["opponent"].astype(str)
    probabilities["game_id"] = probabilities["game_id"].astype(str)
    probabilities["win_probability"] = (
        pd.to_numeric(probabilities["win_probability"], errors="raise")
        .astype(float)
        .clip(lower=0.001, upper=0.999)
    )
    if "projected_public_pick_pct" not in probabilities.columns:
        probabilities["projected_public_pick_pct"] = 0.0
    probabilities["projected_public_pick_pct"] = (
        pd.to_numeric(probabilities["projected_public_pick_pct"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    return probabilities.sort_values(["week", "game_id", "team"]).reset_index(drop=True)


def _resolve_weeks(
    probabilities: pd.DataFrame,
    start_week: int,
    weeks: list[int] | tuple[int, ...] | None,
) -> list[int]:
    available = sorted(probabilities["week"].astype(int).unique().tolist())
    if weeks is None:
        return [week for week in available if int(week) >= int(start_week)]
    requested = {int(week) for week in weeks if int(week) >= int(start_week)}
    return [week for week in available if week in requested]


def _resolve_sample_entry_count(
    pool_size: int,
    public_field_sample_size: int | None,
) -> int:
    if pool_size <= 0:
        return 0
    if public_field_sample_size is None:
        return int(pool_size)
    return int(min(pool_size, public_field_sample_size))


def _resolve_behavior_config(
    behavior_config: PublicBehaviorConfig | str | None,
    *,
    chalkiness: float,
    future_awareness: float,
    randomness: float,
) -> PublicBehaviorConfig:
    if behavior_config is not None:
        return resolve_public_behavior_config(behavior_config)
    return public_behavior_from_legacy(
        chalkiness=chalkiness,
        future_awareness=future_awareness,
        randomness=randomness,
    )


def _mean_or_zero(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _empty_public_field_result(
    *,
    pool_size: int,
    simulations: int,
    weeks: list[int] | None = None,
    teams: list[str] | None = None,
) -> PublicFieldSimulationResult:
    resolved_weeks = [int(week) for week in weeks or []]
    resolved_teams = [str(team) for team in teams or []]
    team_to_code = {team: index for index, team in enumerate(resolved_teams)}
    empty_pick = np.empty((int(simulations), 0, len(resolved_weeks)), dtype=np.int16)
    empty_win = np.empty(
        (len(resolved_weeks), int(simulations), len(resolved_teams)),
        dtype=bool,
    )
    return PublicFieldSimulationResult(
        entry_paths=pd.DataFrame(),
        week_by_week_public_ownership=pd.DataFrame(),
        remaining_field_distribution=pd.DataFrame(),
        team_usage_exhaustion=pd.DataFrame(),
        scarcity_by_week=pd.DataFrame(),
        path_clusters=pd.DataFrame(),
        path_convergence=pd.DataFrame(),
        diagnostics={
            "model": PUBLIC_FIELD_PICK_SOURCE,
            "pool_size": int(pool_size),
            "simulations": int(simulations),
            "sample_entry_count": 0,
            "entry_weight": 0.0,
            "clustering_strength": 0.0,
            "elite_path_bias": 1.0,
            "late_season_overlap_weight": 2.0,
            "path_convergence_temperature": 1.0,
            "late_season_convergence_score": 0.0,
        },
        weeks=resolved_weeks,
        teams=resolved_teams,
        team_to_code=team_to_code,
        game_to_code={},
        pick_matrix=empty_pick,
        pick_game_matrix=empty_pick.copy(),
        pick_win_matrix=np.empty((int(simulations), 0, len(resolved_weeks)), dtype=bool),
        win_matrix=empty_win,
        entry_weight=0.0,
        pool_size=int(pool_size),
        sample_entry_count=0,
        simulations=int(simulations),
    )


def _validate_positive_int(name: str, value: int, allow_zero: bool = False) -> None:
    numeric = int(value)
    if allow_zero and numeric == 0:
        return
    if numeric <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_behavior_parameters(
    *,
    chalkiness: float,
    future_awareness: float,
    randomness: float,
) -> None:
    if float(chalkiness) <= 0:
        raise ValueError("chalkiness must be positive.")
    if float(future_awareness) < 0:
        raise ValueError("future_awareness must be non-negative.")
    if not 0 <= float(randomness) <= 1:
        raise ValueError("randomness must be between 0 and 1.")


def _as_dataframe(data: pd.DataFrame | list[dict[str, object]] | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
