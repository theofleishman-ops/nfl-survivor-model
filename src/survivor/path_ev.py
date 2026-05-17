"""Single-entry survivor path expected-value optimizer.

This module keeps the existing weekly heuristic ranking in the loop only as a
candidate generator.  The objective used to choose a path is expected contest
equity: average final payout share across simulated seasons for one entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from survivor.odds import add_no_vig_probabilities
from survivor.optimizer import rank_weekly_picks
from survivor.public_field import (
    DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE,
    PUBLIC_FIELD_PICK_SOURCE,
    PublicFieldSimulationResult,
    simulate_public_field_paths,
)
from survivor.team_strength import build_team_strength_priors
from survivor.win_probability import (
    DEFAULT_TEAM_STRENGTH_SCALE,
    estimate_game_win_probability,
    estimate_team_strength_win_probability,
)


DEFAULT_POOL_SIZE = 5000
DEFAULT_TOP_K = 5
DEFAULT_BEAM_WIDTH = 100
DEFAULT_SIMULATIONS = 10000
DEFAULT_HOME_WIN_PROBABILITY = 0.52
DEFAULT_PATH_EV_HORIZON = "available"
DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT = 0.0
DEFAULT_FORWARD_TEAM_STRENGTH_SCALE = DEFAULT_TEAM_STRENGTH_SCALE
DEFAULT_USE_PUBLIC_FIELD_SIMULATION = True
UNRELIABLE_FALLBACK_COVERAGE_THRESHOLD = 0.25
PATH_EV_HORIZONS = ("week", "current", "available", "full-season")
PROBABILITY_SOURCE_REAL_MONEYLINE = "real_moneyline"
PROBABILITY_SOURCE_REAL_SPREAD = "real_spread"
PROBABILITY_SOURCE_PROJECTED_TEAM_STRENGTH = "projected_team_strength"
PROBABILITY_SOURCE_FALLBACK_DEFAULT = "fallback_default"
REAL_PROBABILITY_SOURCES = frozenset(
    {
        PROBABILITY_SOURCE_REAL_MONEYLINE,
        PROBABILITY_SOURCE_REAL_SPREAD,
    },
)
PROJECTED_PROBABILITY_SOURCES = frozenset(
    {PROBABILITY_SOURCE_PROJECTED_TEAM_STRENGTH},
)
FALLBACK_PROBABILITY_SOURCES = frozenset({PROBABILITY_SOURCE_FALLBACK_DEFAULT})
PROJECTED_PUBLIC_PICK_SOURCE = "projected_ownership"

TEAM_POPULARITY_PLACEHOLDER = {
    "DAL": 1.35,
    "KC": 1.25,
    "GB": 1.18,
    "PIT": 1.18,
    "SF": 1.18,
    "PHI": 1.16,
    "BUF": 1.14,
    "CHI": 1.10,
    "LV": 1.10,
    "NYG": 1.10,
    "NYJ": 1.10,
    "LAR": 1.08,
    "MIA": 1.08,
    "NE": 1.08,
    "DET": 1.06,
    "BAL": 1.05,
    "CIN": 1.05,
    "MIN": 1.05,
    "SEA": 1.05,
}

PATH_COLUMNS = [
    "path_id",
    "path_rank",
    "path_key",
    "week",
    "team",
    "opponent",
    "game_id",
    "win_probability",
    "probability_source",
    "projected_public_pick_pct",
    "public_pick_source",
    "heuristic_rank",
    "heuristic_final_score",
    "cumulative_survival_probability",
    "estimated_path_ev",
]

ASSUMPTIONS = (
    "Expected value is average contest equity for one survivor entry.",
    "Contest equity is modeled as 1 / total survivors when the path is alive, otherwise 0.",
    "Path EV is estimated as path survival probability times expected equity conditional on the path surviving.",
    "The selected entry is removed from public ownership counts for its fixed pick in each week.",
    "Winner-take-all or equal split among final survivors is assumed.",
    "Current public pick percentages are used when present.",
    "Future public ownership is simulated as public-entry paths with used-team constraints when multiple weeks are evaluated.",
    "Public entries prefer win probability and chalk, weakly account for future scarcity, and occasionally choose contrarian paths.",
    "Future win probabilities use no-vig moneyline first, then spread, then team-strength ratings with opponent strength and configurable home field, then a conservative default.",
    "Single-week evaluations use exact current-week ownership because used-team constraints have no future week to affect.",
)


@dataclass(frozen=True)
class PathEVResult:
    """Monte Carlo evaluation for one fixed survivor path."""

    path: pd.DataFrame
    path_ev: float
    expected_contest_equity: float
    path_survival_probability: float
    expected_final_field_size: float
    expected_final_public_entries: float
    expected_survivors_if_alive: float
    expected_equity_if_alive: float
    baseline_contest_equity: float
    entry_fee: float | None
    prize_pool: float | None
    entry_cost: float | None
    baseline_value: float | None
    ev_dollars: float | None
    ev_multiple: float | None
    expected_edge: float | None
    ev_multiple_vs_entry_fee: float | None
    ev_multiple_vs_baseline: float | None
    expected_edge_vs_entry_fee: float | None
    expected_edge_vs_baseline: float | None
    survival_probability_by_week: pd.DataFrame
    expected_field_size_by_week: pd.DataFrame
    simulation_summary: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SingleEntryPathOptimizationResult:
    """Full optimizer output for one survivor entry."""

    best_path: pd.DataFrame
    best_path_ev: float
    expected_contest_equity: float
    path_survival_probability: float
    expected_final_field_size: float
    expected_final_public_entries: float
    expected_survivors_if_alive: float
    expected_equity_if_alive: float
    baseline_contest_equity: float
    entry_fee: float | None
    prize_pool: float | None
    entry_cost: float | None
    baseline_value: float | None
    ev_dollars: float | None
    ev_multiple: float | None
    expected_edge: float | None
    ev_multiple_vs_entry_fee: float | None
    ev_multiple_vs_baseline: float | None
    expected_edge_vs_entry_fee: float | None
    expected_edge_vs_baseline: float | None
    survival_probability_by_week: pd.DataFrame
    expected_field_size_by_week: pd.DataFrame
    top_paths: pd.DataFrame
    heuristic_comparison: pd.DataFrame
    candidate_paths: pd.DataFrame
    evaluated_paths: pd.DataFrame
    assumptions: tuple[str, ...] = ASSUMPTIONS
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _BeamPath:
    steps: tuple[dict[str, Any], ...]
    used_teams: frozenset[str]
    cumulative_survival_probability: float
    expected_public_entries: float
    estimated_path_ev: float


@dataclass(frozen=True)
class _PathModelData:
    schedule: pd.DataFrame
    team_probabilities: pd.DataFrame
    public_picks: pd.DataFrame
    rankings_by_week: dict[int, pd.DataFrame]
    weeks: list[int]


@dataclass(frozen=True)
class _CommonSimulation:
    weeks: list[int]
    team_wins: dict[tuple[int, str], np.ndarray]
    public_distributions: dict[int, pd.DataFrame]
    public_field: PublicFieldSimulationResult | None = None


def build_forward_game_probabilities(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame | None = None,
    team_strength_df: pd.DataFrame | None = None,
    team_strength_home_field_adjustment: float = DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    team_strength_scale: float = DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
) -> pd.DataFrame:
    """Build team-level win and ownership projections for every scheduled game.

    Probability sources are tagged as:
    ``real_moneyline``, ``real_spread``, ``projected_team_strength``, or
    ``fallback_default``.
    """
    schedule = _prepare_schedule(schedule_df)
    team_probabilities = _prepare_team_probabilities(
        schedule,
        _as_dataframe(odds_df),
        _as_dataframe(team_strength_df),
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    return _add_projected_public_picks(
        team_probabilities,
        _normalize_public_picks(_as_dataframe(public_picks_df)),
    )


def generate_candidate_paths(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    start_week: int,
    used_teams: set[str] | list[str] | tuple[str, ...] | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_paths: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    path_ev_horizon: str = DEFAULT_PATH_EV_HORIZON,
    team_strength_df: pd.DataFrame | None = None,
    team_strength_home_field_adjustment: float = DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    team_strength_scale: float = DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
) -> pd.DataFrame:
    """Generate fixed survivor paths with a controlled beam search.

    The weekly ranking heuristic is used only to order candidate expansions.
    The ``estimated_path_ev`` column is a cheap beam-pruning proxy, not the true
    Monte Carlo objective used by :func:`optimize_single_entry_path`.
    """
    _validate_positive_int("pool_size", pool_size)
    _validate_positive_int("top_k", top_k)
    _validate_positive_int("beam_width", beam_width)
    if max_paths is not None:
        _validate_positive_int("max_paths", max_paths)

    model = _prepare_path_model_data(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        start_week=start_week,
        pool_size=pool_size,
        team_strength_df=team_strength_df,
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    weeks = _resolve_horizon_weeks(model, start_week, path_ev_horizon)
    if not weeks:
        return _empty_paths_frame()

    normalized_used = frozenset(str(team).strip() for team in used_teams or () if str(team).strip())
    beam = [
        _BeamPath(
            steps=(),
            used_teams=normalized_used,
            cumulative_survival_probability=1.0,
            expected_public_entries=float(max(pool_size - 1, 0)),
            estimated_path_ev=1.0 / max(float(pool_size), 1.0),
        ),
    ]

    public_survival_rates = _expected_public_survival_rates(model.team_probabilities)
    for week in weeks:
        ranked = model.rankings_by_week[week]
        field_survival_rate = public_survival_rates.get(week, 0.5)
        expanded: list[_BeamPath] = []

        for path in beam:
            available = ranked[~ranked["team"].astype(str).isin(path.used_teams)].head(top_k)
            for candidate in available.to_dict("records"):
                probability = float(candidate["win_probability"])
                cumulative_survival = path.cumulative_survival_probability * probability
                expected_public = path.expected_public_entries * field_survival_rate
                estimated_ev = cumulative_survival / max(
                    expected_public + cumulative_survival,
                    1.0,
                )
                step = {
                    "week": int(week),
                    "team": str(candidate["team"]),
                    "opponent": str(candidate.get("opponent", "")),
                    "game_id": candidate.get("game_id"),
                    "win_probability": probability,
                    "probability_source": candidate.get("probability_source", "unknown"),
                    "projected_public_pick_pct": float(
                        candidate.get("projected_public_pick_pct", 0.0),
                    ),
                    "public_pick_source": candidate.get("public_pick_source", "unknown"),
                    "heuristic_rank": _int_or_nan(candidate.get("rank")),
                    "heuristic_final_score": _float_or_nan(candidate.get("final_score")),
                    "cumulative_survival_probability": cumulative_survival,
                    "estimated_path_ev": estimated_ev,
                }
                expanded.append(
                    _BeamPath(
                        steps=path.steps + (step,),
                        used_teams=path.used_teams | {str(candidate["team"])},
                        cumulative_survival_probability=cumulative_survival,
                        expected_public_entries=expected_public,
                        estimated_path_ev=estimated_ev,
                    ),
                )

        if not expanded:
            break

        expanded.sort(
            key=lambda item: (
                item.estimated_path_ev,
                item.cumulative_survival_probability,
                len(item.steps),
            ),
            reverse=True,
        )
        beam = expanded[:beam_width]

    limit = max_paths if max_paths is not None else len(beam)
    return _paths_frame(beam[:limit])


def evaluate_path_ev(
    path_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    pool_size: int = DEFAULT_POOL_SIZE,
    simulations: int = DEFAULT_SIMULATIONS,
    random_seed: int | None = None,
    entry_fee: float | None = None,
    prize_pool: float | None = None,
    team_strength_df: pd.DataFrame | None = None,
    team_strength_home_field_adjustment: float = DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    team_strength_scale: float = DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
    use_public_field_simulation: bool = DEFAULT_USE_PUBLIC_FIELD_SIMULATION,
    public_field_sample_size: int | None = DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE,
) -> PathEVResult:
    """Evaluate a single fixed path using Monte Carlo contest equity."""
    _validate_positive_int("pool_size", pool_size)
    _validate_positive_int("simulations", simulations)
    _validate_non_negative_optional("entry_fee", entry_fee)
    _validate_non_negative_optional("prize_pool", prize_pool)
    _validate_positive_optional("public_field_sample_size", public_field_sample_size)

    path = _as_dataframe(path_df)
    if path.empty:
        raise ValueError("path_df must contain at least one path row.")

    path_ids = path["path_id"].dropna().unique().tolist() if "path_id" in path.columns else []
    if len(path_ids) > 1:
        raise ValueError("evaluate_path_ev expects one path. Pass one path_id at a time.")

    start_week = int(pd.to_numeric(path["week"], errors="raise").min())
    model = _prepare_path_model_data(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        start_week=start_week,
        pool_size=pool_size,
        team_strength_df=team_strength_df,
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    prepared_path = _prepare_paths_for_evaluation(path, model.team_probabilities)
    weeks = sorted(prepared_path["week"].astype(int).unique().tolist())
    simulation = _simulate_common_seasons(
        model=model,
        weeks=weeks,
        pool_size=pool_size,
        simulations=simulations,
        random_seed=random_seed,
        use_public_field_simulation=use_public_field_simulation,
        public_field_sample_size=public_field_sample_size,
    )
    evaluated, survival_by_week, field_by_week = _evaluate_prepared_paths(
        paths=prepared_path,
        simulation=simulation,
        simulations=simulations,
        pool_size=pool_size,
        entry_fee=entry_fee,
        prize_pool=prize_pool,
    )
    summary = evaluated.iloc[0].to_dict()
    evaluated_path = prepared_path.copy()
    evaluated_path["path_ev"] = float(summary["path_ev"])
    path_field_by_week = field_by_week[
        field_by_week["path_id"].astype(int) == int(summary["path_id"])
    ].copy()
    field_with_path = path_field_by_week.merge(
        survival_by_week[
            ["week", "path_survival_probability"]
        ],
        on="week",
        how="left",
    ).drop(columns=["path_id"], errors="ignore")
    diagnostics = _build_path_diagnostics(
        path=prepared_path,
        model=model,
        horizon="explicit-path",
        start_week=start_week,
        pool_size=pool_size,
        entry_fee=entry_fee,
        prize_pool=prize_pool,
        simulations=simulations,
        public_field=simulation.public_field,
        path_field_by_week=field_with_path,
    )

    return PathEVResult(
        path=evaluated_path.reset_index(drop=True),
        path_ev=float(summary["path_ev"]),
        expected_contest_equity=float(summary["expected_contest_equity"]),
        path_survival_probability=float(summary["path_survival_probability"]),
        expected_final_field_size=float(summary["expected_final_field_size"]),
        expected_final_public_entries=float(summary["expected_final_public_entries"]),
        expected_survivors_if_alive=float(summary["expected_survivors_if_alive"]),
        expected_equity_if_alive=float(summary["expected_equity_if_alive"]),
        baseline_contest_equity=float(summary["baseline_contest_equity"]),
        entry_fee=_optional_float(summary["entry_fee"]),
        prize_pool=_optional_float(summary["prize_pool"]),
        entry_cost=_optional_float(summary["entry_cost"]),
        baseline_value=_optional_float(summary["baseline_value"]),
        ev_dollars=_optional_float(summary["ev_dollars"]),
        ev_multiple=_optional_float(summary["ev_multiple"]),
        expected_edge=_optional_float(summary["expected_edge"]),
        ev_multiple_vs_entry_fee=_optional_float(summary["ev_multiple_vs_entry_fee"]),
        ev_multiple_vs_baseline=_optional_float(summary["ev_multiple_vs_baseline"]),
        expected_edge_vs_entry_fee=_optional_float(summary["expected_edge_vs_entry_fee"]),
        expected_edge_vs_baseline=_optional_float(summary["expected_edge_vs_baseline"]),
        survival_probability_by_week=survival_by_week.drop(columns=["path_id"], errors="ignore"),
        expected_field_size_by_week=field_with_path,
        simulation_summary={
            "simulations": int(simulations),
            "pool_size": int(pool_size),
            "start_week": int(min(weeks)),
            "end_week": int(max(weeks)),
        },
        diagnostics=diagnostics,
    )


def optimize_single_entry_path(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    start_week: int,
    used_teams: set[str] | list[str] | tuple[str, ...] | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_paths: int | None = None,
    simulations: int = DEFAULT_SIMULATIONS,
    random_seed: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    entry_fee: float | None = None,
    prize_pool: float | None = None,
    path_ev_horizon: str = DEFAULT_PATH_EV_HORIZON,
    team_strength_df: pd.DataFrame | None = None,
    team_strength_home_field_adjustment: float = DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    team_strength_scale: float = DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
    use_public_field_simulation: bool = DEFAULT_USE_PUBLIC_FIELD_SIMULATION,
    public_field_sample_size: int | None = DEFAULT_PUBLIC_FIELD_SAMPLE_SIZE,
) -> SingleEntryPathOptimizationResult:
    """Find the single-entry path with the highest simulated contest equity."""
    _validate_positive_int("simulations", simulations)
    _validate_non_negative_optional("entry_fee", entry_fee)
    _validate_non_negative_optional("prize_pool", prize_pool)
    _validate_positive_optional("public_field_sample_size", public_field_sample_size)
    candidate_paths = generate_candidate_paths(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        start_week=start_week,
        used_teams=used_teams,
        pool_size=pool_size,
        max_paths=max_paths,
        top_k=top_k,
        beam_width=beam_width,
        path_ev_horizon=path_ev_horizon,
        team_strength_df=team_strength_df,
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    if candidate_paths.empty:
        raise ValueError("No candidate paths were generated.")

    model = _prepare_path_model_data(
        schedule_df=schedule_df,
        odds_df=odds_df,
        public_picks_df=public_picks_df,
        start_week=start_week,
        pool_size=pool_size,
        team_strength_df=team_strength_df,
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    prepared_paths = _prepare_paths_for_evaluation(candidate_paths, model.team_probabilities)
    weeks = sorted(prepared_paths["week"].astype(int).unique().tolist())
    horizon = _normalize_path_ev_horizon(path_ev_horizon)
    simulation = _simulate_common_seasons(
        model=model,
        weeks=weeks,
        pool_size=pool_size,
        simulations=simulations,
        random_seed=random_seed,
        use_public_field_simulation=use_public_field_simulation,
        public_field_sample_size=public_field_sample_size,
    )
    evaluated_paths, survival_by_week, field_by_week = _evaluate_prepared_paths(
        paths=prepared_paths,
        simulation=simulation,
        simulations=simulations,
        pool_size=pool_size,
        entry_fee=entry_fee,
        prize_pool=prize_pool,
    )
    evaluated_paths = evaluated_paths.sort_values(
        ["path_ev", "path_survival_probability", "estimated_path_ev"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    evaluated_paths["path_ev_rank"] = range(1, len(evaluated_paths) + 1)

    best = evaluated_paths.iloc[0]
    best_path_id = int(best["path_id"])
    best_path = prepared_paths[prepared_paths["path_id"].astype(int) == best_path_id].copy()
    best_path["path_ev"] = float(best["path_ev"])
    best_path["path_ev_rank"] = 1
    best_survival = survival_by_week[survival_by_week["path_id"].astype(int) == best_path_id].copy()
    best_field_by_week = field_by_week[
        field_by_week["path_id"].astype(int) == best_path_id
    ].copy()
    best_field = best_field_by_week.merge(
        best_survival[["week", "path_survival_probability"]],
        on="week",
        how="left",
    ).drop(columns=["path_id"], errors="ignore")
    heuristic_comparison = _build_heuristic_comparison(
        start_week=start_week,
        rankings=model.rankings_by_week.get(int(start_week), pd.DataFrame()),
        evaluated_paths=evaluated_paths,
        prepared_paths=prepared_paths,
        best_path_id=best_path_id,
    )
    diagnostics = _build_path_diagnostics(
        path=best_path,
        model=model,
        horizon=horizon,
        start_week=start_week,
        pool_size=pool_size,
        entry_fee=entry_fee,
        prize_pool=prize_pool,
        simulations=simulations,
        public_field=simulation.public_field,
        path_field_by_week=best_field,
    )
    diagnostics["horizon_comparison"] = _build_horizon_comparison(
        path=best_path,
        model=model,
        selected_horizon=horizon,
        start_week=start_week,
        pool_size=pool_size,
        simulations=simulations,
        random_seed=random_seed,
        entry_fee=entry_fee,
        prize_pool=prize_pool,
        use_public_field_simulation=use_public_field_simulation,
        public_field_sample_size=public_field_sample_size,
        primary_summary=best.to_dict(),
    )

    return SingleEntryPathOptimizationResult(
        best_path=best_path.reset_index(drop=True),
        best_path_ev=float(best["path_ev"]),
        expected_contest_equity=float(best["expected_contest_equity"]),
        path_survival_probability=float(best["path_survival_probability"]),
        expected_final_field_size=float(best["expected_final_field_size"]),
        expected_final_public_entries=float(best["expected_final_public_entries"]),
        expected_survivors_if_alive=float(best["expected_survivors_if_alive"]),
        expected_equity_if_alive=float(best["expected_equity_if_alive"]),
        baseline_contest_equity=float(best["baseline_contest_equity"]),
        entry_fee=_optional_float(best["entry_fee"]),
        prize_pool=_optional_float(best["prize_pool"]),
        entry_cost=_optional_float(best["entry_cost"]),
        baseline_value=_optional_float(best["baseline_value"]),
        ev_dollars=_optional_float(best["ev_dollars"]),
        ev_multiple=_optional_float(best["ev_multiple"]),
        expected_edge=_optional_float(best["expected_edge"]),
        ev_multiple_vs_entry_fee=_optional_float(best["ev_multiple_vs_entry_fee"]),
        ev_multiple_vs_baseline=_optional_float(best["ev_multiple_vs_baseline"]),
        expected_edge_vs_entry_fee=_optional_float(best["expected_edge_vs_entry_fee"]),
        expected_edge_vs_baseline=_optional_float(best["expected_edge_vs_baseline"]),
        survival_probability_by_week=best_survival.drop(columns=["path_id"], errors="ignore").reset_index(drop=True),
        expected_field_size_by_week=best_field.reset_index(drop=True),
        top_paths=evaluated_paths,
        heuristic_comparison=heuristic_comparison,
        candidate_paths=candidate_paths,
        evaluated_paths=evaluated_paths,
        diagnostics=diagnostics,
    )


def _prepare_path_model_data(
    *,
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    start_week: int,
    pool_size: int,
    team_strength_df: pd.DataFrame | None = None,
    team_strength_home_field_adjustment: float = DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    team_strength_scale: float = DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
) -> _PathModelData:
    schedule = _prepare_schedule(schedule_df)
    team_probabilities = _prepare_team_probabilities(
        schedule,
        _as_dataframe(odds_df),
        _as_dataframe(team_strength_df),
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    team_probabilities = _add_projected_public_picks(
        team_probabilities,
        _normalize_public_picks(_as_dataframe(public_picks_df)),
    )
    public_picks = team_probabilities[
        ["week", "team", "projected_public_pick_pct"]
    ].rename(columns={"projected_public_pick_pct": "public_pick_pct"})
    weeks = [
        int(week)
        for week in sorted(schedule["week"].astype(int).unique().tolist())
        if int(week) >= int(start_week)
    ]
    available_weeks = set(team_probabilities["week"].astype(int).unique().tolist())
    weeks = [week for week in weeks if week in available_weeks]
    rankings_by_week = _rankings_by_week(
        weeks=weeks,
        schedule=schedule,
        team_probabilities=team_probabilities,
        public_picks=public_picks,
        pool_size=pool_size,
    )
    return _PathModelData(
        schedule=schedule,
        team_probabilities=team_probabilities,
        public_picks=public_picks,
        rankings_by_week=rankings_by_week,
        weeks=weeks,
    )


def _normalize_path_ev_horizon(value: str) -> str:
    horizon = str(value).strip().lower().replace("_", "-")
    if horizon == "current":
        return "week"
    if horizon in {"full", "season", "fullseason"}:
        return "full-season"
    if horizon not in {"week", "available", "full-season", "explicit-path"}:
        choices = ", ".join(PATH_EV_HORIZONS)
        raise ValueError(f"path_ev_horizon must be one of: {choices}.")
    return horizon


def _resolve_horizon_weeks(
    model: _PathModelData,
    start_week: int,
    path_ev_horizon: str,
) -> list[int]:
    horizon = _normalize_path_ev_horizon(path_ev_horizon)
    eligible_weeks = [
        int(week)
        for week in model.weeks
        if int(week) >= int(start_week)
    ]
    if horizon == "full-season":
        return eligible_weeks
    if horizon == "week":
        return eligible_weeks[:1]
    if horizon == "available":
        available = _available_odds_weeks(model, start_week)
        return available or eligible_weeks[:1]
    return eligible_weeks


def _available_odds_weeks(model: _PathModelData, start_week: int) -> list[int]:
    weeks: list[int] = []
    for week in model.weeks:
        if int(week) < int(start_week):
            continue
        if not _week_has_real_odds(model.team_probabilities, int(week)):
            break
        weeks.append(int(week))
    return weeks


def _week_has_real_odds(team_probabilities: pd.DataFrame, week: int) -> bool:
    week_rows = team_probabilities[
        team_probabilities["week"].astype(int) == int(week)
    ].copy()
    if week_rows.empty:
        return False
    sources = week_rows["probability_source"].astype(str).str.lower()
    return bool(sources.isin(REAL_PROBABILITY_SOURCES).all())


def _build_path_diagnostics(
    *,
    path: pd.DataFrame,
    model: _PathModelData,
    horizon: str,
    start_week: int,
    pool_size: int,
    entry_fee: float | None,
    prize_pool: float | None,
    simulations: int,
    public_field: PublicFieldSimulationResult | None = None,
    path_field_by_week: pd.DataFrame | None = None,
) -> dict[str, Any]:
    path_df = path.sort_values("week").reset_index(drop=True).copy()
    weeks = [int(week) for week in path_df["week"].astype(int).tolist()]
    sources = path_df.get("probability_source", pd.Series(dtype=object)).astype(str).str.lower()
    real_mask = sources.isin(REAL_PROBABILITY_SOURCES)
    projected_mask = sources.isin(PROJECTED_PROBABILITY_SOURCES)
    fallback_mask = sources.isin(FALLBACK_PROBABILITY_SOURCES)
    fallback_rows = path_df[fallback_mask].copy()
    projected_rows = path_df[projected_mask].copy()
    fallback_coverage = _fallback_coverage(path_df)
    reliability_label = _full_season_reliability_label(horizon, fallback_coverage)
    ownership_methods = _value_counts_dict(path_df, "public_pick_source")
    baseline_fair_value = (
        float(prize_pool) / float(pool_size)
        if prize_pool is not None and float(pool_size) > 0
        else None
    )
    diagnostics = {
        "horizon": horizon,
        "number_of_weeks_evaluated": len(weeks),
        "weeks_evaluated": weeks,
        "weeks_with_real_odds": sorted(
            path_df.loc[real_mask, "week"].astype(int).unique().tolist(),
        ),
        "weeks_with_projected_odds": sorted(
            path_df.loc[projected_mask, "week"].astype(int).unique().tolist(),
        ),
        "weeks_with_projected_team_strength_probabilities": sorted(
            path_df.loc[projected_mask, "week"].astype(int).unique().tolist(),
        ),
        "weeks_using_fallback_probabilities": sorted(
            fallback_rows["week"].astype(int).unique().tolist(),
        ),
        "fallback_probability_count": int(fallback_mask.sum()),
        "fallback_coverage_pct": fallback_coverage,
        "full_season_ev_reliability": reliability_label,
        "full_season_ev_actionability": _full_season_actionability(
            horizon,
            fallback_coverage,
        ),
        "average_projected_win_probability": (
            float(projected_rows["win_probability"].astype(float).mean())
            if not projected_rows.empty
            else None
        ),
        "average_fallback_win_probability": (
            float(fallback_rows["win_probability"].astype(float).mean())
            if not fallback_rows.empty
            else None
        ),
        "probability_sources": _value_counts_dict(path_df, "probability_source"),
        "projected_probability_sources": _value_counts_dict(projected_rows, "probability_source"),
        "fallback_probability_sources": _value_counts_dict(fallback_rows, "probability_source"),
        "ownership_projection_methods": ownership_methods,
        "available_odds_weeks": _available_odds_weeks(model, start_week),
        "full_season_weeks": [
            int(week)
            for week in model.weeks
            if int(week) >= int(start_week)
        ],
        "baseline_fair_value": baseline_fair_value,
        "entry_fee": float(entry_fee) if entry_fee is not None else None,
        "prize_pool": float(prize_pool) if prize_pool is not None else None,
        "pool_size": int(pool_size),
        "simulations": int(simulations),
    }
    diagnostics.update(
        _public_field_diagnostics(
            public_field=public_field,
            path_field_by_week=path_field_by_week,
        ),
    )
    return diagnostics


def _public_field_diagnostics(
    *,
    public_field: PublicFieldSimulationResult | None,
    path_field_by_week: pd.DataFrame | None,
) -> dict[str, Any]:
    if public_field is None:
        return {
            "public_field_model": "independent_weekly_ownership",
            "public_field_path_simulation_enabled": False,
            "public_field_sample_size": None,
            "public_field_entry_weight": None,
            "expected_remaining_field_by_week": [],
            "expected_team_exhaustion": [],
            "projected_public_ownership_by_week": [],
            "public_field_scarcity_by_week": [],
            "scarcity_weeks": [],
            "path_uniqueness_score": None,
            "expected_public_overlap_entries": None,
            "expected_public_overlap_pct": None,
        }

    path_field = _as_dataframe(path_field_by_week)
    scarcity = public_field.scarcity_by_week.copy()
    ownership = public_field.week_by_week_public_ownership.copy()
    exhaustion = public_field.team_usage_exhaustion.copy()
    scarcity_weeks = (
        scarcity.loc[scarcity["scarcity_spike"].astype(bool), "week"]
        .astype(int)
        .tolist()
        if not scarcity.empty and "scarcity_spike" in scarcity.columns
        else []
    )
    path_uniqueness = (
        float(path_field["path_uniqueness_score"].astype(float).mean())
        if not path_field.empty and "path_uniqueness_score" in path_field.columns
        else None
    )
    overlap_entries = (
        float(path_field["expected_public_overlap_entries"].astype(float).mean())
        if not path_field.empty and "expected_public_overlap_entries" in path_field.columns
        else None
    )
    overlap_pct = (
        float(path_field["avg_path_overlap_pct"].astype(float).mean())
        if not path_field.empty and "avg_path_overlap_pct" in path_field.columns
        else None
    )

    return {
        "public_field_model": PUBLIC_FIELD_PICK_SOURCE,
        "public_field_path_simulation_enabled": True,
        "public_field_sample_size": int(public_field.sample_entry_count),
        "public_field_entry_weight": float(public_field.entry_weight),
        "expected_remaining_field_by_week": _records_for_columns(
            scarcity,
            [
                "week",
                "expected_remaining_entries",
                "average_available_teams",
                "max_projected_ownership_pct",
                "expected_chalk_concentration",
                "scarcity_index",
                "scarcity_spike",
            ],
        ),
        "expected_team_exhaustion": _top_records_by_week(
            exhaustion,
            sort_column="burned_pct",
            columns=[
                "week",
                "team",
                "expected_entries_burned_team",
                "burned_pct",
                "expected_remaining_entries_with_team_available",
                "remaining_field_available_pct",
            ],
            per_week=8,
        ),
        "projected_public_ownership_by_week": _top_records_by_week(
            ownership,
            sort_column="projected_ownership_pct",
            columns=[
                "week",
                "team",
                "expected_public_picks",
                "projected_ownership_pct",
            ],
            per_week=8,
        ),
        "public_field_scarcity_by_week": _records_for_columns(
            scarcity,
            [
                "week",
                "expected_remaining_entries",
                "average_available_teams",
                "max_projected_ownership_pct",
                "expected_chalk_concentration",
                "scarcity_index",
                "scarcity_spike",
            ],
        ),
        "scarcity_weeks": sorted(set(int(week) for week in scarcity_weeks)),
        "path_uniqueness_score": path_uniqueness,
        "expected_public_overlap_entries": overlap_entries,
        "expected_public_overlap_pct": overlap_pct,
    }


def _fallback_coverage(path_df: pd.DataFrame) -> float:
    if path_df.empty or "probability_source" not in path_df.columns:
        return 0.0
    sources = path_df["probability_source"].astype(str).str.lower()
    return float(sources.isin(FALLBACK_PROBABILITY_SOURCES).mean())


def _full_season_reliability_label(horizon: str, fallback_coverage: float) -> str:
    if horizon != "full-season":
        return "not_applicable"
    if fallback_coverage > UNRELIABLE_FALLBACK_COVERAGE_THRESHOLD:
        return "UNRELIABLE"
    return "USABLE"


def _full_season_actionability(horizon: str, fallback_coverage: float) -> str:
    if horizon != "full-season":
        return "not_applicable"
    if fallback_coverage > UNRELIABLE_FALLBACK_COVERAGE_THRESHOLD:
        return "not_actionable_high_fallback_coverage"
    return "usable_with_projection_caveats"


def _build_horizon_comparison(
    *,
    path: pd.DataFrame,
    model: _PathModelData,
    selected_horizon: str,
    start_week: int,
    pool_size: int,
    simulations: int,
    random_seed: int | None,
    entry_fee: float | None,
    prize_pool: float | None,
    use_public_field_simulation: bool,
    public_field_sample_size: int | None,
    primary_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    path_weeks = set(path["week"].astype(int).tolist())
    specs = [
        ("Current-week EV", "week", _resolve_horizon_weeks(model, start_week, "week")),
        (
            "Available-odds horizon EV",
            "available",
            _resolve_horizon_weeks(model, start_week, "available"),
        ),
        (
            "Full-season projected EV",
            "full-season",
            _resolve_horizon_weeks(model, start_week, "full-season"),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, horizon, weeks in specs:
        if not weeks:
            rows.append(_empty_horizon_row(label, horizon, weeks, "no weeks available"))
            continue
        if not set(weeks).issubset(path_weeks):
            note = (
                "not evaluated for selected path; run with "
                f"--path-ev-horizon {horizon} to optimize this horizon"
            )
            rows.append(_empty_horizon_row(label, horizon, weeks, note))
            continue
        if horizon == selected_horizon and set(weeks) == path_weeks:
            summary = primary_summary
        else:
            summary = _evaluate_path_for_weeks(
                path=path,
                model=model,
                weeks=weeks,
                pool_size=pool_size,
                simulations=simulations,
                random_seed=random_seed,
                entry_fee=entry_fee,
                prize_pool=prize_pool,
                use_public_field_simulation=use_public_field_simulation,
                public_field_sample_size=public_field_sample_size,
            )
        coverage = _fallback_coverage(path[path["week"].astype(int).isin(set(weeks))])
        rows.append(_horizon_row(label, horizon, weeks, summary, coverage))
    return rows


def _evaluate_path_for_weeks(
    *,
    path: pd.DataFrame,
    model: _PathModelData,
    weeks: list[int],
    pool_size: int,
    simulations: int,
    random_seed: int | None,
    entry_fee: float | None,
    prize_pool: float | None,
    use_public_field_simulation: bool,
    public_field_sample_size: int | None,
) -> dict[str, Any]:
    subset = path[path["week"].astype(int).isin(set(weeks))].copy()
    prepared = _prepare_paths_for_evaluation(subset, model.team_probabilities)
    simulation = _simulate_common_seasons(
        model=model,
        weeks=weeks,
        pool_size=pool_size,
        simulations=simulations,
        random_seed=random_seed,
        use_public_field_simulation=use_public_field_simulation,
        public_field_sample_size=public_field_sample_size,
    )
    evaluated, _, _ = _evaluate_prepared_paths(
        paths=prepared,
        simulation=simulation,
        simulations=simulations,
        pool_size=pool_size,
        entry_fee=entry_fee,
        prize_pool=prize_pool,
    )
    return evaluated.iloc[0].to_dict()


def _horizon_row(
    label: str,
    horizon: str,
    weeks: list[int],
    summary: dict[str, Any],
    fallback_coverage: float,
) -> dict[str, Any]:
    reliability = _full_season_reliability_label(horizon, fallback_coverage)
    note = (
        "not actionable because fallback coverage exceeds 25%"
        if reliability == "UNRELIABLE"
        else ""
    )
    return {
        "label": label,
        "horizon": horizon,
        "status": reliability if reliability == "UNRELIABLE" else "evaluated",
        "weeks_evaluated": len(weeks),
        "fallback_coverage_pct": fallback_coverage,
        "full_season_ev_reliability": reliability,
        "path_ev": _optional_float(summary.get("path_ev")),
        "ev_dollars": _optional_float(summary.get("ev_dollars")),
        "ev_multiple_vs_entry_fee": _optional_float(
            summary.get("ev_multiple_vs_entry_fee"),
        ),
        "ev_multiple_vs_baseline": _optional_float(
            summary.get("ev_multiple_vs_baseline"),
        ),
        "expected_edge_vs_baseline": _optional_float(
            summary.get("expected_edge_vs_baseline"),
        ),
        "path_survival_probability": _optional_float(
            summary.get("path_survival_probability"),
        ),
        "expected_survivors_if_alive": _optional_float(
            summary.get("expected_survivors_if_alive"),
        ),
        "note": note,
    }


def _empty_horizon_row(
    label: str,
    horizon: str,
    weeks: list[int],
    note: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "horizon": horizon,
        "status": "not evaluated",
        "weeks_evaluated": len(weeks),
        "fallback_coverage_pct": None,
        "full_season_ev_reliability": "not_evaluated",
        "path_ev": None,
        "ev_dollars": None,
        "ev_multiple_vs_entry_fee": None,
        "ev_multiple_vs_baseline": None,
        "expected_edge_vs_baseline": None,
        "path_survival_probability": None,
        "expected_survivors_if_alive": None,
        "note": note,
    }


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    values = df[column].dropna().astype(str)
    if values.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in values.value_counts().sort_index().items()
    }


def _records_for_columns(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    available = [column for column in columns if column in df.columns]
    if not available:
        return []
    return df[available].copy().to_dict("records")


def _top_records_by_week(
    df: pd.DataFrame,
    *,
    sort_column: str,
    columns: list[str],
    per_week: int,
) -> list[dict[str, Any]]:
    if df.empty or sort_column not in df.columns or "week" not in df.columns:
        return []
    frames: list[pd.DataFrame] = []
    for _, week_df in df.groupby("week", sort=True):
        frames.append(
            week_df.sort_values(sort_column, ascending=False).head(per_week),
        )
    if not frames:
        return []
    return _records_for_columns(pd.concat(frames, ignore_index=True), columns)


def _prepare_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule = _as_dataframe(schedule_df)
    required = {"week", "game_id", "home_team", "away_team"}
    missing = required - set(schedule.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"schedule data is missing required columns: {missing_text}")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="raise").astype(int)
    return schedule.sort_values(["week", "game_id"]).reset_index(drop=True)


def _prepare_team_probabilities(
    schedule: pd.DataFrame,
    odds_df: pd.DataFrame,
    team_strength_df: pd.DataFrame,
    *,
    team_strength_home_field_adjustment: float = DEFAULT_TEAM_STRENGTH_HOME_FIELD_ADJUSTMENT,
    team_strength_scale: float = DEFAULT_FORWARD_TEAM_STRENGTH_SCALE,
) -> pd.DataFrame:
    _validate_team_strength_parameters(
        team_strength_home_field_adjustment=team_strength_home_field_adjustment,
        team_strength_scale=team_strength_scale,
    )
    teams = _schedule_team_rows(schedule)
    h2h = _extract_h2h_probabilities(odds_df)
    spreads = _extract_spread_probabilities(odds_df)
    strength = _extract_team_strength(odds_df, team_strength_df)

    probabilities = teams.merge(
        h2h[["week", "game_id", "team", "h2h_probability"]],
        on=["week", "game_id", "team"],
        how="left",
    ).merge(
        spreads[["week", "game_id", "team", "spread_probability"]],
        on=["week", "game_id", "team"],
        how="left",
    )

    if not strength.empty:
        probabilities = probabilities.merge(
            strength.rename(columns={"team_strength_score": "team_strength_score"}),
            on="team",
            how="left",
        ).merge(
            strength.rename(
                columns={
                    "team": "opponent",
                    "team_strength_score": "opponent_team_strength_score",
                },
            ),
            on="opponent",
            how="left",
        )
    else:
        probabilities["team_strength_score"] = np.nan
        probabilities["opponent_team_strength_score"] = np.nan

    win_probabilities: list[float] = []
    sources: list[str] = []
    for row in probabilities.to_dict("records"):
        h2h_probability = _float_or_none(row.get("h2h_probability"))
        if h2h_probability is not None:
            win_probabilities.append(h2h_probability)
            sources.append(PROBABILITY_SOURCE_REAL_MONEYLINE)
            continue

        spread_probability = _float_or_none(row.get("spread_probability"))
        if spread_probability is not None:
            win_probabilities.append(spread_probability)
            sources.append(PROBABILITY_SOURCE_REAL_SPREAD)
            continue

        team_strength = _float_or_none(row.get("team_strength_score"))
        opponent_strength = _float_or_none(row.get("opponent_team_strength_score"))
        if team_strength is not None and opponent_strength is not None:
            win_probabilities.append(
                estimate_team_strength_win_probability(
                    team_rating=team_strength,
                    opponent_rating=opponent_strength,
                    is_home=row.get("is_home"),
                    home_field_adjustment=team_strength_home_field_adjustment,
                    scale=team_strength_scale,
                ),
            )
            sources.append(PROBABILITY_SOURCE_PROJECTED_TEAM_STRENGTH)
            continue

        win_probabilities.append(
            DEFAULT_HOME_WIN_PROBABILITY if bool(row.get("is_home")) else 1 - DEFAULT_HOME_WIN_PROBABILITY,
        )
        sources.append(PROBABILITY_SOURCE_FALLBACK_DEFAULT)

    probabilities["win_probability"] = [_clip_probability(value) for value in win_probabilities]
    probabilities["probability_source"] = sources
    probabilities = _normalize_game_probabilities(probabilities)
    probabilities["no_vig_win_probability"] = probabilities["win_probability"]
    probabilities["moneyline"] = np.nan
    probabilities["implied_probability"] = np.nan
    return probabilities[
        [
            "week",
            "game_id",
            "team",
            "opponent",
            "is_home",
            "moneyline",
            "implied_probability",
            "win_probability",
            "no_vig_win_probability",
            "probability_source",
            "team_strength_score",
            "opponent_team_strength_score",
        ]
    ].sort_values(["week", "game_id", "team"]).reset_index(drop=True)


def _schedule_team_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in schedule.to_dict("records"):
        rows.append(
            {
                "week": int(row["week"]),
                "game_id": row["game_id"],
                "team": str(row["home_team"]),
                "opponent": str(row["away_team"]),
                "is_home": True,
            },
        )
        rows.append(
            {
                "week": int(row["week"]),
                "game_id": row["game_id"],
                "team": str(row["away_team"]),
                "opponent": str(row["home_team"]),
                "is_home": False,
            },
        )
    return pd.DataFrame(rows)


def _extract_h2h_probabilities(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return _empty_probability_signal("h2h_probability")

    odds = odds_df.copy()
    try:
        team_odds = add_no_vig_probabilities(odds)
    except Exception:
        if {"week", "game_id", "team", "no_vig_win_probability"}.issubset(odds.columns):
            team_odds = odds.copy()
        else:
            return _empty_probability_signal("h2h_probability")

    if "market_type" in team_odds.columns:
        market = team_odds["market_type"].astype(str).str.lower()
        team_odds = team_odds[market.isin({"h2h", "moneyline"})].copy()
    if team_odds.empty or "no_vig_win_probability" not in team_odds.columns:
        return _empty_probability_signal("h2h_probability")

    required = {"week", "game_id", "team"}
    if not required.issubset(team_odds.columns):
        return _empty_probability_signal("h2h_probability")

    team_odds["week"] = pd.to_numeric(team_odds["week"], errors="coerce")
    team_odds["h2h_probability"] = pd.to_numeric(
        team_odds["no_vig_win_probability"],
        errors="coerce",
    )
    team_odds = team_odds.dropna(subset=["week", "game_id", "team", "h2h_probability"])
    if team_odds.empty:
        return _empty_probability_signal("h2h_probability")
    team_odds["week"] = team_odds["week"].astype(int)
    return (
        team_odds.groupby(["week", "game_id", "team"], as_index=False)
        .agg(h2h_probability=("h2h_probability", "mean"))
        .reset_index(drop=True)
    )


def _extract_spread_probabilities(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty or not {"week", "game_id", "team", "spread"}.issubset(odds_df.columns):
        return _empty_probability_signal("spread_probability")

    spreads = odds_df.copy()
    if "market_type" in spreads.columns:
        market = spreads["market_type"].astype(str).str.lower()
        spreads = spreads[market.isin({"spreads", "spread"})].copy()
    spreads["week"] = pd.to_numeric(spreads["week"], errors="coerce")
    spreads["spread"] = pd.to_numeric(spreads["spread"], errors="coerce")
    spreads = spreads.dropna(subset=["week", "game_id", "team", "spread"])
    if spreads.empty:
        return _empty_probability_signal("spread_probability")

    spreads["week"] = spreads["week"].astype(int)
    spreads["spread_probability"] = spreads.apply(estimate_game_win_probability, axis=1)
    return (
        spreads.groupby(["week", "game_id", "team"], as_index=False)
        .agg(spread_probability=("spread_probability", "mean"))
        .reset_index(drop=True)
    )


def _extract_team_strength(
    odds_df: pd.DataFrame,
    team_strength_df: pd.DataFrame,
) -> pd.DataFrame:
    explicit_strength = _normalize_team_strength_input(team_strength_df)
    if not explicit_strength.empty:
        return explicit_strength

    if odds_df.empty:
        return pd.DataFrame(columns=["team", "team_strength_score"])

    if {"team", "team_strength_score"}.issubset(odds_df.columns):
        strength = odds_df[["team", "team_strength_score"]].copy()
        strength["team_strength_score"] = pd.to_numeric(
            strength["team_strength_score"],
            errors="coerce",
        )
        strength = strength.dropna(subset=["team", "team_strength_score"])
        if not strength.empty:
            return strength.groupby("team", as_index=False).agg(
                team_strength_score=("team_strength_score", "mean"),
            )

    if "market_type" not in odds_df.columns:
        return pd.DataFrame(columns=["team", "team_strength_score"])
    futures = odds_df[
        odds_df["market_type"].astype(str).str.lower().isin(
            {"super_bowl", "conference", "division", "playoff", "win_total"},
        )
    ].copy()
    if futures.empty:
        return pd.DataFrame(columns=["team", "team_strength_score"])
    try:
        strength = build_team_strength_priors(futures)
    except Exception:
        return pd.DataFrame(columns=["team", "team_strength_score"])
    return strength[["team", "team_strength_score"]].copy()


def _normalize_team_strength_input(team_strength_df: pd.DataFrame) -> pd.DataFrame:
    if team_strength_df.empty:
        return pd.DataFrame(columns=["team", "team_strength_score"])

    strength = team_strength_df.copy()
    if "team_strength_score" not in strength.columns and "rating" in strength.columns:
        strength = strength.rename(columns={"rating": "team_strength_score"})
    required = {"team", "team_strength_score"}
    missing = required - set(strength.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"team strength data is missing required columns: {missing_text}")

    strength["team"] = strength["team"].astype(str).str.strip()
    strength["team_strength_score"] = pd.to_numeric(
        strength["team_strength_score"],
        errors="raise",
    )
    strength = strength.dropna(subset=["team", "team_strength_score"])
    if strength.empty:
        return pd.DataFrame(columns=["team", "team_strength_score"])
    duplicate_teams = strength.duplicated("team", keep=False)
    if bool(duplicate_teams.any()):
        examples = ", ".join(sorted(strength.loc[duplicate_teams, "team"].astype(str).unique())[:5])
        raise ValueError(f"team strength data has duplicate teams: {examples}.")
    return strength[["team", "team_strength_score"]].reset_index(drop=True)


def _validate_team_strength_parameters(
    *,
    team_strength_home_field_adjustment: float,
    team_strength_scale: float,
) -> None:
    try:
        float(team_strength_home_field_adjustment)
    except (TypeError, ValueError) as exc:
        raise ValueError("team_strength_home_field_adjustment must be numeric.") from exc
    try:
        scale = float(team_strength_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("team_strength_scale must be numeric.") from exc
    if scale <= 0:
        raise ValueError("team_strength_scale must be positive.")


def _normalize_game_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    output = probabilities.copy()
    normalized: list[pd.DataFrame] = []
    for (_, game_id), game_df in output.groupby(["week", "game_id"], sort=True):
        group = game_df.copy()
        values = group["win_probability"].astype(float).clip(lower=0, upper=1)
        total = float(values.sum())
        if total <= 0:
            group["win_probability"] = 1.0 / len(group)
        else:
            group["win_probability"] = values / total
        normalized.append(group)
    return pd.concat(normalized, ignore_index=True)


def _add_projected_public_picks(
    team_probabilities: pd.DataFrame,
    public_picks_df: pd.DataFrame,
) -> pd.DataFrame:
    output = team_probabilities.copy()
    output["projected_public_pick_pct"] = 0.0
    output["public_pick_source"] = PROJECTED_PUBLIC_PICK_SOURCE
    output["team_popularity_weight"] = (
        output["team"]
        .astype(str)
        .map(TEAM_POPULARITY_PLACEHOLDER)
        .fillna(1.0)
        .astype(float)
    )
    output["week_alternative_scarcity"] = 0.0
    output["ownership_projection_weight"] = 0.0

    public_picks = _normalize_public_picks(public_picks_df)
    for week, week_df in output.groupby("week", sort=True):
        week_mask = output["week"].astype(int) == int(week)
        week_public = public_picks[public_picks["week"].astype(int) == int(week)].copy()
        scarcity = _week_alternative_scarcity(week_df["win_probability"])
        output.loc[week_mask, "week_alternative_scarcity"] = scarcity
        if not week_public.empty and float(week_public["public_pick_pct"].sum()) > 0:
            picks_by_team = dict(
                zip(
                    week_public["team"].astype(str),
                    week_public["public_pick_pct"].astype(float),
                    strict=False,
                ),
            )
            output.loc[week_mask, "projected_public_pick_pct"] = output.loc[
                week_mask,
                "team",
            ].astype(str).map(picks_by_team).fillna(0.0)
            output.loc[week_mask, "public_pick_source"] = "public_picks"
            output.loc[week_mask, "ownership_projection_weight"] = output.loc[
                week_mask,
                "projected_public_pick_pct",
            ].astype(float)
            continue

        probabilities = week_df["win_probability"].astype(float).clip(
            lower=0.01,
            upper=0.99,
        )
        popularity = (
            week_df["team"]
            .astype(str)
            .map(TEAM_POPULARITY_PLACEHOLDER)
            .fillna(1.0)
            .astype(float)
        )
        exponent = 2.0 + 2.0 * scarcity
        favorite_boost = np.where(probabilities >= 0.70, 1.0 + 0.50 * scarcity, 1.0)
        weights = (probabilities**exponent) * popularity * favorite_boost
        total_weight = float(weights.sum())
        projected = weights / total_weight if total_weight > 0 else np.full(len(week_df), 1 / len(week_df))
        output.loc[week_df.index, "projected_public_pick_pct"] = projected
        output.loc[week_df.index, "ownership_projection_weight"] = weights

    return output


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


def _rankings_by_week(
    *,
    weeks: list[int],
    schedule: pd.DataFrame,
    team_probabilities: pd.DataFrame,
    public_picks: pd.DataFrame,
    pool_size: int,
) -> dict[int, pd.DataFrame]:
    rankings: dict[int, pd.DataFrame] = {}
    odds_for_ranking = team_probabilities.rename(
        columns={"win_probability": "path_win_probability"},
    ).copy()
    for week in weeks:
        try:
            ranked = rank_weekly_picks(
                week=week,
                schedule_df=schedule,
                odds_df=odds_for_ranking,
                public_picks_df=public_picks,
                entries_df=None,
                pool_size=pool_size,
            )
        except Exception:
            ranked = _fallback_week_ranking(
                week,
                team_probabilities[team_probabilities["week"].astype(int) == int(week)],
            )

        enrich = team_probabilities[
            [
                "week",
                "team",
                "win_probability",
                "probability_source",
                "projected_public_pick_pct",
                "public_pick_source",
            ]
        ].copy()
        ranked = ranked.merge(enrich, on=["week", "team"], how="left")
        if "win_probability" not in ranked.columns or ranked["win_probability"].isna().any():
            ranked["win_probability"] = ranked["no_vig_win_probability"]
        if "projected_public_pick_pct" not in ranked.columns:
            ranked["projected_public_pick_pct"] = ranked["public_pick_pct"]
        rankings[week] = ranked.sort_values(
            ["rank" if "rank" in ranked.columns else "final_score", "win_probability"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return rankings


def _fallback_week_ranking(week: int, week_probabilities: pd.DataFrame) -> pd.DataFrame:
    ranked = week_probabilities.copy()
    ranked["public_pick_pct"] = ranked["projected_public_pick_pct"]
    ranked["no_vig_win_probability"] = ranked["win_probability"]
    ranked["final_score"] = ranked["win_probability"] * (1 - ranked["projected_public_pick_pct"])
    ranked = ranked.sort_values(
        ["final_score", "win_probability"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _expected_public_survival_rates(team_probabilities: pd.DataFrame) -> dict[int, float]:
    rates: dict[int, float] = {}
    distributions = _public_distributions(team_probabilities)
    for week, distribution in distributions.items():
        rates[int(week)] = float(
            (
                distribution["pick_probability"].astype(float)
                * distribution["win_probability"].astype(float)
            ).sum(),
        )
    return rates


def _paths_frame(paths: list[_BeamPath]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path_number, path in enumerate(paths, start=1):
        path_key = _format_path_key(path.steps)
        for step in path.steps:
            row = dict(step)
            row["path_id"] = path_number
            row["path_rank"] = path_number
            row["path_key"] = path_key
            rows.append(row)
    if not rows:
        return _empty_paths_frame()
    return pd.DataFrame(rows)[PATH_COLUMNS].sort_values(["path_id", "week"]).reset_index(drop=True)


def _prepare_paths_for_evaluation(paths: pd.DataFrame, team_probabilities: pd.DataFrame) -> pd.DataFrame:
    path_df = _as_dataframe(paths)
    required = {"week", "team"}
    missing = required - set(path_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"path data is missing required columns: {missing_text}")

    path_df["week"] = pd.to_numeric(path_df["week"], errors="raise").astype(int)
    if "path_id" not in path_df.columns:
        path_df["path_id"] = 1
    path_df["path_id"] = pd.to_numeric(path_df["path_id"], errors="raise").astype(int)
    path_df["team"] = path_df["team"].astype(str)

    duplicate_weeks = path_df.duplicated(["path_id", "week"], keep=False)
    if bool(duplicate_weeks.any()):
        raise ValueError("Each path can contain only one team per week.")
    duplicate_teams = path_df.duplicated(["path_id", "team"], keep=False)
    if bool(duplicate_teams.any()):
        raise ValueError("Survivor paths cannot reuse a team.")

    enrich_columns = [
        "week",
        "team",
        "opponent",
        "game_id",
        "is_home",
        "win_probability",
        "probability_source",
        "projected_public_pick_pct",
        "public_pick_source",
    ]
    enriched = path_df.merge(
        team_probabilities[enrich_columns],
        on=["week", "team"],
        how="left",
        suffixes=("", "_model"),
    )
    if enriched["game_id"].isna().any():
        missing_rows = enriched[enriched["game_id"].isna()][["week", "team"]]
        examples = ", ".join(
            f"W{int(row['week'])}:{row['team']}" for row in missing_rows.head(5).to_dict("records")
        )
        raise ValueError(f"Path includes teams without scheduled model rows: {examples}")

    for column in [
        "opponent",
        "game_id",
        "win_probability",
        "probability_source",
        "projected_public_pick_pct",
        "public_pick_source",
    ]:
        model_column = f"{column}_model"
        if model_column in enriched.columns:
            if column not in enriched.columns:
                enriched[column] = enriched[model_column]
            else:
                enriched[column] = enriched[column].where(enriched[column].notna(), enriched[model_column])
            enriched = enriched.drop(columns=[model_column])

    if "path_key" not in enriched.columns:
        path_keys = {
            path_id: _format_path_key(group.sort_values("week").to_dict("records"))
            for path_id, group in enriched.groupby("path_id", sort=True)
        }
        enriched["path_key"] = enriched["path_id"].map(path_keys)

    if "estimated_path_ev" not in enriched.columns:
        enriched["estimated_path_ev"] = np.nan
    return enriched.sort_values(["path_id", "week"]).reset_index(drop=True)


def _simulate_common_seasons(
    *,
    model: _PathModelData,
    weeks: list[int],
    pool_size: int,
    simulations: int,
    random_seed: int | None,
    use_public_field_simulation: bool,
    public_field_sample_size: int | None,
) -> _CommonSimulation:
    rng = np.random.default_rng(random_seed)
    team_wins: dict[tuple[int, str], np.ndarray] = {}
    distributions = _public_distributions(model.team_probabilities)

    for week in weeks:
        week_probabilities = model.team_probabilities[
            model.team_probabilities["week"].astype(int) == int(week)
        ].copy()
        for _, game_df in week_probabilities.groupby("game_id", sort=True):
            game = game_df.sort_values("team").reset_index(drop=True)
            if len(game) < 2:
                continue
            team_a = str(game.iloc[0]["team"])
            team_b = str(game.iloc[1]["team"])
            probability_a = float(game.iloc[0]["win_probability"])
            draws = rng.random(simulations)
            a_wins = draws < probability_a
            team_wins[(int(week), team_a)] = a_wins
            team_wins[(int(week), team_b)] = ~a_wins

    public_field = None
    public_entry_count = max(int(pool_size) - 1, 0)
    if (
        use_public_field_simulation
        and len(weeks) > 1
        and public_entry_count > 0
    ):
        public_field = simulate_public_field_paths(
            schedule_df=model.schedule,
            odds_df=pd.DataFrame(),
            public_picks_df=model.public_picks,
            start_week=int(min(weeks)),
            pool_size=public_entry_count,
            simulations=simulations,
            random_seed=random_seed,
            public_field_sample_size=public_field_sample_size,
            team_probabilities_df=model.team_probabilities,
            weeks=weeks,
            include_entry_paths=False,
        )

    return _CommonSimulation(
        weeks=[int(week) for week in weeks],
        team_wins=team_wins,
        public_distributions={
            int(week): distributions[int(week)]
            for week in weeks
            if int(week) in distributions
        },
        public_field=public_field,
    )


def _evaluate_prepared_paths(
    *,
    paths: pd.DataFrame,
    simulation: _CommonSimulation,
    simulations: int,
    pool_size: int,
    entry_fee: float | None,
    prize_pool: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if simulation.public_field is not None:
        return _evaluate_prepared_paths_with_public_field(
            paths=paths,
            public_field=simulation.public_field,
            pool_size=pool_size,
            entry_fee=entry_fee,
            prize_pool=prize_pool,
        )

    summaries: list[dict[str, Any]] = []
    survival_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for path_id, path in paths.groupby("path_id", sort=True):
        ordered = path.sort_values("week").reset_index(drop=True)
        path_survival_probability = 1.0
        public_entries_if_alive = np.full(
            simulations,
            max(float(pool_size) - 1.0, 0.0),
            dtype=float,
        )
        previous_cumulative_path_ev = 1.0 / max(float(pool_size), 1.0)
        for row in ordered.to_dict("records"):
            week = int(row["week"])
            team = str(row["team"])
            game_id = str(row["game_id"])
            path_survival_probability *= float(row["win_probability"])
            public_entries_if_alive = _advance_public_entries_conditional_on_path(
                public_entries_before=public_entries_if_alive,
                distribution=simulation.public_distributions[week],
                simulation=simulation,
                week=week,
                selected_team=team,
                selected_game_id=game_id,
            )
            expected_public_entries = float(public_entries_if_alive.mean())
            expected_total_entries = expected_public_entries + 1.0
            equity_if_alive = 1.0 / np.maximum(public_entries_if_alive + 1.0, 1.0)
            expected_equity_if_alive = float(equity_if_alive.mean())
            cumulative_path_ev = float(path_survival_probability * expected_equity_if_alive)
            survival_rows.append(
                {
                    "path_id": int(path_id),
                    "week": week,
                    "path_survival_probability": float(path_survival_probability),
                },
            )
            field_rows.append(
                {
                    "path_id": int(path_id),
                    "week": week,
                    "expected_public_entries": expected_public_entries,
                    "expected_total_entries": expected_total_entries,
                    "expected_equity_if_alive": expected_equity_if_alive,
                    "cumulative_path_ev": cumulative_path_ev,
                    "weekly_ev_delta": cumulative_path_ev - previous_cumulative_path_ev,
                },
            )
            previous_cumulative_path_ev = cumulative_path_ev

        final_public_entries = public_entries_if_alive
        total_survivors_if_alive = final_public_entries + 1.0
        equity_if_alive = 1.0 / np.maximum(total_survivors_if_alive, 1.0)
        expected_equity_if_alive = float(equity_if_alive.mean())
        path_ev = float(path_survival_probability * expected_equity_if_alive)
        expected_survivors_if_alive = float(total_survivors_if_alive.mean())
        estimated_path_ev = _last_numeric(ordered, "estimated_path_ev")
        value_metrics = _value_metrics(
            path_ev=path_ev,
            pool_size=pool_size,
            entry_fee=entry_fee,
            prize_pool=prize_pool,
        )
        summaries.append(
            {
                "path_id": int(path_id),
                "path": _format_path_key(ordered.to_dict("records")),
                "first_week": int(ordered["week"].min()),
                "first_team": str(ordered.iloc[0]["team"]),
                "path_ev": path_ev,
                "expected_contest_equity": path_ev,
                "path_survival_probability": float(path_survival_probability),
                "expected_final_field_size": expected_survivors_if_alive,
                "expected_final_public_entries": float(final_public_entries.mean()),
                "expected_survivors_if_alive": expected_survivors_if_alive,
                "expected_equity_if_alive": expected_equity_if_alive,
                **value_metrics,
                "estimated_path_ev": estimated_path_ev,
                "cumulative_survival_probability": _last_numeric(
                    ordered,
                    "cumulative_survival_probability",
                ),
            },
        )

    evaluated = pd.DataFrame(summaries).sort_values(
        ["path_ev", "path_survival_probability", "estimated_path_ev"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    survival_by_week = pd.DataFrame(survival_rows).sort_values(
        ["path_id", "week"],
    ).reset_index(drop=True)
    field_by_week = pd.DataFrame(field_rows).sort_values(
        ["path_id", "week"],
    ).reset_index(drop=True)
    return evaluated, survival_by_week, field_by_week


def _evaluate_prepared_paths_with_public_field(
    *,
    paths: pd.DataFrame,
    public_field: PublicFieldSimulationResult,
    pool_size: int,
    entry_fee: float | None,
    prize_pool: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    survival_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for path_id, path in paths.groupby("path_id", sort=True):
        ordered = path.sort_values("week").reset_index(drop=True)
        path_survival_probability = 1.0
        previous_cumulative_path_ev = 1.0 / max(float(pool_size), 1.0)
        public_alive = np.ones(
            (public_field.simulations, public_field.sample_entry_count),
            dtype=bool,
        )

        for row in ordered.to_dict("records"):
            week = int(row["week"])
            team = str(row["team"])
            game_id = str(row["game_id"])
            week_index = public_field.week_to_index.get(week)
            if week_index is None:
                raise ValueError(f"Public field simulation is missing week {week}.")
            if team not in public_field.team_to_code:
                raise ValueError(f"Public field simulation is missing team {team}.")

            team_code = int(public_field.team_to_code[team])
            game_code = int(public_field.game_to_code[str(game_id)])
            picks = public_field.pick_matrix[:, :, week_index]
            pick_games = public_field.pick_game_matrix[:, :, week_index]
            wins = public_field.pick_win_matrix[:, :, week_index]
            same_game = pick_games == game_code
            conditioned_wins = np.where(picks == team_code, True, wins)
            conditioned_wins = np.where(
                same_game & (picks != team_code),
                False,
                conditioned_wins,
            )

            public_entries_before = public_alive.sum(axis=1).astype(float) * public_field.entry_weight
            overlap_entries = (
                (public_alive & (picks == team_code)).sum(axis=1).astype(float)
                * public_field.entry_weight
            )
            overlap_pct = np.divide(
                overlap_entries,
                public_entries_before,
                out=np.zeros_like(overlap_entries),
                where=public_entries_before > 0,
            )

            public_alive = public_alive & conditioned_wins
            public_entries_after = public_alive.sum(axis=1).astype(float) * public_field.entry_weight
            path_survival_probability *= float(row["win_probability"])
            expected_public_entries = float(public_entries_after.mean())
            expected_total_entries = expected_public_entries + 1.0
            equity_if_alive = 1.0 / np.maximum(public_entries_after + 1.0, 1.0)
            expected_equity_if_alive = float(equity_if_alive.mean())
            cumulative_path_ev = float(
                path_survival_probability * expected_equity_if_alive,
            )
            avg_overlap_pct = float(overlap_pct.mean())
            survival_rows.append(
                {
                    "path_id": int(path_id),
                    "week": week,
                    "path_survival_probability": float(path_survival_probability),
                },
            )
            field_rows.append(
                {
                    "path_id": int(path_id),
                    "week": week,
                    "expected_public_entries_before_week": float(
                        public_entries_before.mean(),
                    ),
                    "expected_public_entries": expected_public_entries,
                    "expected_total_entries": expected_total_entries,
                    "expected_equity_if_alive": expected_equity_if_alive,
                    "expected_public_overlap_entries": float(overlap_entries.mean()),
                    "avg_path_overlap_pct": avg_overlap_pct,
                    "path_uniqueness_score": float(1.0 - avg_overlap_pct),
                    "cumulative_path_ev": cumulative_path_ev,
                    "weekly_ev_delta": cumulative_path_ev - previous_cumulative_path_ev,
                    "public_field_model": PUBLIC_FIELD_PICK_SOURCE,
                },
            )
            previous_cumulative_path_ev = cumulative_path_ev

        final_public_entries = public_alive.sum(axis=1).astype(float) * public_field.entry_weight
        total_survivors_if_alive = final_public_entries + 1.0
        equity_if_alive = 1.0 / np.maximum(total_survivors_if_alive, 1.0)
        expected_equity_if_alive = float(equity_if_alive.mean())
        path_ev = float(path_survival_probability * expected_equity_if_alive)
        expected_survivors_if_alive = float(total_survivors_if_alive.mean())
        estimated_path_ev = _last_numeric(ordered, "estimated_path_ev")
        value_metrics = _value_metrics(
            path_ev=path_ev,
            pool_size=pool_size,
            entry_fee=entry_fee,
            prize_pool=prize_pool,
        )
        summaries.append(
            {
                "path_id": int(path_id),
                "path": _format_path_key(ordered.to_dict("records")),
                "first_week": int(ordered["week"].min()),
                "first_team": str(ordered.iloc[0]["team"]),
                "path_ev": path_ev,
                "expected_contest_equity": path_ev,
                "path_survival_probability": float(path_survival_probability),
                "expected_final_field_size": expected_survivors_if_alive,
                "expected_final_public_entries": float(final_public_entries.mean()),
                "expected_survivors_if_alive": expected_survivors_if_alive,
                "expected_equity_if_alive": expected_equity_if_alive,
                **value_metrics,
                "estimated_path_ev": estimated_path_ev,
                "cumulative_survival_probability": _last_numeric(
                    ordered,
                    "cumulative_survival_probability",
                ),
            },
        )

    evaluated = pd.DataFrame(summaries).sort_values(
        ["path_ev", "path_survival_probability", "estimated_path_ev"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    survival_by_week = pd.DataFrame(survival_rows).sort_values(
        ["path_id", "week"],
    ).reset_index(drop=True)
    field_by_week = pd.DataFrame(field_rows).sort_values(
        ["path_id", "week"],
    ).reset_index(drop=True)
    return evaluated, survival_by_week, field_by_week


def _advance_public_entries_conditional_on_path(
    *,
    public_entries_before: np.ndarray,
    distribution: pd.DataFrame,
    simulation: _CommonSimulation,
    week: int,
    selected_team: str,
    selected_game_id: str,
) -> np.ndarray:
    teams = distribution["team"].astype(str).tolist()
    probabilities = distribution["pick_probability"].astype(float).to_numpy()
    counts = probabilities[:, np.newaxis] * (public_entries_before[np.newaxis, :] + 1.0)

    selected_indices = [index for index, team in enumerate(teams) if team == selected_team]
    if selected_indices:
        selected_index = selected_indices[0]
        counts[selected_index] = counts[selected_index] - 1.0
        deficit = np.maximum(-counts[selected_index], 0.0)
        counts[selected_index] = np.maximum(counts[selected_index], 0.0)
        if bool(np.any(deficit > 0)):
            other_indices = [index for index in range(len(teams)) if index != selected_index]
            if other_indices:
                other_total = counts[other_indices].sum(axis=0)
                target_other_total = np.maximum(other_total - deficit, 0.0)
                scale = np.divide(
                    target_other_total,
                    other_total,
                    out=np.zeros_like(other_total),
                    where=other_total > 0,
                )
                counts[other_indices] = counts[other_indices] * scale

    survivors = np.zeros_like(public_entries_before, dtype=float)
    game_ids = distribution["game_id"].astype(str).tolist()
    for index, team in enumerate(teams):
        if team == selected_team:
            wins = np.ones_like(public_entries_before, dtype=bool)
        elif game_ids[index] == selected_game_id:
            wins = np.zeros_like(public_entries_before, dtype=bool)
        else:
            wins = simulation.team_wins[(int(week), team)]
        survivors += np.where(wins, counts[index], 0.0)
    return survivors


def _public_distributions(team_probabilities: pd.DataFrame) -> dict[int, pd.DataFrame]:
    distributions: dict[int, pd.DataFrame] = {}
    for week, week_df in team_probabilities.groupby("week", sort=True):
        distribution = week_df[
            ["week", "game_id", "team", "win_probability", "projected_public_pick_pct"]
        ].copy()
        weights = distribution["projected_public_pick_pct"].astype(float).clip(lower=0)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            weights = distribution["win_probability"].astype(float).clip(lower=0.01)
            total_weight = float(weights.sum())
        distribution["pick_probability"] = (
            weights / total_weight if total_weight > 0 else 1.0 / len(distribution)
        )
        distributions[int(week)] = distribution.sort_values("team").reset_index(drop=True)
    return distributions


def _build_heuristic_comparison(
    *,
    start_week: int,
    rankings: pd.DataFrame,
    evaluated_paths: pd.DataFrame,
    prepared_paths: pd.DataFrame,
    best_path_id: int,
) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()

    current_paths = prepared_paths[
        prepared_paths["week"].astype(int) == int(start_week)
    ][["path_id", "team"]].copy()
    path_scores = current_paths.merge(
        evaluated_paths[
            [
                "path_id",
                "path_ev",
                "path_ev_rank",
                "path_survival_probability",
                "expected_final_field_size",
            ]
        ],
        on="path_id",
        how="left",
    )
    best_by_team = (
        path_scores.sort_values(["team", "path_ev"], ascending=[True, False])
        .drop_duplicates("team")
        .rename(
            columns={
                "path_ev": "best_path_ev_for_current_pick",
                "path_ev_rank": "best_path_ev_rank_for_current_pick",
            },
        )
    )

    comparison_columns = [
        "rank",
        "team",
        "opponent",
        "no_vig_win_probability",
        "public_pick_pct",
        "final_score",
    ]
    comparison = rankings[[column for column in comparison_columns if column in rankings.columns]].copy()
    comparison = comparison.merge(best_by_team, on="team", how="left")
    best_current_team = prepared_paths[
        (prepared_paths["path_id"].astype(int) == int(best_path_id))
        & (prepared_paths["week"].astype(int) == int(start_week))
    ]["team"].astype(str)
    comparison["selected_by_path_ev"] = comparison["team"].astype(str).isin(set(best_current_team))
    return comparison.rename(
        columns={
            "rank": "heuristic_rank",
            "final_score": "heuristic_final_score",
        },
    ).sort_values("heuristic_rank").reset_index(drop=True)


def _normalize_public_picks(public_picks_df: pd.DataFrame) -> pd.DataFrame:
    if public_picks_df.empty:
        return pd.DataFrame(columns=["week", "team", "public_pick_pct"])

    picks = public_picks_df.copy()
    if "public_pick_pct" not in picks.columns:
        if "pick_share" in picks.columns:
            picks = picks.rename(columns={"pick_share": "public_pick_pct"})
        else:
            raise ValueError("Public pick data needs public_pick_pct or pick_share column.")
    required = {"week", "team", "public_pick_pct"}
    missing = required - set(picks.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Public pick data is missing required columns: {missing_text}")
    picks["week"] = pd.to_numeric(picks["week"], errors="raise").astype(int)
    picks["team"] = picks["team"].astype(str)
    picks["public_pick_pct"] = pd.to_numeric(
        picks["public_pick_pct"],
        errors="raise",
    ).astype(float)
    if not picks.empty and picks["public_pick_pct"].max() > 1:
        picks["public_pick_pct"] = picks["public_pick_pct"] / 100
    if not picks["public_pick_pct"].between(0, 1).all():
        raise ValueError("public_pick_pct values must be between 0 and 1.")
    return picks[["week", "team", "public_pick_pct"]].copy()


def _empty_probability_signal(column: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["week", "game_id", "team", column])


def _empty_paths_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PATH_COLUMNS)


def _format_path_key(steps: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> str:
    return " > ".join(f"W{int(step['week'])}:{step['team']}" for step in steps)


def _last_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _value_metrics(
    *,
    path_ev: float,
    pool_size: int,
    entry_fee: float | None,
    prize_pool: float | None,
) -> dict[str, float | None]:
    baseline_contest_equity = 1.0 / float(pool_size)
    entry_cost = float(entry_fee) if entry_fee is not None else None
    baseline_value = (
        float(prize_pool) / float(pool_size)
        if prize_pool is not None
        else None
    )
    ev_dollars = float(path_ev) * float(prize_pool) if prize_pool is not None else None
    ev_multiple_vs_entry_fee = (
        ev_dollars / entry_cost
        if ev_dollars is not None and entry_cost not in {None, 0}
        else None
    )
    ev_multiple_vs_baseline = (
        ev_dollars / baseline_value
        if ev_dollars is not None and baseline_value not in {None, 0}
        else None
    )
    expected_edge_vs_entry_fee = (
        ev_multiple_vs_entry_fee - 1.0
        if ev_multiple_vs_entry_fee is not None
        else None
    )
    expected_edge_vs_baseline = (
        ev_multiple_vs_baseline - 1.0
        if ev_multiple_vs_baseline is not None
        else None
    )
    return {
        "baseline_contest_equity": baseline_contest_equity,
        "entry_fee": float(entry_fee) if entry_fee is not None else None,
        "prize_pool": float(prize_pool) if prize_pool is not None else None,
        "entry_cost": entry_cost,
        "baseline_value": baseline_value,
        "ev_dollars": ev_dollars,
        "ev_multiple": float(ev_multiple_vs_entry_fee)
        if ev_multiple_vs_entry_fee is not None
        else None,
        "expected_edge": float(expected_edge_vs_entry_fee)
        if expected_edge_vs_entry_fee is not None
        else None,
        "ev_multiple_vs_entry_fee": float(ev_multiple_vs_entry_fee)
        if ev_multiple_vs_entry_fee is not None
        else None,
        "ev_multiple_vs_baseline": float(ev_multiple_vs_baseline)
        if ev_multiple_vs_baseline is not None
        else None,
        "expected_edge_vs_entry_fee": float(expected_edge_vs_entry_fee)
        if expected_edge_vs_entry_fee is not None
        else None,
        "expected_edge_vs_baseline": float(expected_edge_vs_baseline)
        if expected_edge_vs_baseline is not None
        else None,
    }


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


def _float_or_nan(value: Any) -> float:
    converted = _float_or_none(value)
    return float("nan") if converted is None else float(converted)


def _optional_float(value: Any) -> float | None:
    converted = _float_or_none(value)
    return None if converted is None else float(converted)


def _int_or_nan(value: Any) -> int | float:
    converted = _float_or_none(value)
    return float("nan") if converted is None else int(converted)


def _clip_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _as_dataframe(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def _validate_positive_int(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be positive when provided.")


def _validate_non_negative_optional(name: str, value: float | None) -> None:
    if value is not None and float(value) < 0:
        raise ValueError(f"{name} cannot be negative.")
