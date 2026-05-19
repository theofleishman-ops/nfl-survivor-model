"""Path clustering and future-route crowding diagnostics for survivor EV.

Weekly ownership answers "who is taking KC this week?"  These helpers answer a
different question: "how many entries are likely to arrive at a similar
KC/BUF/PHI/DET route later?"  The functions are intentionally deterministic and
small enough to use in tests, reports, and path-EV replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_LATE_SEASON_OVERLAP_WEIGHT = 2.0
DEFAULT_PATH_CONVERGENCE_TEMPERATURE = 0.85
DEFAULT_ELITE_WIN_PROBABILITY = 0.70
DEFAULT_LATE_WINDOW = 4


@dataclass(frozen=True)
class _PathSummary:
    path_id: Any
    path_key: str
    late_path_key: str
    teams: tuple[str, ...]
    weeks: tuple[int, ...]
    weight: float
    survived_full_path: bool
    rows: pd.DataFrame


def compute_path_similarity(
    path_a: pd.DataFrame | list[dict[str, Any]] | list[str],
    path_b: pd.DataFrame | list[dict[str, Any]] | list[str],
    *,
    elite_teams: set[str] | list[str] | tuple[str, ...] | None = None,
    late_season_overlap_weight: float = DEFAULT_LATE_SEASON_OVERLAP_WEIGHT,
    elite_win_probability: float = DEFAULT_ELITE_WIN_PROBABILITY,
) -> dict[str, float]:
    """Return path similarity diagnostics between two survivor routes.

    The score blends unordered team overlap, same-week ordered overlap, late
    overlap, and shared elite-team overlap.  Late weeks receive extra weight so
    two routes that converge onto the same obvious favorites late in the season
    cluster more strongly than routes that merely share an early pick.
    """

    first = _normalize_path(path_a)
    second = _normalize_path(path_b)
    if first.empty or second.empty:
        return _empty_similarity()

    elite = _resolve_elite_teams(
        [first, second],
        elite_teams=elite_teams,
        elite_win_probability=elite_win_probability,
    )
    teams_a = set(first["team"].astype(str))
    teams_b = set(second["team"].astype(str))
    exact_team_overlap = _jaccard(teams_a, teams_b)

    by_week_a = dict(zip(first["week"].astype(int), first["team"].astype(str), strict=False))
    by_week_b = dict(zip(second["week"].astype(int), second["team"].astype(str), strict=False))
    common_weeks = sorted(set(by_week_a) & set(by_week_b))
    ordered_overlap = (
        float(sum(1 for week in common_weeks if by_week_a[week] == by_week_b[week]))
        / float(len(common_weeks))
        if common_weeks
        else 0.0
    )

    all_weeks = sorted(set(by_week_a) | set(by_week_b))
    late_weights = _late_week_weights(all_weeks, late_season_overlap_weight)
    weighted_matches = 0.0
    weighted_total = 0.0
    for week in common_weeks:
        weight = late_weights.get(int(week), 1.0)
        weighted_total += weight
        if by_week_a[week] == by_week_b[week]:
            weighted_matches += weight
    late_ordered_overlap = (
        weighted_matches / weighted_total if weighted_total > 0 else 0.0
    )

    late_weeks = _late_weeks(all_weeks)
    late_teams_a = {team for week, team in by_week_a.items() if week in late_weeks}
    late_teams_b = {team for week, team in by_week_b.items() if week in late_weeks}
    late_team_overlap = _jaccard(late_teams_a, late_teams_b)
    late_season_overlap = max(late_ordered_overlap, late_team_overlap)

    elite_a = teams_a & elite
    elite_b = teams_b & elite
    elite_team_overlap = _jaccard(elite_a, elite_b)
    late_elite_a = late_teams_a & elite
    late_elite_b = late_teams_b & elite
    late_elite_overlap = _jaccard(late_elite_a, late_elite_b)

    similarity = (
        0.25 * exact_team_overlap
        + 0.25 * ordered_overlap
        + 0.35 * late_season_overlap
        + 0.15 * max(elite_team_overlap, late_elite_overlap)
    )
    shared_late_elites = len(late_elite_a & late_elite_b)
    if shared_late_elites >= 2:
        similarity += min(0.10, 0.025 * shared_late_elites) * min(
            max(float(late_season_overlap_weight) / 2.0, 0.5),
            2.0,
        )

    return {
        "similarity_score": _clip01(similarity),
        "exact_team_overlap": _clip01(exact_team_overlap),
        "ordered_overlap": _clip01(ordered_overlap),
        "late_season_overlap": _clip01(late_season_overlap),
        "late_ordered_overlap": _clip01(late_ordered_overlap),
        "late_team_overlap": _clip01(late_team_overlap),
        "elite_team_overlap": _clip01(elite_team_overlap),
        "late_elite_team_overlap": _clip01(late_elite_overlap),
    }


def estimate_path_popularity(
    candidate_optimal_paths: pd.DataFrame,
    projected_ownership_by_week: pd.DataFrame | None = None,
    *,
    pool_size: int = 5000,
    clustering_strength: float = 0.0,
    elite_path_bias: float = 1.0,
    late_season_overlap_weight: float = DEFAULT_LATE_SEASON_OVERLAP_WEIGHT,
    path_convergence_temperature: float = DEFAULT_PATH_CONVERGENCE_TEMPERATURE,
) -> pd.DataFrame:
    """Estimate route-level popularity for candidate paths.

    This is not a replacement for public-field simulation.  It is a route
    ownership prior that uses weekly ownership, late-week weighting, and elite
    team concentration to estimate how many entries may land on an obvious
    future path archetype.
    """

    candidates = _as_dataframe(candidate_optimal_paths)
    if candidates.empty:
        return pd.DataFrame()
    if {"week", "team"} - set(candidates.columns):
        raise ValueError("candidate_optimal_paths needs week and team columns.")

    projections = _ownership_lookup(projected_ownership_by_week)
    rows: list[dict[str, Any]] = []
    for path_id, path in _candidate_path_groups(candidates):
        ordered = _normalize_path(path)
        if ordered.empty:
            continue
        ownership = _ownership_for_path(ordered, projections)
        weights = _late_weights_for_path(ordered, late_season_overlap_weight)
        log_ownership = np.log(np.clip(ownership, 1e-6, 1.0))
        independent_probability = float(np.exp(log_ownership.sum()))
        geometric_ownership = float(np.exp(log_ownership.mean()))
        late_geometric_ownership = float(
            np.exp(np.average(log_ownership, weights=weights)),
        )
        elite_score = _elite_path_score(ordered, ownership)
        convergence_pressure = _clip01(
            0.55 * late_geometric_ownership
            + 0.30 * elite_score
            + 0.15 * geometric_ownership,
        )
        temperature = max(float(path_convergence_temperature), 0.05)
        clustered_probability = independent_probability ** temperature
        clustered_probability *= 1.0 + float(clustering_strength) * (
            convergence_pressure + float(elite_path_bias) * elite_score
        )
        clustered_probability = _clip01(clustered_probability)
        expected_duplicate_count = float(pool_size) * clustered_probability
        expected_identical_count = float(pool_size) * independent_probability
        rows.append(
            {
                "path_id": path_id,
                "path": _format_path_key(ordered),
                "path_key": _format_path_key(ordered),
                "late_path_key": _format_path_key(_late_slice(ordered)),
                "independent_path_probability": independent_probability,
                "clustered_path_popularity": clustered_probability,
                "expected_duplicate_path_count": expected_duplicate_count,
                "expected_identical_path_count": expected_identical_count,
                "path_clustering_score": _cluster_score(expected_duplicate_count, pool_size),
                "late_season_congestion_score": convergence_pressure,
                "cluster_adjusted_uniqueness_score": 1.0
                / (1.0 + expected_duplicate_count),
                "average_weekly_ownership": float(np.mean(ownership)),
                "late_weighted_ownership": late_geometric_ownership,
                "elite_path_score": elite_score,
            },
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["expected_duplicate_path_count", "late_season_congestion_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def cluster_public_paths(
    simulated_public_paths: pd.DataFrame | Any,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    late_season_overlap_weight: float = DEFAULT_LATE_SEASON_OVERLAP_WEIGHT,
    late_window: int = DEFAULT_LATE_WINDOW,
) -> pd.DataFrame:
    """Cluster simulated public paths into crowded future-route archetypes."""

    summaries = _public_path_summaries(simulated_public_paths, late_window=late_window)
    if not summaries:
        return pd.DataFrame()

    exact = pd.DataFrame(
        [
            {
                "path_id": summary.path_id,
                "path_key": summary.path_key,
                "late_path_key": summary.late_path_key,
                "expected_entries": summary.weight,
                "expected_surviving_entries": summary.weight
                if summary.survived_full_path
                else 0.0,
                "observed_path_count": 1,
                "rows": summary.rows,
            }
            for summary in summaries
        ],
    )

    exact_grouped = (
        exact.groupby(["path_key", "late_path_key"], sort=False)
        .agg(
            expected_entries=("expected_entries", "sum"),
            expected_surviving_entries=("expected_surviving_entries", "sum"),
            observed_path_count=("observed_path_count", "sum"),
        )
        .reset_index()
    )
    total_entries = float(exact_grouped["expected_entries"].sum())
    if total_entries <= 0:
        total_entries = 1.0

    rows: list[dict[str, Any]] = []
    for cluster_number, (late_key, late_group) in enumerate(
        exact_grouped.groupby("late_path_key", sort=False),
        start=1,
    ):
        representative = late_group.sort_values(
            ["expected_entries", "path_key"],
            ascending=[False, True],
        ).iloc[0]
        expected_entries = float(late_group["expected_entries"].sum())
        expected_survivors = float(late_group["expected_surviving_entries"].sum())
        rows.append(
            {
                "cluster_id": cluster_number,
                "path_key": str(representative["path_key"]),
                "late_path_key": str(late_key),
                "expected_entries": expected_entries,
                "expected_surviving_entries": expected_survivors,
                "observed_path_count": int(late_group["observed_path_count"].sum()),
                "member_path_count": int(len(late_group)),
                "cluster_share": expected_entries / total_entries,
                "surviving_cluster_share": expected_survivors / total_entries,
                "late_season_congestion_score": _clip01(expected_entries / total_entries),
                "similarity_threshold": float(similarity_threshold),
                "late_season_overlap_weight": float(late_season_overlap_weight),
            },
        )

    return pd.DataFrame(rows).sort_values(
        ["expected_entries", "member_path_count", "late_path_key"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def estimate_future_path_overlap(
    simulated_public_paths: pd.DataFrame | Any,
    candidate_optimal_paths: pd.DataFrame,
    projected_ownership_by_week: pd.DataFrame | None = None,
    *,
    pool_size: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    clustering_strength: float = 0.0,
    elite_path_bias: float = 1.0,
    late_season_overlap_weight: float = DEFAULT_LATE_SEASON_OVERLAP_WEIGHT,
    path_convergence_temperature: float = DEFAULT_PATH_CONVERGENCE_TEMPERATURE,
) -> dict[str, Any]:
    """Estimate future path overlap between public paths and candidates."""

    candidates = _as_dataframe(candidate_optimal_paths)
    if candidates.empty:
        return {
            "candidate_overlap": pd.DataFrame(),
            "path_clusters": pd.DataFrame(),
            "overlap_statistics": {},
        }

    summaries = _public_path_summaries(simulated_public_paths)
    if not summaries:
        popularity = estimate_path_popularity(
            candidates,
            projected_ownership_by_week,
            pool_size=int(pool_size or 0),
            clustering_strength=clustering_strength,
            elite_path_bias=elite_path_bias,
            late_season_overlap_weight=late_season_overlap_weight,
            path_convergence_temperature=path_convergence_temperature,
        )
        return {
            "candidate_overlap": popularity,
            "path_clusters": pd.DataFrame(),
            "overlap_statistics": _overlap_statistics(popularity),
        }

    clusters = cluster_public_paths(
        simulated_public_paths,
        similarity_threshold=similarity_threshold,
        late_season_overlap_weight=late_season_overlap_weight,
    )
    total_public_entries = float(sum(summary.weight for summary in summaries))
    if pool_size is None:
        pool_size = int(round(total_public_entries))

    candidate_rows: list[dict[str, Any]] = []
    for path_id, path in _candidate_path_groups(candidates):
        ordered = _normalize_path(path)
        if ordered.empty:
            continue
        similarities = []
        exact_entries = 0.0
        exact_survivors = 0.0
        near_entries = 0.0
        near_survivors = 0.0
        weighted_similarity = 0.0
        weighted_late_similarity = 0.0
        for summary in summaries:
            similarity = compute_path_similarity(
                ordered,
                summary.rows,
                late_season_overlap_weight=late_season_overlap_weight,
            )
            score = float(similarity["similarity_score"])
            late_score = float(similarity["late_season_overlap"])
            weight = float(summary.weight)
            weighted_similarity += score * weight
            weighted_late_similarity += late_score * weight
            similarities.append(score)
            if summary.path_key == _format_path_key(ordered):
                exact_entries += weight
                if summary.survived_full_path:
                    exact_survivors += weight
            if score >= float(similarity_threshold):
                near_entries += weight
                if summary.survived_full_path:
                    near_survivors += weight
        denominator = max(total_public_entries, 1.0)
        duplicate_count = near_entries
        candidate_rows.append(
            {
                "path_id": path_id,
                "path": _format_path_key(ordered),
                "path_key": _format_path_key(ordered),
                "late_path_key": _format_path_key(_late_slice(ordered)),
                "expected_duplicate_path_count": duplicate_count,
                "expected_identical_path_count": exact_entries,
                "expected_identical_path_survivors": exact_survivors,
                "expected_near_identical_path_survivors": near_survivors,
                "path_clustering_score": _cluster_score(duplicate_count, int(pool_size)),
                "late_season_congestion_score": weighted_late_similarity / denominator,
                "average_public_path_similarity": weighted_similarity / denominator,
                "cluster_adjusted_uniqueness_score": 1.0 / (1.0 + duplicate_count),
                "similarity_threshold": float(similarity_threshold),
                "max_public_path_similarity": max(similarities) if similarities else 0.0,
            },
        )

    candidate_overlap = pd.DataFrame(candidate_rows)
    if not candidate_overlap.empty:
        candidate_overlap = candidate_overlap.sort_values(
            ["expected_duplicate_path_count", "late_season_congestion_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return {
        "candidate_overlap": candidate_overlap,
        "path_clusters": clusters,
        "overlap_statistics": _overlap_statistics(candidate_overlap),
    }


def _public_path_summaries(
    simulated_public_paths: pd.DataFrame | Any,
    *,
    late_window: int = DEFAULT_LATE_WINDOW,
) -> list[_PathSummary]:
    if hasattr(simulated_public_paths, "entry_paths"):
        entry_paths = getattr(simulated_public_paths, "entry_paths")
        if isinstance(entry_paths, pd.DataFrame) and not entry_paths.empty:
            return _public_path_summaries(entry_paths, late_window=late_window)
        if hasattr(simulated_public_paths, "pick_matrix"):
            return _matrix_public_path_summaries(
                simulated_public_paths,
                late_window=late_window,
            )
        return []

    paths = _as_dataframe(simulated_public_paths)
    if paths.empty:
        return []
    if {"week", "team"} - set(paths.columns):
        raise ValueError("simulated_public_paths needs week and team columns.")

    group_columns = _public_group_columns(paths)
    simulation_count = (
        int(paths["simulation"].nunique())
        if "simulation" in paths.columns
        else 1
    )
    summaries: list[_PathSummary] = []
    for path_id, group in paths.groupby(group_columns, sort=False):
        ordered = _normalize_path(group)
        if ordered.empty:
            continue
        entry_weight = (
            float(pd.to_numeric(group["entry_weight"], errors="coerce").dropna().iloc[0])
            if "entry_weight" in group.columns
            and not pd.to_numeric(group["entry_weight"], errors="coerce").dropna().empty
            else 1.0
        )
        normalized_weight = entry_weight / max(float(simulation_count), 1.0)
        survived = _survived_full_path(group)
        summaries.append(
            _PathSummary(
                path_id=path_id,
                path_key=_format_path_key(ordered),
                late_path_key=_format_path_key(_late_slice(ordered, late_window=late_window)),
                teams=tuple(ordered["team"].astype(str).tolist()),
                weeks=tuple(ordered["week"].astype(int).tolist()),
                weight=normalized_weight,
                survived_full_path=survived,
                rows=ordered,
            ),
        )
    return summaries


def _matrix_public_path_summaries(
    public_field: Any,
    *,
    late_window: int,
) -> list[_PathSummary]:
    pick_matrix = np.asarray(getattr(public_field, "pick_matrix"))
    if pick_matrix.size == 0:
        return []
    weeks = [int(week) for week in getattr(public_field, "weeks")]
    teams = [str(team) for team in getattr(public_field, "teams")]
    pick_win_matrix = np.asarray(getattr(public_field, "pick_win_matrix"))
    entry_weight = float(getattr(public_field, "entry_weight", 1.0))
    simulations, sample_entries, _ = pick_matrix.shape
    normalized_weight = entry_weight / max(float(simulations), 1.0)
    summaries: list[_PathSummary] = []
    for simulation_index in range(simulations):
        for entry_index in range(sample_entries):
            codes = pick_matrix[simulation_index, entry_index, :]
            rows = []
            for week, code in zip(weeks, codes, strict=False):
                if int(code) < 0:
                    continue
                rows.append({"week": int(week), "team": teams[int(code)]})
            ordered = _normalize_path(rows)
            if ordered.empty:
                continue
            survived = bool(pick_win_matrix[simulation_index, entry_index, : len(ordered)].all())
            summaries.append(
                _PathSummary(
                    path_id=(simulation_index + 1, entry_index + 1),
                    path_key=_format_path_key(ordered),
                    late_path_key=_format_path_key(_late_slice(ordered, late_window=late_window)),
                    teams=tuple(ordered["team"].astype(str).tolist()),
                    weeks=tuple(ordered["week"].astype(int).tolist()),
                    weight=normalized_weight,
                    survived_full_path=survived,
                    rows=ordered,
                ),
            )
    return summaries


def _normalize_path(
    path: pd.DataFrame | list[dict[str, Any]] | list[str],
) -> pd.DataFrame:
    if isinstance(path, list) and path and all(isinstance(item, str) for item in path):
        rows = [
            {"week": index + 1, "team": str(team)}
            for index, team in enumerate(path)
        ]
        df = pd.DataFrame(rows)
    else:
        df = _as_dataframe(path)
    if df.empty:
        return pd.DataFrame(columns=["week", "team"])
    if "team" not in df.columns:
        raise ValueError("path data needs a team column.")
    output = df.copy()
    if "week" not in output.columns:
        output["week"] = range(1, len(output) + 1)
    output["week"] = pd.to_numeric(output["week"], errors="raise").astype(int)
    output["team"] = output["team"].astype(str)
    output = output[output["team"].notna() & (output["team"].astype(str) != "")]
    return output.sort_values("week").reset_index(drop=True)


def _candidate_path_groups(paths: pd.DataFrame) -> list[tuple[Any, pd.DataFrame]]:
    if "path_id" in paths.columns:
        return [
            (path_id, group.copy())
            for path_id, group in paths.groupby("path_id", sort=True)
        ]
    return [(1, paths.copy())]


def _public_group_columns(paths: pd.DataFrame) -> list[str]:
    if {"simulation", "entry_id"}.issubset(paths.columns):
        return ["simulation", "entry_id"]
    if "path_id" in paths.columns:
        return ["path_id"]
    return ["_single_path_id"] if "_single_path_id" in paths.columns else _with_single_path(paths)


def _with_single_path(paths: pd.DataFrame) -> list[str]:
    paths["_single_path_id"] = 1
    return ["_single_path_id"]


def _survived_full_path(group: pd.DataFrame) -> bool:
    if "alive_after" in group.columns:
        ordered = group.sort_values("week")
        values = ordered["alive_after"].dropna()
        if not values.empty:
            return bool(values.iloc[-1])
    if "won" in group.columns:
        wins = group["won"].dropna()
        if not wins.empty:
            return bool(wins.astype(bool).all())
    return True


def _ownership_lookup(
    projected_ownership_by_week: pd.DataFrame | None,
) -> dict[tuple[int, str], float]:
    projections = _as_dataframe(projected_ownership_by_week)
    if projections.empty:
        return {}
    ownership_column = _ownership_column(projections)
    if ownership_column is None:
        return {}
    result: dict[tuple[int, str], float] = {}
    for row in projections.to_dict("records"):
        result[(int(row["week"]), str(row["team"]))] = _clip01(float(row[ownership_column]))
    return result


def _ownership_column(df: pd.DataFrame) -> str | None:
    for column in [
        "projected_public_pick_pct",
        "public_pick_pct",
        "projected_ownership_pct",
        "pick_probability",
    ]:
        if column in df.columns:
            return column
    return None


def _ownership_for_path(
    path: pd.DataFrame,
    projections: dict[tuple[int, str], float],
) -> np.ndarray:
    values = []
    own_column = _ownership_column(path)
    for row in path.to_dict("records"):
        if own_column is not None and not pd.isna(row.get(own_column)):
            value = float(row[own_column])
        else:
            value = projections.get((int(row["week"]), str(row["team"])), 0.0)
        values.append(max(_clip01(value), 1e-6))
    return np.asarray(values, dtype=float)


def _late_weights_for_path(
    path: pd.DataFrame,
    late_season_overlap_weight: float,
) -> np.ndarray:
    weeks = path["week"].astype(int).tolist()
    weights_by_week = _late_week_weights(weeks, late_season_overlap_weight)
    return np.asarray([weights_by_week[int(week)] for week in weeks], dtype=float)


def _late_week_weights(
    weeks: list[int] | tuple[int, ...],
    late_season_overlap_weight: float,
) -> dict[int, float]:
    ordered = sorted(set(int(week) for week in weeks))
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    late_weight = max(float(late_season_overlap_weight), 1.0)
    return {
        week: 1.0 + (late_weight - 1.0) * index / float(len(ordered) - 1)
        for index, week in enumerate(ordered)
    }


def _late_weeks(weeks: list[int] | tuple[int, ...]) -> set[int]:
    ordered = sorted(set(int(week) for week in weeks))
    if not ordered:
        return set()
    start = max(0, len(ordered) // 2)
    return set(ordered[start:])


def _late_slice(path: pd.DataFrame, *, late_window: int = DEFAULT_LATE_WINDOW) -> pd.DataFrame:
    if path.empty:
        return path.copy()
    return path.sort_values("week").tail(max(int(late_window), 1)).copy()


def _elite_path_score(path: pd.DataFrame, ownership: np.ndarray) -> float:
    if path.empty:
        return 0.0
    win_probability = (
        pd.to_numeric(path["win_probability"], errors="coerce").fillna(0.0).to_numpy()
        if "win_probability" in path.columns
        else np.zeros(len(path), dtype=float)
    )
    ownership = np.asarray(ownership, dtype=float)
    elite_signal = np.maximum(win_probability - 0.60, 0.0) / 0.30
    ownership_signal = np.maximum(ownership - 0.15, 0.0) / 0.35
    combined = np.clip(0.65 * elite_signal + 0.35 * ownership_signal, 0.0, 1.0)
    weights = _late_weights_for_path(path, DEFAULT_LATE_SEASON_OVERLAP_WEIGHT)
    return _clip01(float(np.average(combined, weights=weights)))


def _resolve_elite_teams(
    paths: list[pd.DataFrame],
    *,
    elite_teams: set[str] | list[str] | tuple[str, ...] | None,
    elite_win_probability: float,
) -> set[str]:
    if elite_teams is not None:
        return {str(team) for team in elite_teams}
    teams: set[str] = set()
    for path in paths:
        if "win_probability" not in path.columns:
            continue
        mask = pd.to_numeric(path["win_probability"], errors="coerce").fillna(0) >= float(
            elite_win_probability,
        )
        teams.update(path.loc[mask, "team"].astype(str).tolist())
    return teams


def _overlap_statistics(candidate_overlap: pd.DataFrame) -> dict[str, Any]:
    if candidate_overlap.empty:
        return {
            "expected_duplicate_path_count": 0.0,
            "expected_identical_path_survivors": 0.0,
            "path_clustering_score": 0.0,
            "late_season_congestion_score": 0.0,
        }
    first = candidate_overlap.iloc[0]
    return {
        "expected_duplicate_path_count": _float(first.get("expected_duplicate_path_count")),
        "expected_identical_path_survivors": _float(
            first.get("expected_identical_path_survivors"),
        ),
        "path_clustering_score": _float(first.get("path_clustering_score")),
        "late_season_congestion_score": _float(first.get("late_season_congestion_score")),
    }


def _cluster_score(duplicate_count: float, pool_size: int) -> float:
    scale = max(np.sqrt(max(float(pool_size), 1.0)), 1.0)
    return _clip01(1.0 - float(np.exp(-max(float(duplicate_count), 0.0) / scale)))


def _format_path_key(path: pd.DataFrame) -> str:
    if path.empty:
        return ""
    return " > ".join(
        f"W{int(row['week'])}:{row['team']}" for row in path.sort_values("week").to_dict("records")
    )


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    if not union:
        return 0.0
    return float(len(first & second)) / float(len(union))


def _empty_similarity() -> dict[str, float]:
    return {
        "similarity_score": 0.0,
        "exact_team_overlap": 0.0,
        "ordered_overlap": 0.0,
        "late_season_overlap": 0.0,
        "late_ordered_overlap": 0.0,
        "late_team_overlap": 0.0,
        "elite_team_overlap": 0.0,
        "late_elite_team_overlap": 0.0,
    }


def _clip01(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _as_dataframe(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)
