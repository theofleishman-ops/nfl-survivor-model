import pandas as pd
import pandas.testing as pdt
import pytest

from survivor.path_clustering import (
    cluster_public_paths,
    compute_path_similarity,
    estimate_path_popularity,
)
from survivor.path_ev import evaluate_path_ev
from survivor.public_behavior import PublicBehaviorConfig
from survivor.public_field import simulate_public_field_paths


def test_similar_late_paths_cluster_together():
    public_paths = pd.DataFrame(
        [
            *_entry_path(1, 1, ["A", "B", "KC", "BUF", "PHI", "DET"]),
            *_entry_path(1, 2, ["C", "D", "KC", "BUF", "PHI", "DET"]),
            *_entry_path(1, 3, ["E", "F", "SEA", "MIN", "GB", "BAL"]),
        ],
    )

    clusters = cluster_public_paths(public_paths, late_window=4)

    crowded = clusters[clusters["late_path_key"].str.contains("KC")]
    assert not crowded.empty
    assert int(crowded.iloc[0]["member_path_count"]) == 2
    assert float(crowded.iloc[0]["expected_entries"]) == pytest.approx(2.0)


def test_late_overlap_matters_more_than_early_overlap():
    base = _candidate_path(["A", "B", "KC", "BUF", "PHI", "DET"])
    early_match = _candidate_path(["A", "B", "LAR", "MIA", "GB", "BAL"])
    late_match = _candidate_path(["LAR", "MIA", "KC", "BUF", "PHI", "DET"])

    early = compute_path_similarity(base, early_match, elite_teams={"KC", "BUF", "PHI", "DET"})
    late = compute_path_similarity(base, late_match, elite_teams={"KC", "BUF", "PHI", "DET"})

    assert late["late_season_overlap"] > early["late_season_overlap"]
    assert late["similarity_score"] > early["similarity_score"]


def test_popular_elite_paths_become_crowded():
    candidates = pd.concat(
        [
            _candidate_path(["KC", "BUF", "PHI", "DET"], path_id=1, ownership=0.32),
            _candidate_path(["TEN", "CAR", "NYG", "ARI"], path_id=2, ownership=0.04),
        ],
        ignore_index=True,
    )

    popularity = estimate_path_popularity(
        candidates,
        candidates[["week", "team", "projected_public_pick_pct"]],
        pool_size=5000,
        clustering_strength=0.4,
        elite_path_bias=1.5,
        path_convergence_temperature=0.75,
    ).set_index("path_id")

    assert popularity.loc[1, "expected_duplicate_path_count"] > popularity.loc[
        2,
        "expected_duplicate_path_count",
    ]
    assert popularity.loc[1, "path_clustering_score"] > popularity.loc[
        2,
        "path_clustering_score",
    ]


def test_clustering_reduces_uniqueness_and_ev():
    path = pd.DataFrame(
        [
            {"week": 1, "team": "A"},
            {"week": 2, "team": "B"},
            {"week": 3, "team": "C"},
            {"week": 4, "team": "D"},
        ],
    )
    no_cluster = evaluate_path_ev(
        path,
        _schedule(),
        _odds(),
        _public_picks(),
        pool_size=300,
        simulations=160,
        random_seed=18,
        public_field_sample_size=120,
        public_behavior_config=_behavior("no_cluster", clustering_strength=0.0),
    )
    clustered = evaluate_path_ev(
        path,
        _schedule(),
        _odds(),
        _public_picks(),
        pool_size=300,
        simulations=160,
        random_seed=18,
        public_field_sample_size=120,
        public_behavior_config=_behavior(
            "clustered",
            clustering_strength=0.85,
            elite_path_bias=2.0,
            late_season_overlap_weight=3.0,
            path_convergence_temperature=0.65,
        ),
    )

    assert clustered.path_ev < no_cluster.path_ev
    assert clustered.diagnostics["cluster_adjusted_uniqueness_score"] < no_cluster.diagnostics[
        "cluster_adjusted_uniqueness_score"
    ]
    assert clustered.diagnostics["expected_duplicate_path_count"] > 0


def test_path_clustering_seed_behavior_is_deterministic():
    kwargs = {
        "schedule_df": pd.DataFrame(),
        "odds_df": pd.DataFrame(),
        "public_picks_df": pd.DataFrame(),
        "team_probabilities_df": _probabilities(),
        "start_week": 1,
        "pool_size": 120,
        "simulations": 30,
        "random_seed": 2026,
        "public_field_sample_size": 80,
        "behavior_config": _behavior(
            "clustered",
            clustering_strength=0.7,
            elite_path_bias=1.8,
            late_season_overlap_weight=2.8,
            path_convergence_temperature=0.7,
        ),
        "include_entry_paths": False,
    }

    first = simulate_public_field_paths(**kwargs)
    second = simulate_public_field_paths(**kwargs)

    pdt.assert_frame_equal(first.path_clusters, second.path_clusters)
    pdt.assert_frame_equal(first.path_convergence, second.path_convergence)
    assert (first.pick_matrix == second.pick_matrix).all()


def _entry_path(simulation: int, entry_id: int, teams: list[str]) -> list[dict[str, object]]:
    return [
        {
            "simulation": simulation,
            "entry_id": entry_id,
            "week": week,
            "team": team,
            "entry_weight": 1.0,
            "won": True,
            "alive_after": True,
        }
        for week, team in enumerate(teams, start=1)
    ]


def _candidate_path(
    teams: list[str],
    *,
    path_id: int = 1,
    ownership: float = 0.20,
) -> pd.DataFrame:
    rows = []
    for week, team in enumerate(teams, start=1):
        rows.append(
            {
                "path_id": path_id,
                "week": week,
                "team": team,
                "win_probability": 0.75 if team in {"KC", "BUF", "PHI", "DET"} else 0.58,
                "projected_public_pick_pct": ownership,
            },
        )
    return pd.DataFrame(rows)


def _behavior(name: str, **overrides: float) -> PublicBehaviorConfig:
    values = {
        "name": name,
        "description": name,
        "chalkiness": 4.0,
        "randomness": 0.0,
        "future_awareness": 0.0,
        "popularity_weight": 1.2,
        "scarcity_weight": 1.0,
        "contrarian_rate": 0.0,
        "max_single_team_ownership": 1.0,
        "ownership_temperature": 0.9,
        "clustering_strength": 0.0,
        "elite_path_bias": 1.0,
        "late_season_overlap_weight": 2.0,
        "path_convergence_temperature": 1.0,
    }
    values.update(overrides)
    return PublicBehaviorConfig(**values)


def _schedule() -> pd.DataFrame:
    rows = []
    for week, favorite, underdog in [
        (1, "A", "E"),
        (2, "B", "F"),
        (3, "C", "G"),
        (4, "D", "H"),
    ]:
        rows.append(
            {
                "week": week,
                "game_id": f"W{week}_{favorite}_{underdog}",
                "home_team": favorite,
                "away_team": underdog,
            },
        )
    return pd.DataFrame(rows)


def _odds() -> pd.DataFrame:
    rows = []
    for week, favorite, underdog, probability in [
        (1, "A", "E", 0.86),
        (2, "B", "F", 0.84),
        (3, "C", "G", 0.82),
        (4, "D", "H", 0.80),
    ]:
        game_id = f"W{week}_{favorite}_{underdog}"
        rows.append(_odds_row(week, game_id, favorite, underdog, probability))
        rows.append(_odds_row(week, game_id, underdog, favorite, 1 - probability))
    return pd.DataFrame(rows)


def _public_picks() -> pd.DataFrame:
    rows = []
    for week, favorite, underdog in [
        (1, "A", "E"),
        (2, "B", "F"),
        (3, "C", "G"),
        (4, "D", "H"),
    ]:
        rows.append({"week": week, "team": favorite, "public_pick_pct": 0.72})
        rows.append({"week": week, "team": underdog, "public_pick_pct": 0.28})
    return pd.DataFrame(rows)


def _probabilities() -> pd.DataFrame:
    return _odds().rename(columns={"no_vig_win_probability": "win_probability"})


def _odds_row(
    week: int,
    game_id: str,
    team: str,
    opponent: str,
    probability: float,
) -> dict[str, object]:
    return {
        "week": week,
        "game_id": game_id,
        "team": team,
        "opponent": opponent,
        "no_vig_win_probability": probability,
        "projected_public_pick_pct": 0.72 if probability > 0.5 else 0.28,
    }
