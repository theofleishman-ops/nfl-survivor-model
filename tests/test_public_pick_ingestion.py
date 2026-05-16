import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from survivor.public_pick_ingestion import (
    AGGREGATED_PUBLIC_PICK_COLUMNS,
    NORMALIZED_PUBLIC_PICK_COLUMNS,
    aggregate_public_pick_sources,
    load_public_picks_csv,
    load_public_picks_json,
    normalize_public_pick_records,
    validate_public_picks_against_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_csv_public_pick_normalization(tmp_path):
    raw_path = tmp_path / "public_picks_raw.csv"
    raw_path.write_text(
        "\n".join(
            [
                "season,week,team,source,public_pick_pct,sample_size,pulled_at",
                "2026,1,Philadelphia Eagles,yahoo,0.24,1000,2026-09-01T12:00:00Z",
            ],
        ),
        encoding="utf-8",
    )

    result = normalize_public_pick_records(load_public_picks_csv(raw_path), _schedule())

    assert list(result.columns) == list(NORMALIZED_PUBLIC_PICK_COLUMNS)
    assert result.loc[0, "team"] == "PHI"
    assert result.loc[0, "opponent"] == "DAL"
    assert result.loc[0, "game_id"] == "2026_W01_DAL_AT_PHI"
    assert result.loc[0, "public_pick_pct"] == pytest.approx(0.24)


def test_json_public_pick_normalization(tmp_path):
    raw_path = tmp_path / "public_picks_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "public_picks": [
                    {
                        "season": 2026,
                        "week": 1,
                        "team": "Kansas City Chiefs",
                        "provider": "espn",
                        "ownership_pct": "18%",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    result = normalize_public_pick_records(load_public_picks_json(raw_path), _schedule())

    assert result.loc[0, "team"] == "KC"
    assert result.loc[0, "opponent"] == "LAC"
    assert result.loc[0, "source"] == "espn"
    assert result.loc[0, "public_pick_pct"] == pytest.approx(0.18)


def test_team_alias_handling_and_schedule_join():
    result = normalize_public_pick_records(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "LA Chargers",
                "source": "survivorgrid",
                "public_pick_pct": 0.08,
            },
        ],
        _schedule(),
    )

    assert result.loc[0, "team"] == "LAC"
    assert result.loc[0, "opponent"] == "KC"
    assert result.loc[0, "game_id"] == "2026_W01_KC_AT_LAC"


def test_team_not_playing_that_week_fails():
    with pytest.raises(ValueError, match="not playing in week 2"):
        normalize_public_pick_records(
            [
                {
                    "season": 2026,
                    "week": 2,
                    "team": "KC",
                    "source": "yahoo",
                    "public_pick_pct": 0.10,
                },
            ],
            _schedule(),
        )


def test_percent_outside_zero_one_fails():
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_public_pick_records(
            [
                {
                    "season": 2026,
                    "week": 1,
                    "team": "KC",
                    "source": "yahoo",
                    "public_pick_pct": 1.20,
                },
            ],
            _schedule(),
        )


def test_source_weekly_total_above_one_fails():
    with pytest.raises(ValueError, match="totals cannot exceed 1.0"):
        normalize_public_pick_records(
            [
                {
                    "season": 2026,
                    "week": 1,
                    "team": "PHI",
                    "source": "yahoo",
                    "public_pick_pct": 0.60,
                },
                {
                    "season": 2026,
                    "week": 1,
                    "team": "KC",
                    "source": "yahoo",
                    "public_pick_pct": 0.50,
                },
            ],
            _schedule(),
        )


def test_partial_source_totals_below_one_are_allowed():
    result = normalize_public_pick_records(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "PHI",
                "source": "yahoo",
                "public_pick_pct": 0.25,
            },
            {
                "season": 2026,
                "week": 1,
                "team": "KC",
                "source": "yahoo",
                "public_pick_pct": 0.15,
            },
        ],
        _schedule(),
    )

    assert validate_public_picks_against_schedule(result, _schedule()) == []
    assert result["public_pick_pct"].sum() == pytest.approx(0.40)


def test_multi_source_aggregation_simple_average():
    normalized = normalize_public_pick_records(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "KC",
                "source": "yahoo",
                "public_pick_pct": 0.20,
            },
            {
                "season": 2026,
                "week": 1,
                "team": "KC",
                "source": "espn",
                "public_pick_pct": 0.40,
            },
        ],
        _schedule(),
    )

    aggregated = aggregate_public_pick_sources(normalized)

    assert list(aggregated.columns) == list(AGGREGATED_PUBLIC_PICK_COLUMNS)
    assert aggregated.loc[0, "consensus_public_pick_pct"] == pytest.approx(0.30)
    assert aggregated.loc[0, "public_pick_pct"] == pytest.approx(0.30)
    assert aggregated.loc[0, "source_count"] == 2
    assert aggregated.loc[0, "notes"] == "simple_average"


def test_sample_size_weighted_aggregation():
    normalized = normalize_public_pick_records(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "KC",
                "source": "yahoo",
                "public_pick_pct": 0.20,
                "sample_size": 100,
            },
            {
                "season": 2026,
                "week": 1,
                "team": "KC",
                "source": "espn",
                "public_pick_pct": 0.40,
                "sample_size": 300,
            },
        ],
        _schedule(),
    )

    aggregated = aggregate_public_pick_sources(normalized)

    assert aggregated.loc[0, "consensus_public_pick_pct"] == pytest.approx(0.35)
    assert aggregated.loc[0, "sample_size"] == 400
    assert aggregated.loc[0, "notes"] == "sample_size_weighted"


def test_cli_dry_run_smoke_test(tmp_path):
    schedule_path = tmp_path / "schedule.csv"
    raw_path = tmp_path / "public_picks_raw.csv"
    output_path = tmp_path / "public_picks.csv"
    _schedule().to_csv(schedule_path, index=False)
    raw_path.write_text(
        "\n".join(
            [
                "team,public_pick_pct",
                "KC,0.22",
            ],
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_public_picks.py",
            "--input",
            str(raw_path),
            "--format",
            "csv",
            "--season",
            "2026",
            "--week",
            "1",
            "--schedule",
            str(schedule_path),
            "--output",
            str(output_path),
            "--source",
            "yahoo",
            "--aggregate",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert output_path.exists() is False
    assert "Dry run complete" in result.stdout
    assert "Aggregated to 1 consensus public pick rows" in result.stdout
    assert "2026_W01_KC_AT_LAC" in result.stdout


def test_output_joins_to_schedule_game_id_and_opponent():
    result = normalize_public_pick_records(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "Cowboys",
                "source": "manual",
                "public_pick_pct": 0.04,
            },
        ],
        _schedule(),
    )

    row = result.iloc[0]
    assert row["team"] == "DAL"
    assert row["opponent"] == "PHI"
    assert row["game_id"] == "2026_W01_DAL_AT_PHI"


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_W01_DAL_AT_PHI",
                "away_team": "DAL",
                "home_team": "PHI",
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_W01_KC_AT_LAC",
                "away_team": "KC",
                "home_team": "LAC",
            },
        ],
    )
