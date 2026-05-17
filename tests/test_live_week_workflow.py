import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_live_week as live_week_cli
from survivor.live_week import (
    LiveWeekOptions,
    LiveWeekWorkflowError,
    StepStatus,
    build_operator_summary,
    run_live_week,
)
from survivor.odds_ingestion import normalize_odds_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "data" / "sample" / "real_week_1"


def test_successful_live_week_workflow_uses_real_week_fixture(tmp_path):
    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=FIXTURE_DIR,
            reports_dir=tmp_path / "reports",
            simulations=5,
            entries=40,
        ),
    )

    assert "NFL Survivor Week 1 Summary" in result.summary_text
    assert "Schedule: OK (16 games)" in result.summary_text
    assert "Odds: OK (16 games, 1 sportsbook, local)" in result.summary_text
    assert "Public Picks: OK (1 source, local)" in result.summary_text
    assert "Path EV: skipped" in result.summary_text
    assert result.rankings_report_path.exists()
    assert result.simulation_report_path.exists()
    assert result.portfolio_report_path.exists()
    assert result.path_ev_report_path is None
    assert result.summary_report_path.exists()


def test_live_week_workflow_runs_path_ev_when_enabled(tmp_path):
    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=FIXTURE_DIR,
            reports_dir=tmp_path / "reports",
            simulations=3,
            entries=40,
            run_path_ev=True,
            path_ev_simulations=50,
            beam_width=8,
            top_k=3,
            entry_fee=10,
            prize_pool=50000,
        ),
    )

    assert "Path EV: generated" in result.summary_text
    assert "Single-Entry Path EV:" in result.summary_text
    assert "Best Pick:" in result.summary_text
    assert "EV Dollars:" in result.summary_text
    assert "EV Multiple:" in result.summary_text
    assert result.path_ev_result is not None
    assert result.path_ev_report_path.exists()
    assert result.path_ev_report_path.name == "week_1_single_entry_path_ev.md"


def test_live_week_workflow_skips_path_ev_without_flag(tmp_path):
    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=FIXTURE_DIR,
            reports_dir=tmp_path / "reports",
            simulations=3,
            entries=40,
            run_path_ev=False,
        ),
    )

    assert "Path EV: skipped" in result.summary_text
    assert "skipped (use --run-path-ev)" in result.summary_text
    assert result.path_ev_result is None
    assert result.path_ev_report_path is None


def test_missing_odds_failure_is_actionable(tmp_path):
    workspace = _copy_fixture(tmp_path)
    (workspace / "odds.csv").unlink()

    with pytest.raises(LiveWeekWorkflowError, match="Missing odds file"):
        run_live_week(
            LiveWeekOptions(
                season=2026,
                week=1,
                data_dir=workspace,
                reports_dir=tmp_path / "reports",
                simulations=2,
            ),
        )


def test_missing_public_picks_failure_is_actionable(tmp_path):
    workspace = _copy_fixture(tmp_path)
    (workspace / "public_picks.csv").unlink()

    with pytest.raises(LiveWeekWorkflowError, match="Missing public picks file"):
        run_live_week(
            LiveWeekOptions(
                season=2026,
                week=1,
                data_dir=workspace,
                reports_dir=tmp_path / "reports",
                simulations=2,
            ),
        )


def test_public_pick_import_writes_consensus_for_workflow(tmp_path):
    workspace = _copy_fixture(tmp_path)
    (workspace / "public_picks.csv").unlink()
    source_dir = PROJECT_ROOT / "data" / "sample" / "public_pick_sources"

    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=workspace,
            reports_dir=tmp_path / "reports",
            simulations=2,
            public_pick_inputs=(
                source_dir / "yahoo_week1_public_picks.csv",
                source_dir / "espn_week1_public_picks.csv",
                source_dir / "survivorgrid_week1_public_picks.csv",
            ),
            public_pick_format="csv",
        ),
    )

    imported = pd.read_csv(workspace / "public_picks.csv")
    assert imported["source_count"].max() == 3
    assert "Public Picks: OK (3 sources, imported)" in result.summary_text


def test_dry_run_calculates_without_writing_reports(tmp_path):
    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=FIXTURE_DIR,
            reports_dir=tmp_path / "reports",
            simulations=3,
            dry_run=True,
        ),
    )

    assert "Mode: DRY RUN" in result.summary_text
    assert result.rankings_report_path is None
    assert result.simulation_report_path is None
    assert result.portfolio_report_path is None
    assert result.summary_report_path is None
    assert not (tmp_path / "reports").exists()


def test_workflow_summary_generation_mentions_top_pick_and_exposure():
    rankings = pd.DataFrame(
        [
            {
                "rank": 1,
                "team": "KC",
                "opponent": "LV",
                "no_vig_win_probability": 0.784,
                "public_pick_pct": 0.312,
                "leverage_score": 0.123,
            }
        ],
    )
    exposure = pd.DataFrame(
        [
            {"team": "KC", "entries_allocated": 14},
            {"team": "BUF", "entries_allocated": 10},
        ],
    )

    summary = build_operator_summary(
        options=LiveWeekOptions(week=1, simulations=10000, entries=40),
        statuses=(StepStatus("Schedule", "OK"),),
        rankings=rankings,
        exposure=exposure,
        report_paths={"rankings": Path("outputs/reports/week_1_report.md")},
    )

    assert "KC over LV" in summary
    assert "Win Prob: 78.4%" in summary
    assert "Consensus Ownership: 31.2%" in summary
    assert "KC: 14" in summary


def test_week_summary_report_links_generated_reports(tmp_path):
    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=FIXTURE_DIR,
            reports_dir=tmp_path / "reports",
            simulations=3,
        ),
    )

    report = result.summary_report_path.read_text(encoding="utf-8")
    assert "[Rankings](week_1_report.md)" in report
    assert "[Simulation](week_1_simulation_report.md)" in report
    assert "[Portfolio](week_1_portfolio_report.md)" in report


def test_week_summary_report_includes_path_ev_metrics(tmp_path):
    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=FIXTURE_DIR,
            reports_dir=tmp_path / "reports",
            simulations=3,
            entries=40,
            run_path_ev=True,
            path_ev_simulations=50,
            beam_width=8,
            top_k=3,
            entry_fee=10,
            prize_pool=50000,
        ),
    )

    report = result.summary_report_path.read_text(encoding="utf-8")
    assert "## Single-Entry Path EV" in report
    assert "- Path EV estimate:" in report
    assert "- EV dollars:" in report
    assert "- EV multiple:" in report
    assert "- Expected edge:" in report
    assert "- Comparison to heuristic top pick:" in report
    assert "[Single-Entry Path EV](week_1_single_entry_path_ev.md)" in report


def test_api_refresh_skip_does_not_call_provider(tmp_path):
    workspace = _copy_fixture(tmp_path)

    class ExplodingProvider:
        def __init__(self, **_kwargs):
            raise AssertionError("provider should not be constructed")

    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=workspace,
            reports_dir=tmp_path / "reports",
            simulations=2,
            refresh_odds=False,
        ),
        odds_provider_factory=ExplodingProvider,
    )

    assert "Odds: OK" in result.summary_text


def test_mocked_api_refresh_writes_normalized_odds(tmp_path):
    workspace = _copy_fixture(tmp_path)
    (workspace / "odds.csv").unlink()
    captured = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def normalize(self, schedule_df):
            return normalize_odds_records(_raw_odds_records(), schedule_df)

    result = run_live_week(
        LiveWeekOptions(
            season=2026,
            week=1,
            data_dir=workspace,
            reports_dir=tmp_path / "reports",
            simulations=2,
            refresh_odds=True,
            sportsbooks=("fanduel", "draftkings"),
            odds_regions=("us",),
            markets=("h2h",),
        ),
        odds_provider_factory=FakeProvider,
    )

    refreshed = pd.read_csv(workspace / "odds.csv")
    assert captured["bookmakers"] == "fanduel,draftkings"
    assert set(refreshed["market_type"]) == {"h2h"}
    assert "refreshed" in result.summary_text


def test_cli_smoke_runs_live_week(tmp_path, capsys):
    result = live_week_cli.main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--data-dir",
            str(FIXTURE_DIR),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--simulations",
            "2",
            "--entries",
            "40",
            "--run-path-ev",
            "--path-ev-simulations",
            "20",
            "--beam-width",
            "8",
            "--top-k",
            "3",
            "--entry-fee",
            "10",
            "--prize-pool",
            "50000",
        ],
    )

    stdout = capsys.readouterr().out
    assert result == 0
    assert "NFL Survivor Week 1 Summary" in stdout
    assert "Path EV: generated" in stdout


def _copy_fixture(tmp_path: Path) -> Path:
    workspace = tmp_path / "real_week_1"
    shutil.copytree(FIXTURE_DIR, workspace)
    return workspace


def _raw_odds_records() -> list[dict[str, object]]:
    schedule = pd.read_csv(FIXTURE_DIR / "schedule.csv")
    odds = []
    for row in schedule.to_dict("records"):
        odds.append(
            {
                "season": 2026,
                "week": int(row["week"]),
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_moneyline": -150,
                "away_moneyline": 130,
                "sportsbook": "fakebook",
                "pulled_at": "2026-09-01T12:00:00Z",
                "source": "test",
            },
        )
    return odds
