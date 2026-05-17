"""Single-command live weekly survivor workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from survivor.config import PROJECT_ROOT
from survivor.config_runtime import (
    DEFAULT_RUNTIME_CONFIG,
    RuntimeConfig,
    csv_text,
)
from survivor.loaders import (
    load_entries_df,
    load_odds_df,
    load_public_picks_df,
    load_schedule_df,
)
from survivor.odds import add_no_vig_probabilities
from survivor.odds_ingestion import (
    NORMALIZED_ODDS_COLUMNS,
    aggregate_book_odds,
    validate_odds_against_schedule,
)
from survivor.odds_providers.the_odds_api import API_KEY_ENV_VAR, TheOddsAPIProvider
from survivor.optimizer import rank_weekly_picks
from survivor.portfolio import optimize_portfolio_for_week
from survivor.portfolio_reports import write_portfolio_report
from survivor.public_pick_ingestion import (
    aggregate_public_pick_sources,
    normalize_public_pick_records,
    validate_public_picks_against_schedule,
)
from survivor.public_pick_providers.manual import ManualPublicPickProvider
from survivor.reports import write_weekly_report
from survivor.schemas import (
    validate_odds_schedule_relationship,
    validate_public_picks_schedule_relationship,
)
from survivor.simulation_reports import write_simulation_report
from survivor.simulator import simulate_many_seasons


@dataclass(frozen=True)
class LiveWeekOptions:
    """Options for a live week workflow run."""

    week: int
    season: int = DEFAULT_RUNTIME_CONFIG.season
    data_dir: Path = DEFAULT_RUNTIME_CONFIG.data_dir
    reports_dir: Path = DEFAULT_RUNTIME_CONFIG.reports_dir
    simulations: int = DEFAULT_RUNTIME_CONFIG.simulation_count
    entries: int = DEFAULT_RUNTIME_CONFIG.entry_count
    aggression: str = DEFAULT_RUNTIME_CONFIG.aggression
    pool_size: int = DEFAULT_RUNTIME_CONFIG.pool_size
    seed: int = DEFAULT_RUNTIME_CONFIG.seed
    refresh_odds: bool = False
    sportsbooks: tuple[str, ...] = DEFAULT_RUNTIME_CONFIG.sportsbooks
    odds_regions: tuple[str, ...] = DEFAULT_RUNTIME_CONFIG.odds_regions
    markets: tuple[str, ...] = DEFAULT_RUNTIME_CONFIG.markets
    odds_format: str = "american"
    public_pick_inputs: tuple[Path, ...] = field(default_factory=tuple)
    public_pick_format: str | None = None
    public_pick_source: str = "manual_import"
    aggregate_public_picks: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class StepStatus:
    """Human-readable status for an operator summary."""

    label: str
    status: str
    detail: str = ""

    def line(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.label}: {self.status}{suffix}"


@dataclass(frozen=True)
class LiveWeekResult:
    """Workflow artifacts and summary text."""

    summary_text: str
    statuses: tuple[StepStatus, ...]
    rankings_report_path: Path | None
    simulation_report_path: Path | None
    portfolio_report_path: Path | None
    summary_report_path: Path | None
    rankings: pd.DataFrame
    portfolio_exposure: pd.DataFrame
    portfolio_metrics: dict[str, Any]


class LiveWeekWorkflowError(RuntimeError):
    """Raised when the live week workflow cannot continue safely."""


def run_live_week(
    options: LiveWeekOptions,
    *,
    odds_provider_factory: type[TheOddsAPIProvider] | None = None,
) -> LiveWeekResult:
    """Run the end-to-end live weekly workflow."""
    _validate_options(options)
    season_dir = _resolve_season_dir(options.season, options.data_dir)

    schedule_path = season_dir / "schedule.csv"
    schedule_df = _load_schedule(schedule_path, options)
    schedule_status = StepStatus(
        "Schedule",
        "OK",
        f"{_week_game_count(schedule_df, options.week)} games",
    )

    odds_df, odds_refresh_detail = _load_or_refresh_odds(
        options=options,
        schedule_df=schedule_df,
        season_dir=season_dir,
        provider_factory=odds_provider_factory,
    )
    model_odds_df = _model_odds_frame(odds_df)
    odds_status = _validate_week_odds(
        options=options,
        schedule_df=schedule_df,
        original_odds_df=odds_df,
        model_odds_df=model_odds_df,
        refresh_detail=odds_refresh_detail,
    )

    public_picks_df, public_summary_df, public_import_detail = _load_or_import_public_picks(
        options=options,
        schedule_df=schedule_df,
        season_dir=season_dir,
    )
    public_status = _validate_week_public_picks(
        options=options,
        schedule_df=schedule_df,
        public_picks_df=public_picks_df,
        public_summary_df=public_summary_df,
        import_detail=public_import_detail,
    )

    entries_path = season_dir / "entries.csv"
    entries_df = _load_entries(entries_path, options)

    rankings = rank_weekly_picks(
        week=options.week,
        schedule_df=schedule_df,
        odds_df=model_odds_df,
        public_picks_df=public_picks_df,
        entries_df=entries_df,
        pool_size=options.pool_size,
    )

    rankings_report_path: Path | None = None
    if not options.dry_run:
        rankings_report_path = write_weekly_report(
            rankings,
            options.week,
            output_dir=options.reports_dir,
        )
    rankings_status = StepStatus(
        "Rankings",
        "dry run" if options.dry_run else "generated",
    )

    simulation_schedule = _schedule_with_available_odds(
        schedule_df=schedule_df,
        model_odds_df=model_odds_df,
        start_week=options.week,
    )
    simulation_result = simulate_many_seasons(
        schedule_df=simulation_schedule,
        odds_df=model_odds_df,
        public_picks_df=public_picks_df,
        rankings_df=rankings,
        entries_df=entries_df,
        start_week=options.week,
        simulations=options.simulations,
        pool_size=options.pool_size,
        personal_entry_count=options.entries,
        seed=options.seed,
    )
    simulation_report_path: Path | None = None
    if not options.dry_run:
        simulation_report_path = write_simulation_report(
            simulation_result,
            output_dir=options.reports_dir,
        )
    simulation_status = StepStatus(
        "Simulations",
        "dry run" if options.dry_run else "generated",
        f"{options.simulations}",
    )

    allocation, exposure, metrics = optimize_portfolio_for_week(
        week=options.week,
        rankings_df=rankings,
        entries_df=entries_df,
        personal_entry_count=options.entries,
        aggression=options.aggression,
        random_seed=options.seed,
    )
    portfolio_report_path: Path | None = None
    if not options.dry_run:
        portfolio_report_path = write_portfolio_report(
            week=options.week,
            allocation_df=allocation,
            exposure_df=exposure,
            metrics=metrics,
            rankings_df=rankings,
            aggression=options.aggression,
            output_dir=options.reports_dir,
        )
    portfolio_status = StepStatus(
        "Portfolio",
        "dry run" if options.dry_run else "generated",
        _portfolio_status_detail(metrics, options),
    )

    statuses = (
        schedule_status,
        odds_status,
        public_status,
        rankings_status,
        simulation_status,
        portfolio_status,
    )
    report_paths = {
        "rankings": rankings_report_path,
        "simulation": simulation_report_path,
        "portfolio": portfolio_report_path,
    }
    summary_text = build_operator_summary(
        options=options,
        statuses=statuses,
        rankings=rankings,
        exposure=exposure,
        report_paths=report_paths,
    )

    summary_report_path: Path | None = None
    if not options.dry_run:
        summary_report_path = write_week_summary_report(
            options=options,
            statuses=statuses,
            rankings=rankings,
            exposure=exposure,
            report_paths=report_paths,
        )

    return LiveWeekResult(
        summary_text=summary_text,
        statuses=statuses,
        rankings_report_path=rankings_report_path,
        simulation_report_path=simulation_report_path,
        portfolio_report_path=portfolio_report_path,
        summary_report_path=summary_report_path,
        rankings=rankings,
        portfolio_exposure=exposure,
        portfolio_metrics=metrics,
    )


def build_operator_summary(
    *,
    options: LiveWeekOptions,
    statuses: tuple[StepStatus, ...],
    rankings: pd.DataFrame,
    exposure: pd.DataFrame,
    report_paths: dict[str, Path | None],
) -> str:
    """Build the concise terminal summary for operators."""
    top_pick = rankings.sort_values("rank").iloc[0]
    lines = [
        "========================================",
        f"NFL Survivor Week {options.week} Summary",
        "========================================",
    ]
    if options.dry_run:
        lines.append("Mode: DRY RUN (no reports or imported data written)")
    lines.extend(status.line() for status in statuses)
    lines.extend(
        [
            "",
            "Top Pick:",
            f"{top_pick['team']} over {top_pick['opponent']}",
            f"Win Prob: {float(top_pick['no_vig_win_probability']):.1%}",
            f"Consensus Ownership: {float(top_pick['public_pick_pct']):.1%}",
            f"Leverage Score: {float(top_pick['leverage_score']):.3f}",
            "",
            "Portfolio Exposure:",
        ],
    )

    if exposure.empty:
        lines.append("No portfolio exposure generated.")
    else:
        for row in exposure.sort_values(
            ["entries_allocated", "team"],
            ascending=[False, True],
        ).to_dict("records"):
            lines.append(f"{row['team']}: {int(row['entries_allocated'])}")

    lines.extend(["", "Reports:"])
    written_paths = [path for path in report_paths.values() if path is not None]
    if written_paths:
        lines.extend(_display_path(path) for path in written_paths)
    else:
        lines.append("No reports written.")
    lines.append("========================================")
    return "\n".join(lines)


def write_week_summary_report(
    *,
    options: LiveWeekOptions,
    statuses: tuple[StepStatus, ...],
    rankings: pd.DataFrame,
    exposure: pd.DataFrame,
    report_paths: dict[str, Path | None],
) -> Path:
    """Write the weekly report index that links generated reports."""
    report_dir = Path(options.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"week_{options.week}_summary.md"
    top_pick = rankings.sort_values("rank").iloc[0]

    lines = [
        f"# NFL Survivor Week {options.week} Summary",
        "",
        f"- Season: {options.season}",
        f"- Simulations: {options.simulations:,}",
        f"- Portfolio entries: {options.entries:,}",
        f"- Aggression: {options.aggression}",
        "",
        "## Data Checks",
        "",
        *[f"- {status.line()}" for status in statuses],
        "",
        "## Top Pick",
        "",
        (
            f"{top_pick['team']} over {top_pick['opponent']} with "
            f"{float(top_pick['no_vig_win_probability']):.1%} win probability, "
            f"{float(top_pick['public_pick_pct']):.1%} ownership, and "
            f"{float(top_pick['leverage_score']):.3f} leverage."
        ),
        "",
        "## Portfolio Exposure",
        "",
        _markdown_exposure_table(exposure),
        "",
        "## Reports",
        "",
    ]
    for label, path in report_paths.items():
        if path is None:
            lines.append(f"- {label.title()}: not written")
        else:
            lines.append(f"- [{label.title()}]({_relative_link(summary_path, path)})")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _load_schedule(path: Path, options: LiveWeekOptions) -> pd.DataFrame:
    if not path.exists():
        raise LiveWeekWorkflowError(
            f"Missing schedule file: {path}\n"
            "How to fix: create the season workspace with "
            f"`python scripts/create_real_data_workspace.py --season {options.season}` "
            "or pass --data-dir to a directory that contains schedule.csv.",
        )
    try:
        schedule = load_schedule_df(path)
    except Exception as exc:
        raise LiveWeekWorkflowError(
            f"Schedule validation failed for {path}:\n{exc}\n"
            "How to fix: repair schedule.csv so it contains canonical week, "
            "game_id, away_team, and home_team values.",
        ) from exc
    if _week_game_count(schedule, options.week) == 0:
        raise LiveWeekWorkflowError(
            f"Schedule file {path} has no games for week {options.week}.\n"
            "How to fix: verify --week and --season, or update schedule.csv with "
            "the requested NFL week.",
        )
    return schedule


def _load_or_refresh_odds(
    *,
    options: LiveWeekOptions,
    schedule_df: pd.DataFrame,
    season_dir: Path,
    provider_factory: type[TheOddsAPIProvider] | None,
) -> tuple[pd.DataFrame, str]:
    odds_path = season_dir / "odds.csv"
    if not options.refresh_odds:
        if not odds_path.exists():
            raise LiveWeekWorkflowError(
                f"Missing odds file: {odds_path}\n"
                "How to fix: run this workflow with --refresh-odds after setting "
                f"{API_KEY_ENV_VAR}, or import odds with "
                "`python scripts/import_odds.py --provider the-odds-api ...`.",
            )
        return _load_odds_file(odds_path, schedule_df), "local"

    factory = provider_factory or TheOddsAPIProvider
    provider = factory(
        season=options.season,
        week=options.week,
        regions=csv_text(options.odds_regions),
        markets=csv_text(options.markets),
        odds_format=options.odds_format,
        bookmakers=csv_text(options.sportsbooks),
    )
    try:
        refreshed = provider.normalize(schedule_df)
    except RuntimeError as exc:
        message = str(exc)
        if API_KEY_ENV_VAR in message:
            raise LiveWeekWorkflowError(
                f"Missing The Odds API key: {API_KEY_ENV_VAR} is not set.\n"
                "How to fix: set it for this shell, for example "
                f"`$env:{API_KEY_ENV_VAR} = \"...\"`, or rerun without "
                "--refresh-odds to use an existing local odds.csv.",
            ) from exc
        raise LiveWeekWorkflowError(
            f"The Odds API refresh failed:\n{exc}\n"
            "How to fix: verify network access, API account quota, requested "
            "sportsbooks, regions, and markets.",
        ) from exc
    except Exception as exc:
        raise LiveWeekWorkflowError(
            f"The Odds API refresh produced invalid odds:\n{exc}\n"
            "How to fix: confirm the API has NFL odds for this week and that "
            "schedule.csv uses canonical teams/game IDs.",
        ) from exc

    try:
        validate_odds_against_schedule(refreshed, schedule_df, raise_on_error=True)
    except ValueError as exc:
        raise LiveWeekWorkflowError(
            f"Refreshed odds do not join to the schedule:\n{exc}\n"
            "How to fix: check the requested week and the canonical schedule game IDs.",
        ) from exc

    if not options.dry_run:
        odds_path.parent.mkdir(parents=True, exist_ok=True)
        refreshed.to_csv(odds_path, index=False)
    return refreshed, "refreshed" if not options.dry_run else "refreshed dry run"


def _load_odds_file(path: Path, schedule_df: pd.DataFrame) -> pd.DataFrame:
    try:
        odds = load_odds_df(path)
    except Exception as exc:
        raise LiveWeekWorkflowError(
            f"Odds validation failed for {path}:\n{exc}\n"
            "How to fix: regenerate odds.csv with scripts/import_odds.py or repair "
            "the required odds columns.",
        ) from exc

    relationship_errors = validate_odds_schedule_relationship(odds, schedule_df)
    if relationship_errors:
        details = "\n".join(f"- {error}" for error in relationship_errors)
        raise LiveWeekWorkflowError(
            f"Odds game IDs do not join to schedule.csv:\n{details}\n"
            "How to fix: re-import odds against the same schedule.csv so every "
            "game_id, week, away_team, and home_team matches.",
        )
    return odds


def _model_odds_frame(odds_df: pd.DataFrame) -> pd.DataFrame:
    if set(NORMALIZED_ODDS_COLUMNS).issubset(odds_df.columns):
        aggregated = aggregate_book_odds(odds_df)
        return aggregated[
            aggregated["market_type"].astype(str).str.lower().isin(
                {"h2h", "moneyline"},
            )
        ].copy()
    return odds_df.copy()


def _validate_week_odds(
    *,
    options: LiveWeekOptions,
    schedule_df: pd.DataFrame,
    original_odds_df: pd.DataFrame,
    model_odds_df: pd.DataFrame,
    refresh_detail: str,
) -> StepStatus:
    try:
        team_odds = add_no_vig_probabilities(model_odds_df)
    except Exception as exc:
        raise LiveWeekWorkflowError(
            f"Odds probabilities are invalid for week {options.week}:\n{exc}\n"
            "How to fix: make sure odds.csv includes two moneyline sides per game "
            "or normalized no_vig_win_probability values between 0 and 1.",
        ) from exc

    week_odds = team_odds[team_odds["week"].astype(int) == int(options.week)].copy()
    if week_odds.empty:
        raise LiveWeekWorkflowError(
            f"Missing odds for current week {options.week}.\n"
            "How to fix: refresh odds with --refresh-odds or import a file that "
            f"contains week {options.week} rows.",
        )

    scheduled_game_ids = set(
        schedule_df.loc[
            schedule_df["week"].astype(int) == int(options.week),
            "game_id",
        ].astype(str),
    )
    odds_game_ids = set(week_odds["game_id"].astype(str))
    missing_game_ids = sorted(scheduled_game_ids - odds_game_ids)
    if missing_game_ids:
        examples = ", ".join(missing_game_ids[:5])
        suffix = f", and {len(missing_game_ids) - 5} more" if len(missing_game_ids) > 5 else ""
        raise LiveWeekWorkflowError(
            f"Missing odds for {len(missing_game_ids)} scheduled week "
            f"{options.week} games: {examples}{suffix}.\n"
            "How to fix: refresh or import odds after the schedule is finalized, "
            "then confirm every game_id in schedule.csv has two team rows.",
        )

    invalid_probabilities = week_odds[
        week_odds["no_vig_win_probability"].isna()
        | ~week_odds["no_vig_win_probability"].between(0, 1)
    ]
    if not invalid_probabilities.empty:
        raise LiveWeekWorkflowError(
            f"Odds probabilities are missing or outside 0..1 for week {options.week}.\n"
            "How to fix: regenerate odds.csv so every h2h team row has a sane "
            "no_vig_win_probability.",
        )

    probability_sums = week_odds.groupby("game_id")["no_vig_win_probability"].sum()
    bad_sums = probability_sums[~probability_sums.between(0.999, 1.001)]
    if not bad_sums.empty:
        examples = ", ".join(f"{game_id}={value:.3f}" for game_id, value in bad_sums.head(5).items())
        raise LiveWeekWorkflowError(
            f"Odds probabilities do not sum to 1.0 by game: {examples}.\n"
            "How to fix: verify each game has exactly two h2h sides from the same "
            "consensus/normalized odds set.",
        )

    books = _sportsbook_count(original_odds_df, options.week)
    detail = f"{len(scheduled_game_ids)} games, {books} sportsbook{'s' if books != 1 else ''}, {refresh_detail}"
    return StepStatus("Odds", "OK", detail)


def _load_or_import_public_picks(
    *,
    options: LiveWeekOptions,
    schedule_df: pd.DataFrame,
    season_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    public_picks_path = season_dir / "public_picks.csv"
    if options.public_pick_inputs:
        raw_records: list[dict[str, object]] = []
        for input_path in options.public_pick_inputs:
            provider = ManualPublicPickProvider(
                input_path,
                options.public_pick_format,
                source=options.public_pick_source,
                season=options.season,
                week=options.week,
            )
            try:
                raw_records.extend(provider.fetch_public_picks())
            except Exception as exc:
                raise LiveWeekWorkflowError(
                    f"Could not import public picks from {input_path}:\n{exc}\n"
                    "How to fix: provide a readable CSV/JSON file with team and "
                    "public_pick_pct columns, or omit --public-picks-input to use "
                    "the existing public_picks.csv.",
                ) from exc

        try:
            normalized = normalize_public_pick_records(raw_records, schedule_df)
            summary_df = (
                aggregate_public_pick_sources(normalized)
                if options.aggregate_public_picks
                else normalized
            )
            validate_public_picks_against_schedule(
                summary_df,
                schedule_df,
                raise_on_error=True,
            )
        except Exception as exc:
            raise LiveWeekWorkflowError(
                f"Imported public picks are invalid:\n{exc}\n"
                "How to fix: ensure every public pick team plays in schedule.csv "
                "for the requested week and every percentage is between 0 and 1.",
            ) from exc

        if not options.dry_run:
            public_picks_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(public_picks_path, index=False)
        compact = _compact_public_picks(summary_df)
        detail = "imported" if not options.dry_run else "imported dry run"
        return compact, summary_df, detail

    if not public_picks_path.exists():
        raise LiveWeekWorkflowError(
            f"Missing public picks file: {public_picks_path}\n"
            "How to fix: import ownership with --public-picks-input, or run "
            "`python scripts/import_public_picks.py ... --aggregate` to create "
            "public_picks.csv before the weekly workflow.",
        )

    try:
        public_picks = load_public_picks_df(public_picks_path)
        raw_summary = pd.read_csv(public_picks_path)
    except Exception as exc:
        raise LiveWeekWorkflowError(
            f"Public picks validation failed for {public_picks_path}:\n{exc}\n"
            "How to fix: repair public_picks.csv so percentages are decimals "
            "between 0 and 1 and teams join to the schedule.",
        ) from exc
    return public_picks, raw_summary, "local"


def _compact_public_picks(public_picks_df: pd.DataFrame) -> pd.DataFrame:
    pick_column = (
        "consensus_public_pick_pct"
        if "consensus_public_pick_pct" in public_picks_df.columns
        else "public_pick_pct"
    )
    return (
        public_picks_df[["week", "team", pick_column]]
        .rename(columns={pick_column: "public_pick_pct"})
        .copy()
    )


def _validate_week_public_picks(
    *,
    options: LiveWeekOptions,
    schedule_df: pd.DataFrame,
    public_picks_df: pd.DataFrame,
    public_summary_df: pd.DataFrame,
    import_detail: str,
) -> StepStatus:
    relationship_errors = validate_public_picks_schedule_relationship(
        public_picks_df,
        schedule_df,
    )
    if relationship_errors:
        details = "\n".join(f"- {error}" for error in relationship_errors)
        raise LiveWeekWorkflowError(
            f"Public picks do not join to schedule.csv:\n{details}\n"
            "How to fix: normalize team aliases and verify every public-pick team "
            "plays in the requested week.",
        )

    week_picks = public_picks_df[
        public_picks_df["week"].astype(int) == int(options.week)
    ].copy()
    if week_picks.empty:
        raise LiveWeekWorkflowError(
            f"Missing public picks for current week {options.week}.\n"
            "How to fix: add public_picks.csv rows for this week or import them "
            "with --public-picks-input.",
        )
    if not week_picks["public_pick_pct"].between(0, 1).all():
        raise LiveWeekWorkflowError(
            f"Public pick percentages are invalid for week {options.week}.\n"
            "How to fix: store percentages as decimals between 0 and 1, for "
            "example 0.31 for 31%.",
        )

    source_count = _public_source_count(public_summary_df, options.week)
    detail = f"{source_count} source{'s' if source_count != 1 else ''}, {import_detail}"
    return StepStatus("Public Picks", "OK", detail)


def _load_entries(path: Path, options: LiveWeekOptions) -> pd.DataFrame:
    if not path.exists():
        raise LiveWeekWorkflowError(
            f"Missing entries file: {path}\n"
            "How to fix: create entries.csv from data/raw/templates/entries_template.csv "
            "and list active entries plus used teams.",
        )
    try:
        return load_entries_df(path)
    except Exception as exc:
        raise LiveWeekWorkflowError(
            f"Entries validation failed for {path}:\n{exc}\n"
            "How to fix: repair entries.csv so it uses entry_id, active, and "
            "used_teams columns or the supported historical pick schema.",
        ) from exc


def _schedule_with_available_odds(
    *,
    schedule_df: pd.DataFrame,
    model_odds_df: pd.DataFrame,
    start_week: int,
) -> pd.DataFrame:
    team_odds = add_no_vig_probabilities(model_odds_df)
    available_weeks = sorted(
        week
        for week in team_odds["week"].astype(int).unique().tolist()
        if week >= int(start_week)
    )
    return schedule_df[schedule_df["week"].astype(int).isin(available_weeks)].copy()


def _validate_options(options: LiveWeekOptions) -> None:
    if options.week <= 0:
        raise LiveWeekWorkflowError("--week must be positive.")
    if options.simulations <= 0:
        raise LiveWeekWorkflowError("--simulations must be positive.")
    if options.entries <= 0:
        raise LiveWeekWorkflowError("--entries must be positive.")
    if options.pool_size < options.entries:
        raise LiveWeekWorkflowError("--pool-size must be at least --entries.")
    if options.refresh_odds and not options.markets:
        raise LiveWeekWorkflowError("At least one odds market is required.")


def _resolve_season_dir(season: int, data_dir: Path) -> Path:
    base = Path(data_dir)
    if (base / "schedule.csv").exists():
        return base
    return base / str(season)


def _week_game_count(schedule_df: pd.DataFrame, week: int) -> int:
    return int((schedule_df["week"].astype(int) == int(week)).sum())


def _sportsbook_count(odds_df: pd.DataFrame, week: int) -> int:
    if "sportsbook" not in odds_df.columns:
        return 1
    current = odds_df[odds_df["week"].astype(int) == int(week)].copy()
    if "market_type" in current.columns:
        current = current[current["market_type"].astype(str).str.lower() == "h2h"]
    count = current["sportsbook"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    return int(count) if count else 1


def _public_source_count(public_summary_df: pd.DataFrame, week: int) -> int:
    current = public_summary_df[
        pd.to_numeric(public_summary_df["week"], errors="coerce") == int(week)
    ].copy()
    if current.empty:
        return 0
    if "source_count" in current.columns:
        values = pd.to_numeric(current["source_count"], errors="coerce").dropna()
        if not values.empty:
            return int(values.max())
    if "sources" in current.columns:
        sources: set[str] = set()
        for value in current["sources"].dropna():
            sources.update(part.strip() for part in str(value).split(";") if part.strip())
        if sources:
            return len(sources)
    if "source" in current.columns:
        sources = {
            str(value).strip()
            for value in current["source"].dropna()
            if str(value).strip()
        }
        return len(sources) if sources else 1
    return 1


def _portfolio_status_detail(
    metrics: dict[str, Any],
    options: LiveWeekOptions,
) -> str:
    active_entries = int(metrics.get("active_entries", options.entries))
    allocated_entries = int(metrics.get("allocated_entries", active_entries))
    if active_entries == allocated_entries == options.entries:
        return f"{active_entries} entries"
    return f"{allocated_entries} allocated / {active_entries} active entries"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _relative_link(from_path: Path, to_path: Path) -> str:
    try:
        relative = to_path.resolve().relative_to(from_path.resolve().parent)
        return str(relative).replace("\\", "/")
    except ValueError:
        return _display_path(to_path).replace(" ", "%20")


def _markdown_exposure_table(exposure: pd.DataFrame) -> str:
    if exposure.empty:
        return "_No portfolio exposure generated._"

    rows = ["| Team | Entries | Exposure |", "| --- | --- | --- |"]
    for row in exposure.sort_values(
        ["entries_allocated", "team"],
        ascending=[False, True],
    ).to_dict("records"):
        rows.append(
            f"| {row['team']} | {int(row['entries_allocated'])} | "
            f"{float(row['exposure_pct']):.1%} |",
        )
    return "\n".join(rows)
