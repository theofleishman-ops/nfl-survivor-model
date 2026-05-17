# nfl-survivor-model

CSV-driven NFL survivor ranking and simulation prototype.

Phase 1 ranks weekly survivor picks from local CSV inputs using no-vig win
probability, public ownership, leverage, week scarcity, and full-season future
opportunity cost. Phase 2 adds a Monte Carlo engine that simulates game
outcomes, public-field eliminations, personal-entry survival, contest equity,
and upset leverage. Phase 3 adds a multi-entry portfolio optimizer for roughly
40 personal entries. Phase 4 adds real-data CSV templates, validation, import
helpers, and season-aware loaders. Phase 5 adds canonical NFL team and game
identity helpers so future schedule, odds, and public-pick files can join on
stable IDs. Phase 6 imports the official 2026 NFL regular-season schedule into
the real-data workspace using those canonical IDs. Phase 7 adds a sanitized
real Week 1 fixture that exercises the real-data pipeline without private pool
data. Phase 8 adds a provider-agnostic odds ingestion foundation for manual
CSV/JSON imports and future API or plugin adapters. Phase 9 adds automated NFL
odds import from The Odds API. Phase 10 adds provider-agnostic public pick
ingestion for manual CSV/JSON ownership exports and consensus aggregation.
Phase 11 adds a single-command live weekly workflow for validation, rankings,
simulations, portfolio allocation, and markdown report indexing. Phase 12 adds
a true single-entry path EV optimizer that maximizes simulated contest equity
while keeping the heuristic weekly ranking as a diagnostic comparison. It does
not scrape websites or build a dashboard. Phase 13 integrates that path EV
optimizer into the live weekly workflow so the operator summary shows the
heuristic top pick, path EV top pick, portfolio allocation, and dollar EV
metrics when contest economics are provided.

The real 2026 schedule is included. No real odds, real pool data, secrets, API
keys, raw API responses, or scraping code are included.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Tests

```powershell
pytest
```

## Real Data Workflow

This project still has no scraping, committed secrets, or committed live-feed
data. Paste or export CSVs from trusted sources, or import odds from The Odds
API at runtime, import public pick ownership from manual CSV/JSON exports, then
validate the local files before running the model. Public pick percentages
should be decimals, e.g. `0.38` for 38%. Moneylines should be American odds,
e.g. `-350` or `+220`.

Create a season workspace from the small templates:

```powershell
python scripts/create_real_data_workspace.py --season 2026
```

The official 2026 regular-season schedule has already been imported into:

```text
data/raw/2026/schedule.csv
```

The schedule uses canonical team abbreviations, canonical game IDs, and
explicit `TBD` values for late-season flex games whose kickoff dates/times have
not yet been assigned by the NFL.

Fill or replace these CSVs before running real rankings:

```text
data/raw/2026/odds.csv
data/raw/2026/public_picks.csv
data/raw/2026/entries.csv
```

Optional files are also created for later use:

```text
data/raw/2026/pool_history.csv
data/raw/2026/double_pick_weeks.csv
```

Only `data/raw/2026/schedule.csv` is meant to be versioned for the real 2026
workspace. Keep odds, public picks, entries, pool history, and double-pick-week
files local unless you intentionally create sanitized fixtures under
`data/raw/templates/`.

Validate the season files:

```powershell
python scripts/validate_data_files.py --data-dir data/raw/2026
```

Validation checks each file's schema and also verifies that odds `game_id`
values join to the schedule and public-pick teams are actually scheduled in the
same week. Template odds copied into the real 2026 folder should fail until they
are replaced with odds for the imported 2026 schedule.

## Real Week 1 Fixture

A small sanitized fixture lives in:

```text
data/sample/real_week_1/
```

It contains real 2026 Week 1 schedule rows and canonical game IDs copied from
`data/raw/2026/schedule.csv`. The odds and public-pick values are synthetic
placeholders, clearly marked with `SYNTHETIC_FIXTURE`, and exist only to prove
that the real-data join path, rankings, simulations, and portfolio optimizer
can run end to end. Do not use those synthetic values for real contest
decisions.

Run the fixture:

```powershell
python scripts/validate_data_files.py --data-dir data/sample/real_week_1
python scripts/run_weekly_rankings.py --week 1 --data-dir data/sample/real_week_1
python scripts/run_simulations.py --week 1 --data-dir data/sample/real_week_1 --simulations 1000
python scripts/run_single_entry_optimizer.py --season 2026 --week 1 --data-dir data/sample/real_week_1 --simulations 5000 --beam-width 100 --top-k 5
python scripts/run_portfolio_optimizer.py --week 1 --data-dir data/sample/real_week_1 --entries 40 --aggression balanced
```

To replace the synthetic values, edit `data/sample/real_week_1/odds.csv` or
copy it to a private season workspace and replace `home_moneyline` and
`away_moneyline` with real American odds for the same `game_id`, `week`,
`home_team`, and `away_team`. Edit `public_picks.csv` with real decimal public
pick shares for teams scheduled in Week 1, keeping each value between `0` and
`1` and the Week 1 total at or below `1.0`. Then run:

```powershell
python scripts/validate_data_files.py --data-dir data/sample/real_week_1
```

Normalize an exported schedule before using it:

```powershell
python scripts/normalize_schedule_file.py --input raw_schedule.csv --output normalized_schedule.csv --season 2026
```

For the checked-in 2026 schedule, validate the schedule file through tests or
validate the full workspace after the private odds and pool files have been
filled with rows matching that schedule:

```powershell
python scripts/validate_data_files.py --data-dir data/raw/2026
```

Run the live weekly workflow against the real-data workspace once `odds.csv`,
`public_picks.csv`, and `entries.csv` have been filled with real contest data
that matches the schedule game IDs:

```powershell
python scripts/run_live_week.py --season 2026 --week 1
```

The template files live in:

```text
data/raw/templates/
```

## Live Weekly Workflow

`scripts/run_live_week.py` is the operator-facing weekly command. It validates
the schedule and current-week joins, optionally refreshes odds, optionally
imports manual public-pick files, runs rankings, runs simulations, optimizes the
entry portfolio, optionally runs the single-entry path EV optimizer, writes
markdown reports, and prints a concise summary.

Common runs:

```powershell
python scripts/run_live_week.py --season 2026 --week 1
python scripts/run_live_week.py --season 2026 --week 1 --refresh-odds
python scripts/run_live_week.py --season 2026 --week 1 --refresh-odds --simulations 10000 --entries 40
python scripts/run_live_week.py --season 2026 --week 1 --run-path-ev --path-ev-simulations 5000 --beam-width 100 --top-k 5 --entry-fee 10 --prize-pool 50000
```

Dry-run validation and calculation without writing imported data or reports:

```powershell
python scripts/run_live_week.py --season 2026 --week 1 --dry-run --simulations 1000
```

The expected weekly process is:

1. Update `data/raw/2026/entries.csv` with active entries and used teams.
2. Refresh odds with `--refresh-odds`, or import odds separately with
   `scripts/import_odds.py`.
3. Import public picks with `--public-picks-input`, or import them separately
   with `scripts/import_public_picks.py --aggregate`.
4. Run `scripts/run_live_week.py` with the final simulation count, entry count,
   aggression mode, and `--run-path-ev` if you want the true single-entry
   path EV pick in the weekly command center.
5. Review `outputs/reports/week_1_summary.md`, then the linked rankings,
   simulation, portfolio, and optional path EV reports.

Refresh odds through The Odds API:

```powershell
$env:THE_ODDS_API_KEY = "paste-your-api-key-here"
python scripts/run_live_week.py --season 2026 --week 1 --refresh-odds --sportsbooks fanduel,draftkings,betmgm --markets h2h,spreads,totals
```

Import public picks as part of the live workflow:

```powershell
python scripts/run_live_week.py --season 2026 --week 1 --public-picks-input data/sample/public_pick_sources/yahoo_week1_public_picks.csv data/sample/public_pick_sources/espn_week1_public_picks.csv data/sample/public_pick_sources/survivorgrid_week1_public_picks.csv --public-picks-format csv
```

Generated reports:

```text
outputs/reports/week_1_summary.md
outputs/reports/week_1_report.md
outputs/reports/week_1_simulation_report.md
outputs/reports/week_1_portfolio_report.md
outputs/reports/week_1_single_entry_path_ev.md  # when --run-path-ev is used
```

The summary report is an index that links the rankings, simulation, portfolio,
and optional path EV reports. Path EV is skipped unless `--run-path-ev` is set;
the summary says skipped instead of failing. When enabled, the summary includes
the path EV best pick, path EV estimate, EV multiple, expected edge, and a
comparison to the heuristic top pick. Add `--entry-fee` and `--prize-pool` to
also show EV dollars. The terminal summary calls out missing files, malformed
game IDs, missing current-week odds, invalid probabilities, missing public
picks, and missing API keys with a concrete fix.

## Odds Ingestion

Odds ingestion is provider-agnostic by design. The model should not depend on
one sportsbook's page structure or HTML, because those pages change often,
carry legal/terms-of-use concerns, and are hard to test without brittle live
network behavior. Instead, every provider adapts into one normalized schema:

```text
season, week, game_id, sportsbook, market_type, team, opponent,
home_team, away_team, moneyline, spread, spread_price, total, total_price,
implied_probability, no_vig_win_probability, pulled_at, source
```

Supported market types include `h2h`, `spreads`, `totals`, and futures-style
rows for `super_bowl`, `conference`, `division`, `playoff`, and `win_total`.

Manual imports work today:

```powershell
python scripts/import_odds.py --input odds_raw.csv --format csv --season 2026 --schedule data/raw/2026/schedule.csv --output data/raw/2026/odds.csv
python scripts/import_odds.py --input odds_raw.json --format json --season 2026 --schedule data/raw/2026/schedule.csv --output data/raw/2026/odds.csv
```

Automated NFL odds import from The Odds API also works. Create an account and
API key at:

```text
https://the-odds-api.com/
```

Set the key for the current PowerShell session only:

```powershell
$env:THE_ODDS_API_KEY = "paste-your-api-key-here"
```

Then import a week:

```powershell
python scripts/import_odds.py --provider the-odds-api --season 2026 --week 1 --schedule data/raw/2026/schedule.csv --output data/raw/2026/odds.csv --regions us --markets h2h,spreads,totals --odds-format american --sportsbook fanduel,draftkings,betmgm
```

Use `--dry-run` to fetch, normalize, validate, and preview rows without writing
`odds.csv`. Use `--no-write` to validate silently without writing. Raw API
responses are not saved by default. If you need debug output, pass an explicit
ignored path such as:

```powershell
python scripts/import_odds.py --provider the-odds-api --season 2026 --week 1 --schedule data/raw/2026/schedule.csv --output data/raw/2026/odds.csv --save-raw .odds-api-raw/week1.json --dry-run
```

Never commit API keys, `.env` files, shell history containing keys, or raw API
responses. Sanitized mock responses for tests may live under `tests/fixtures/`;
private live responses should stay in ignored local paths.

The importer normalizes team aliases, joins to canonical `game_id` values,
calculates implied probabilities, creates no-vig probabilities when both
moneyline sides are present, writes normalized CSV, and validates the output
against the schedule.

The Odds API provider lives in `src/survivor/odds_providers/the_odds_api.py`.
It reads `THE_ODDS_API_KEY` only from the runtime environment or an explicit
constructor argument used in tests; no keys are stored in the repo.

Current limitations:

- The provider uses The Odds API's current `/v4/sports/americanfootball_nfl/odds`
  endpoint, so it can only import markets that the API is currently returning.
- `--week` is required so current API events can be matched to the local 2026
  schedule by canonical home and away teams.
- Supported game markets are `h2h`, `spreads`, and `totals`; unsupported market
  names are ignored when at least one supported market remains.
- The normalized schema is team-oriented. Totals are retained as game-total rows
  using the home-team row for the over price and the away-team row for the under
  price.

FanDuel or another sportsbook can be added later as a plugin that outputs raw
provider records for this normalizer. Keep that plugin separate from the core
model so scraping-specific risk does not become the main ingestion system.

Recommended odds workflow:

1. Get raw odds from a trusted manual export, JSON file, API, or future plugin.
2. Import and normalize with `scripts/import_odds.py`.
3. Validate with `scripts/validate_data_files.py --data-dir data/raw/2026`.
4. Run rankings, simulations, and portfolio optimization.

## Public Pick Ingestion

Public ownership is the model's field estimate: rankings use it for leverage,
simulations use it to estimate public-field eliminations, and the portfolio
optimizer uses it to avoid being overexposed to fragile chalk.

The normalized source schema is:

```text
season, week, team, opponent, game_id, source, public_pick_pct,
sample_size, pulled_at, notes
```

`sample_size`, `pulled_at`, and `notes` are optional. Team aliases such as
`Chiefs`, `Kansas City Chiefs`, and `K.C.` normalize to canonical teams, and
each row is joined to `schedule.csv` by team/week or `game_id`. Teams not
playing that week fail validation. `public_pick_pct` must be between `0` and
`1`; strings like `18%` are accepted, but bare numeric values should be
decimals such as `0.18`.

Manual CSV and JSON imports work today:

```powershell
python scripts/import_public_picks.py --input raw_public_picks.csv --format csv --season 2026 --week 1 --source yahoo --schedule data/raw/2026/schedule.csv --output data/raw/2026/public_picks.csv --aggregate
python scripts/import_public_picks.py --input raw_public_picks.json --format json --season 2026 --week 1 --source espn --schedule data/raw/2026/schedule.csv --output data/raw/2026/public_picks.csv --aggregate
```

You can also pass multiple same-format source files and aggregate them in one
run:

```powershell
python scripts/import_public_picks.py --input data/sample/public_pick_sources/yahoo_week1_public_picks.csv data/sample/public_pick_sources/espn_week1_public_picks.csv data/sample/public_pick_sources/survivorgrid_week1_public_picks.csv --format csv --season 2026 --week 1 --schedule data/raw/2026/schedule.csv --output data/raw/2026/public_picks.csv --aggregate --dry-run
```

Aggregation writes `consensus_public_pick_pct` and also mirrors that value into
`public_pick_pct` so the existing ranking, simulation, and portfolio workflows
can consume the file directly. The initial method is a simple average across
sources. If every source row for a team has a positive `sample_size`, the
consensus is sample-size weighted. A source's weekly total may be below `1.0`
because some sites show only top picks; a source total materially above `1.0`
fails validation.

The provider interface lives under `src/survivor/public_pick_providers/` so
future Yahoo, ESPN, OfficeFootballPool, SurvivorGrid, RunYourPool, or custom
pool-history adapters can be added without changing the normalizer. No
scraping, API keys, private pool data, or website-specific logic is included.

Synthetic examples live in:

```text
data/sample/public_pick_sources/
data/sample/real_week_1/public_picks_consensus.csv
```

Those values are fake fixtures for deterministic tests and demos, not real
public ownership.

## Canonical Team And Game IDs

Canonical team abbreviations use this stable 32-team set:

```text
ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LAR MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS
```

Aliases normalize through `survivor.teams`:

```python
from survivor.teams import normalize_team_name, is_valid_team

normalize_team_name("Kansas City Chiefs")  # "KC"
normalize_team_name("K.C.")                # "KC"
is_valid_team("Chiefs")                    # True
```

Game IDs use:

```text
{season}_W{week}_{away_team}_AT_{home_team}
```

Example:

```python
from survivor.game_ids import build_game_id, parse_game_id

game_id = build_game_id(2026, 1, "Baltimore Ravens", "Kansas City")
# "2026_W01_BAL_AT_KC"
parse_game_id(game_id)
```

Schedule normalization accepts common column names such as `home`, `homeTeam`,
`awayTeam`, `kickoffTime`, and the fallback pair `favorite` / `underdog`.
It writes canonical `week`, `game_id`, `home_team`, and `away_team` columns:

```python
import pandas as pd
from survivor.loaders import normalize_schedule_df

raw = pd.read_csv("raw_schedule.csv")
schedule = normalize_schedule_df(raw, season=2026)
```

Canonical IDs matter because every downstream table eventually needs to join
schedule rows, odds, public picks, simulation outcomes, reports, and portfolio
allocations without depending on provider-specific spellings like `K.C.`,
`Kansas City`, or `Chiefs`.

## Run Weekly Rankings

The CLI uses `data/sample/` by default:

```powershell
python scripts/run_weekly_rankings.py --week 1
python scripts/run_weekly_rankings.py --week 1 --use-sample
```

It prints the top recommendations and writes:

```text
outputs/reports/week_1_report.md
```

## Run Monte Carlo Simulations

The simulation CLI also uses `data/sample/` by default:

```powershell
python scripts/run_simulations.py --week 1 --simulations 10000
```

Useful options:

```powershell
python scripts/run_simulations.py --week 1 --simulations 2500 --pool-size 5000 --personal-entries 4 --seed 2026
```

It prints headline survival and equity metrics and writes compact outputs to:

```text
outputs/simulations/week_1_simulation_report.md
outputs/simulations/week_1_season_summary.csv
outputs/simulations/week_1_week_summary.csv
outputs/simulations/week_1_leverage_summary.csv
outputs/simulations/week_1_path_summary.csv
```

## Phase 1 Model

The current model is intentionally explicit and decomposable:

```text
final_score =
  win_probability_component
  + leverage_component
  - future_cost_component
```

Inputs:

```text
data/sample/schedule_sample.csv
data/sample/odds_sample.csv
data/sample/public_picks_sample.csv
data/sample/entries_sample.csv
```

Core pieces:

- Odds conversion turns American moneylines into implied probabilities and then
  normalizes two-sided markets into no-vig win probabilities.
- Candidate generation creates one row per team for a requested week, including
  opponent, moneyline, implied probability, no-vig probability, public pick
  percentage, and entry-specific used-team flags when applicable.
- Week scarcity counts teams at `>= 0.60`, `>= 0.65`, `>= 0.70`, and `>= 0.75`
  win probability. Fewer good options means a higher numeric scarcity score.
- Future opportunity cost looks across all remaining weeks. If the current
  candidate has strong future spots, the model compares that team to the next
  best replacement option and weights replacement cost more heavily in scarce
  weeks.
- Leverage rewards lower-owned teams with comparable win probability and
  calculates expected field impact using a configurable pool size, defaulting
  to `5000`.

Default weights live in `src/survivor/optimizer.py` and can be overridden when
calling `rank_weekly_picks`.

## Phase 2 Simulation Model

The Monte Carlo engine lives in `src/survivor/simulator.py`.

Core assumptions:

- Each NFL game winner is sampled from the no-vig win probability.
- The public field picks proportionally to `public_pick_pct` for each week.
- Public entries are tracked as an aggregate field size, while personal entries
  are tracked separately with entry-level alive status, used teams, and paths.
- Personal entries use the existing weekly ranking output as a simple
  best-available strategy inside simulations. The Phase 3 optimizer separately
  creates diversified one-week allocations before future simulation integration.
- Contest equity is approximated as:

```text
personal surviving entries / total surviving entries
```

Key outputs:

- Probability at least one personal entry survives through each week and the
  full simulated season.
- Expected public, personal, and total remaining entries by week.
- Expected contest equity.
- Conditional upset leverage, including expected field shrinkage when owned
  teams lose and the contest-equity lift after those outcomes.
- Path summaries for the personal entries, when enough simulations produce
  surviving paths.

Interpretation guidance:

- A higher survival probability means the personal entries are more likely to
  remain alive, but not necessarily to have high contest equity.
- A high conditional field-shrink value identifies fragile chalk: if that team
  loses, many public entries are removed at once.
- A positive contest-equity lift after a team loses suggests that fading that
  team can create leverage when your own entry survives.
- `uniqueness_value` is a simple first-pass proxy for strong picks that are not
  crowded by public ownership.

## Run Single-Entry Path EV

The weekly ranking model is still useful, but it is a heuristic:

```text
win probability + leverage - future cost
```

For one survivor entry, the direct objective is expected contest equity. A path
has value when it survives through the simulated season and owns a larger share
of the remaining contest. In the current model, if the entry survives and
`N` total entries remain, equity is approximated as `1 / N`; if the entry is
eliminated, equity is `0`.

Run the path optimizer:

```powershell
python scripts/run_single_entry_optimizer.py --season 2026 --week 1 --data-dir data/sample/real_week_1 --simulations 5000 --beam-width 100 --top-k 5
```

It prints the best current-week pick, best path, path EV, and survival
probability, then writes:

```text
outputs/reports/week_1_single_entry_path_ev.md
```

Add dollar reporting when you know the entry fee and prize pool:

```powershell
python scripts/run_single_entry_optimizer.py --season 2026 --week 1 --data-dir data/sample/real_week_1 --simulations 5000 --beam-width 100 --top-k 5 --entry-fee 10 --prize-pool 50000
```

With a prize pool, `EV dollars = expected contest equity * prize_pool`. With
an entry fee or inferred baseline value, the report also shows EV multiple and
expected edge, such as `$11.67`, `1.167x`, and `+16.7%`.

The optimizer first uses a controlled beam search to avoid brute-forcing every
possible team sequence. The weekly heuristic ranking and win probability order
candidate expansion, retaining the top `--top-k` candidates per week and the
top `--beam-width` paths after each expansion. The final decision is then made
by Monte Carlo path EV, not by the heuristic score.

Assumptions and limitations:

- Current public pick percentages are used when available.
- Future ownership uses public-pick rows when present; otherwise it is
  projected from win probability, placeholder team popularity, and the scarcity
  of good alternatives in that week.
- Win probabilities are tagged as `real_moneyline`, `real_spread`,
  `projected_team_strength`, or `fallback_default`. The model uses no-vig
  moneyline first, then spread, then futures/win-total or explicit team-strength
  priors with opponent strength and home field, then a conservative default.
- `--path-ev-horizon available` still evaluates only consecutive weeks with
  real odds. Use `--path-ev-horizon full-season` to optimize against projected
  probabilities for every remaining scheduled week.
- Final equity assumes winner-take-all or equal split among survivors.
- The public field is modeled in aggregate by week and does not track every
  public entry's used-team history.

## Run Portfolio Optimization

The portfolio CLI uses the weekly ranking engine, expands the sample entries to
the requested active portfolio size, and allocates picks across entries:

```powershell
python scripts/run_portfolio_optimizer.py --week 1 --entries 40 --aggression balanced
```

It prints an exposure summary and writes:

```text
outputs/reports/week_1_portfolio_report.md
```

Aggression modes:

- `conservative`: emphasizes no-vig win probability and keeps contrarian
  exposure smaller.
- `balanced`: blends final score, survival probability, leverage, ownership,
  and future flexibility.
- `aggressive`: gives more weight to leverage and uniqueness and can allocate
  small exposure to teams below 50% win probability.

Exposure summary interpretation:

- `entries_allocated` is the number of personal entries assigned to a team.
- `exposure_pct` is that team's share of active personal entries. The default
  cap is 40%.
- `public_pick_pct` shows how crowded the team is in the public field.
- `portfolio_score` is the optimizer's equity-oriented blend of weekly ranking
  score, win probability, leverage, ownership uniqueness, and future
  flexibility.
- The report includes both independent and correlated estimates for the chance
  at least one personal entry survives the week. The correlated estimate groups
  duplicate team picks into one shared game outcome and is the better lens for
  same-team risk.

## Project Layout

```text
data/
  raw/          Local raw inputs, ignored by git
  processed/    Local processed inputs, ignored by git
  sample/       Small fake CSVs for tests and examples
outputs/
  reports/      Generated reports, ignored by git
  simulations/  Generated simulation outputs, ignored by git
  charts/       Generated charts, ignored by git
scripts/        Command-line helpers
src/survivor/   Python package
tests/          Pytest suite
```

## Roadmap

Phase 2 Monte Carlo simulation: simulate pool outcomes across full seasons and
estimate contest equity distributions. Initial engine is implemented.

Phase 3 portfolio optimizer for 40 entries: implemented as a deterministic,
heuristic allocation layer with diversification and correlated-risk metrics.

Phase 4 real-data templates and validation: implemented for manual schedule,
odds, public-pick, entry, pool-history, and double-pick-week CSVs.

Phase 5 canonical team and game identity layer: implemented for NFL team
aliases, deterministic game IDs, schedule normalization, schedule integrity
validation, game-window classification, and bye-week helpers.

Phase 6 real 2026 schedule import: implemented with canonical game IDs,
schedule validation, game-window classification, and bye-week helpers.

Phase 7 sanitized real-week fixture workflow: implemented for real Week 1
schedule joins with synthetic odds and public-pick values.

Phase 8 odds ingestion foundation: implemented for provider-agnostic normalized
odds, manual CSV/JSON imports, provider interfaces, API stubs, and simple
futures-derived team-strength priors.

Phase 9 The Odds API provider: implemented for current NFL `h2h`, `spreads`,
and `totals` odds with environment-based secrets and mocked HTTP tests.

Phase 10 public pick ingestion foundation: implemented for manual CSV/JSON
ownership imports, source-aware schedule validation, provider interfaces, and
simple or sample-size weighted consensus aggregation.

Phase 11 live weekly workflow: implemented for one-command validation, optional
odds refresh, optional public-pick import, rankings, simulations, portfolio
optimization, operator summaries, and markdown report indexing.

Phase 12 single-entry path EV optimizer: implemented with beam-search path
generation, fixed-path Monte Carlo contest-equity evaluation, heuristic
ranking comparison, and markdown reports.

Future work: add ownership forecasting, more API adapters, optional scraper
plugins, and an interactive dashboard without secrets, committed private data,
or scraping in the core model.
