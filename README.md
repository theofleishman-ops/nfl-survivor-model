# nfl-survivor-model

CSV-driven NFL survivor ranking and simulation prototype.

Phase 1 ranks weekly survivor picks from local CSV inputs using no-vig win
probability, public ownership, leverage, week scarcity, and full-season future
opportunity cost. Phase 2 adds a Monte Carlo engine that simulates game
outcomes, public-field eliminations, personal-entry survival, contest equity,
and upset leverage. Phase 3 adds a multi-entry portfolio optimizer for roughly
40 personal entries. Phase 4 adds real-data CSV templates, validation, import
helpers, and season-aware loaders. It does not scrape websites, ingest live
feeds, or build a dashboard.

No real odds, real pool data, secrets, API keys, or scraping code are included.

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

This project still has no scraping, APIs, secrets, or live feeds. Paste or
export CSVs from trusted sources into the local files below, then validate them
before running the model. Public pick percentages should be decimals, e.g.
`0.38` for 38%. Moneylines should be American odds, e.g. `-350` or `+220`.

Create a season workspace from the small templates:

```powershell
python scripts/create_real_data_workspace.py --season 2026
```

Fill these CSVs:

```text
data/raw/2026/schedule.csv
data/raw/2026/odds.csv
data/raw/2026/public_picks.csv
data/raw/2026/entries.csv
```

Optional files are also created for later use:

```text
data/raw/2026/pool_history.csv
data/raw/2026/double_pick_weeks.csv
```

Validate the season files:

```powershell
python scripts/validate_data_files.py --data-dir data/raw/2026
```

Run the model against the real-data workspace:

```powershell
python scripts/run_weekly_rankings.py --week 1 --season 2026 --data-dir data/raw
python scripts/run_simulations.py --week 1 --season 2026 --data-dir data/raw --simulations 10000
python scripts/run_portfolio_optimizer.py --week 1 --season 2026 --data-dir data/raw --entries 40
```

The template files live in:

```text
data/raw/templates/
```

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

Phase 5 ownership forecasting and ingestion adapters: project public pick
percentages before they are known and add validated ingestion adapters for real
odds and public-pick exports, without secrets in the repo.

Phase 6 dashboard: build an interactive UI for rankings, reports, scenarios,
and portfolio recommendations.
