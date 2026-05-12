# nfl-survivor-model

CSV-driven NFL survivor ranking and simulation prototype.

Phase 1 ranks weekly survivor picks from local CSV inputs using no-vig win
probability, public ownership, leverage, week scarcity, and full-season future
opportunity cost. Phase 2 adds a Monte Carlo engine that simulates game
outcomes, public-field eliminations, personal-entry survival, contest equity,
and upset leverage. It does not scrape websites, ingest live feeds, optimize a
40-entry portfolio, or build a dashboard yet.

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

## Run Weekly Rankings

The CLI uses `data/sample/` by default:

```powershell
python scripts/run_weekly_rankings.py --week 1
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
  best-available strategy. Portfolio optimization and diversification are left
  for future phases.
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

Phase 3 portfolio optimizer for 40 entries: allocate picks across controlled
entries with diversification and correlated-risk constraints.

Phase 4 ownership forecasting: project public pick percentages before they are
known or when multiple public sources disagree.

Phase 5 odds/public-pick ingestion: add validated ingestion adapters for real
odds and public-pick exports, without secrets in the repo.

Phase 6 dashboard: build an interactive UI for rankings, reports, scenarios,
and portfolio recommendations.
