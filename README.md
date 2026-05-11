# nfl-survivor-model

Advanced NFL survivor pool optimizer foundation.

This repository will grow into a tool for evaluating survivor pool picks using
win probabilities, public pick behavior, future value, leverage, and portfolio
construction. The initial scaffold is intentionally small: it uses fake sample
CSV data, basic loaders, and a few tests so future modeling work has a clean
place to land.

No real odds, real pool data, secrets, API keys, or scraping code are included.

## Install

Create and activate a virtual environment, then install the starter
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Tests

```powershell
pytest
```

The test suite validates the starter odds utilities and confirms the sample CSV
files can be loaded.

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
src/survivor/   Python package
tests/          Pytest suite
scripts/        Future command-line helpers
```

## Roadmap

Phase 1 CSV prototype: establish data contracts and a minimal local workflow
using sample CSV inputs.

Phase 2 odds ingestion: add validated odds-source adapters and no-vig market
conversion.

Phase 3 public pick ingestion: ingest and validate public pick distributions
from approved sources or manual exports.

Phase 4 full-season optimization: model weekly survival probability, future
availability, and pick constraints across the full schedule.

Phase 5 portfolio EV engine: optimize multiple entries with diversification,
leverage, and expected value tradeoffs.

Phase 6 Streamlit dashboard: expose reports, charts, scenarios, and portfolio
recommendations in an interactive UI.
