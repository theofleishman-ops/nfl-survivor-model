# Real 2026 Week 1 Fixture

This fixture uses real 2026 Week 1 schedule rows and canonical `game_id`
values from `data/raw/2026/schedule.csv`.

The odds and public-pick values are synthetic placeholders. They are included
only so the ranking, simulation, and portfolio CLIs can exercise the real-data
join path end to end without scraping, APIs, or private pool exports.

`public_picks_consensus.csv` demonstrates the richer consensus ownership schema
produced from the synthetic source files in `data/sample/public_pick_sources/`.
The main `public_picks.csv` stays in the compact workflow schema expected by
the current sample CLIs.

Replace `odds.csv` and `public_picks.csv` with real values before making real
contest decisions.
