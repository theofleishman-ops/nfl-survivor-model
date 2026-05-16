"""Manual CSV and JSON odds providers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from survivor.odds_ingestion import load_odds_csv, load_odds_json, normalize_odds_records


class ManualOddsProvider:
    """Provider for local CSV/JSON files supplied by the user."""

    def __init__(self, path: str | Path, file_format: str | None = None) -> None:
        self.path = Path(path)
        self.file_format = (file_format or self.path.suffix.lstrip(".")).lower()

    def fetch_events(self) -> list[dict[str, object]]:
        """Manual files do not have a separate event endpoint."""
        return []

    def fetch_odds(self) -> list[dict[str, object]]:
        """Load raw odds records from the configured local file."""
        if self.file_format == "csv":
            return load_odds_csv(self.path)
        if self.file_format == "json":
            return load_odds_json(self.path)
        raise ValueError(
            f"Unsupported manual odds format {self.file_format!r}; use csv or json.",
        )

    def normalize(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Load and normalize the configured local odds file."""
        return normalize_odds_records(self.fetch_odds(), schedule_df)
