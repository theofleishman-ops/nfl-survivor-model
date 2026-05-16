"""Manual CSV and JSON public pick providers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from survivor.public_pick_ingestion import (
    load_public_picks_csv,
    load_public_picks_json,
    normalize_public_pick_records,
)


class ManualPublicPickProvider:
    """Provider for local CSV/JSON public pick files supplied by the user."""

    def __init__(
        self,
        path: str | Path,
        file_format: str | None = None,
        *,
        source: str = "manual_import",
        season: int | None = None,
        week: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.file_format = (file_format or self.path.suffix.lstrip(".")).lower()
        self.source = source
        self.season = season
        self.week = week

    def fetch_public_picks(self) -> list[dict[str, object]]:
        """Load raw public pick records from the configured local file."""
        if self.file_format == "csv":
            records = load_public_picks_csv(self.path)
        elif self.file_format == "json":
            records = load_public_picks_json(self.path)
        else:
            raise ValueError(
                f"Unsupported manual public pick format {self.file_format!r}; "
                "use csv or json.",
            )
        return [self._apply_defaults(record) for record in records]

    def normalize(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Load and normalize the configured public pick file."""
        return normalize_public_pick_records(self.fetch_public_picks(), schedule_df)

    def _apply_defaults(self, record: dict[str, object]) -> dict[str, object]:
        copied = dict(record)
        if self.season is not None and _is_blank(copied.get("season")):
            copied["season"] = self.season
        if self.week is not None and _is_blank(copied.get("week")):
            copied["week"] = self.week
        if self.source and _is_blank(copied.get("source")):
            copied["source"] = self.source
        return copied


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and value.strip() == ""
