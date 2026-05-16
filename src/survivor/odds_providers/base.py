"""Provider interface for odds ingestion adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

import pandas as pd


class OddsProvider(Protocol):
    """Protocol implemented by all odds ingestion providers."""

    def fetch_events(self) -> Iterable[Mapping[str, object]]:
        """Fetch provider event metadata, if the provider exposes it."""

    def fetch_odds(self) -> Iterable[Mapping[str, object]]:
        """Fetch raw provider odds records."""

    def normalize(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Return provider odds normalized to survivor odds schema."""
