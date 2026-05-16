"""Provider interface for public pick ingestion adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

import pandas as pd


class PublicPickProvider(Protocol):
    """Protocol implemented by public survivor pick providers."""

    source: str

    def fetch_public_picks(self) -> Iterable[Mapping[str, object]]:
        """Fetch raw public pick records from the provider."""

    def normalize(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Return public picks normalized to the survivor ownership schema."""
