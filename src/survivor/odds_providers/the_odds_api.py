"""Stub adapter for The Odds API.

This module is intentionally API-ready but inactive.  It does not require or
store a key until a caller explicitly tries to fetch provider data.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

import pandas as pd


class TheOddsAPIProvider:
    """Future provider adapter for https://the-odds-api.com."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        sport: str = "americanfootball_nfl",
        regions: str = "us",
    ) -> None:
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        self.sport = sport
        self.regions = regions

    def fetch_events(self) -> Iterable[Mapping[str, object]]:
        """Fetch events from The Odds API once implemented."""
        self._require_api_key()
        raise NotImplementedError("The Odds API event fetching is not implemented yet.")

    def fetch_odds(self) -> Iterable[Mapping[str, object]]:
        """Fetch odds from The Odds API once implemented."""
        self._require_api_key()
        raise NotImplementedError("The Odds API odds fetching is not implemented yet.")

    def normalize(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize fetched odds once the provider adapter is implemented."""
        self._require_api_key()
        raise NotImplementedError("The Odds API normalization is not implemented yet.")

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "THE_ODDS_API_KEY is not set. Provide an API key at runtime via the "
                "environment or constructor; never commit API keys to the repo.",
            )
