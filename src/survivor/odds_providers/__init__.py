"""Odds provider adapters."""

from survivor.odds_providers.base import OddsProvider
from survivor.odds_providers.manual import ManualOddsProvider
from survivor.odds_providers.the_odds_api import TheOddsAPIProvider

__all__ = ["ManualOddsProvider", "OddsProvider", "TheOddsAPIProvider"]
