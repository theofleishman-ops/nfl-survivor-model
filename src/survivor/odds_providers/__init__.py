"""Odds provider adapters."""

from survivor.odds_providers.base import OddsProvider
from survivor.odds_providers.manual import ManualOddsProvider

__all__ = ["ManualOddsProvider", "OddsProvider"]
