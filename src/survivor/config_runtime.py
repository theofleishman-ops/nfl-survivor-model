"""Runtime defaults for operator-facing survivor workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from survivor.config import DATA_DIR, OUTPUTS_DIR


@dataclass(frozen=True)
class RuntimeConfig:
    """Default settings for live weekly operations.

    CLI scripts should use these values as parser defaults so command-line
    flags remain the final override layer.
    """

    season: int = 2026
    sportsbooks: tuple[str, ...] = (
        "fanduel",
        "draftkings",
        "betmgm",
        "caesars",
        "betrivers",
        "espnbet",
    )
    simulation_count: int = 10000
    aggression: str = "balanced"
    entry_count: int = 40
    odds_regions: tuple[str, ...] = ("us",)
    markets: tuple[str, ...] = ("h2h", "spreads", "totals")
    pool_size: int = 5000
    seed: int = 2026
    data_dir: Path = DATA_DIR / "raw"
    reports_dir: Path = OUTPUTS_DIR / "reports"


DEFAULT_RUNTIME_CONFIG = RuntimeConfig()


def parse_csv_option(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse comma-separated CLI/config values into a stable tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_parts = value.split(",")
    else:
        raw_parts = []
        for item in value:
            raw_parts.extend(str(item).split(","))
    return tuple(part.strip() for part in raw_parts if part.strip())


def csv_text(values: Sequence[str]) -> str:
    """Return a compact comma-separated string for provider constructors."""
    return ",".join(str(value).strip() for value in values if str(value).strip())
