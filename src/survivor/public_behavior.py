"""Configurable public-field behavior assumptions.

The presets are deliberately simple scenario inputs, not claims about the real
field.  They let full-season path EV be read as a sensitivity range before
historical pool behavior has been calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass


TEAM_POPULARITY_BIAS = {
    "DAL": 1.35,
    "KC": 1.25,
    "GB": 1.18,
    "PIT": 1.18,
    "SF": 1.18,
    "PHI": 1.16,
    "BUF": 1.14,
    "CHI": 1.10,
    "LV": 1.10,
    "NYG": 1.10,
    "NYJ": 1.10,
    "LAR": 1.08,
    "MIA": 1.08,
    "NE": 1.08,
    "DET": 1.06,
    "BAL": 1.05,
    "CIN": 1.05,
    "MIN": 1.05,
    "SEA": 1.05,
}


@dataclass(frozen=True)
class PublicBehaviorConfig:
    """Behavior parameters for projected public survivor picks.

    ``chalkiness`` controls how aggressively the public follows win
    probability. ``randomness`` mixes in a flat choice component.  ``future``
    and ``scarcity`` terms reduce early use of teams with better future spots.
    ``contrarian_rate`` controls how much of the non-chalk behavior leans away
    from popular teams. ``ownership_temperature`` sharpens or softens the final
    weekly ownership distribution before the optional max-team cap is applied.
    ``clustering_strength`` and related path parameters model how much public
    entries converge onto shared elite future routes beyond weekly ownership.
    """

    chalkiness: float
    randomness: float
    future_awareness: float
    popularity_weight: float
    scarcity_weight: float
    contrarian_rate: float
    max_single_team_ownership: float
    ownership_temperature: float
    clustering_strength: float = 0.0
    elite_path_bias: float = 1.0
    late_season_overlap_weight: float = 2.0
    path_convergence_temperature: float = 1.0
    name: str = "custom"
    description: str = "Custom public behavior assumptions."

    def __post_init__(self) -> None:
        _validate_positive("chalkiness", self.chalkiness)
        _validate_probability("randomness", self.randomness)
        _validate_non_negative("future_awareness", self.future_awareness)
        _validate_non_negative("popularity_weight", self.popularity_weight)
        _validate_non_negative("scarcity_weight", self.scarcity_weight)
        _validate_probability("contrarian_rate", self.contrarian_rate)
        _validate_probability("max_single_team_ownership", self.max_single_team_ownership)
        if float(self.max_single_team_ownership) <= 0:
            raise ValueError("max_single_team_ownership must be greater than 0.")
        _validate_positive("ownership_temperature", self.ownership_temperature)
        _validate_non_negative("clustering_strength", self.clustering_strength)
        _validate_non_negative("elite_path_bias", self.elite_path_bias)
        _validate_positive("late_season_overlap_weight", self.late_season_overlap_weight)
        _validate_positive(
            "path_convergence_temperature",
            self.path_convergence_temperature,
        )


def public_behavior_from_legacy(
    *,
    chalkiness: float,
    future_awareness: float,
    randomness: float,
) -> PublicBehaviorConfig:
    """Return the pre-calibration public-field behavior as a config object."""

    return PublicBehaviorConfig(
        name="legacy_public_field",
        description=(
            "Backwards-compatible behavior used before explicit calibration "
            "presets; randomness maps to the old contrarian blend."
        ),
        chalkiness=float(chalkiness),
        randomness=0.0,
        future_awareness=float(future_awareness),
        popularity_weight=1.0,
        scarcity_weight=0.0,
        contrarian_rate=float(randomness),
        max_single_team_ownership=1.0,
        ownership_temperature=1.0,
        clustering_strength=0.0,
        elite_path_bias=1.0,
        late_season_overlap_weight=2.0,
        path_convergence_temperature=1.0,
    )


def _validate_positive(name: str, value: float) -> None:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative(name: str, value: float) -> None:
    if float(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


PUBLIC_BEHAVIOR_PRESETS: dict[str, PublicBehaviorConfig] = {
    "chalk_heavy": PublicBehaviorConfig(
        name="chalk_heavy",
        description=(
            "Public follows the obvious favorites, strongly respects listed "
            "ownership, and allows crowded teams to approach chalk levels."
        ),
        chalkiness=4.2,
        randomness=0.02,
        future_awareness=0.10,
        popularity_weight=1.35,
        scarcity_weight=0.55,
        contrarian_rate=0.03,
        max_single_team_ownership=0.70,
        ownership_temperature=0.82,
        clustering_strength=0.45,
        elite_path_bias=1.45,
        late_season_overlap_weight=2.80,
        path_convergence_temperature=0.72,
    ),
    "balanced_public": PublicBehaviorConfig(
        name="balanced_public",
        description=(
            "Base case: public prefers favorites and known ownership, sometimes "
            "moves away from chalk, and weakly conserves future teams."
        ),
        chalkiness=3.0,
        randomness=0.04,
        future_awareness=0.25,
        popularity_weight=1.0,
        scarcity_weight=1.0,
        contrarian_rate=0.08,
        max_single_team_ownership=0.58,
        ownership_temperature=1.0,
        clustering_strength=0.28,
        elite_path_bias=1.20,
        late_season_overlap_weight=2.20,
        path_convergence_temperature=0.85,
    ),
    "contrarian_public": PublicBehaviorConfig(
        name="contrarian_public",
        description=(
            "Field is less concentrated, less driven by brand popularity, and "
            "more willing to choose viable lower-owned alternatives."
        ),
        chalkiness=2.1,
        randomness=0.10,
        future_awareness=0.25,
        popularity_weight=0.45,
        scarcity_weight=1.0,
        contrarian_rate=0.28,
        max_single_team_ownership=0.40,
        ownership_temperature=1.25,
        clustering_strength=0.10,
        elite_path_bias=0.80,
        late_season_overlap_weight=1.60,
        path_convergence_temperature=1.05,
    ),
    "future_aware_public": PublicBehaviorConfig(
        name="future_aware_public",
        description=(
            "Public is comparatively careful about preserving elite teams for "
            "later scarce weeks, muting early ownership of teams with stronger "
            "future spots."
        ),
        chalkiness=2.6,
        randomness=0.04,
        future_awareness=1.25,
        popularity_weight=0.80,
        scarcity_weight=1.70,
        contrarian_rate=0.08,
        max_single_team_ownership=0.48,
        ownership_temperature=1.05,
        clustering_strength=0.22,
        elite_path_bias=1.05,
        late_season_overlap_weight=2.40,
        path_convergence_temperature=0.90,
    ),
    "naive_public": PublicBehaviorConfig(
        name="naive_public",
        description=(
            "Public mostly reacts to current-week win probability and simple "
            "popularity cues, with little or no future-team conservation."
        ),
        chalkiness=3.3,
        randomness=0.12,
        future_awareness=0.0,
        popularity_weight=0.70,
        scarcity_weight=0.0,
        contrarian_rate=0.03,
        max_single_team_ownership=0.62,
        ownership_temperature=1.12,
        clustering_strength=0.35,
        elite_path_bias=1.10,
        late_season_overlap_weight=2.00,
        path_convergence_temperature=0.82,
    ),
}

PUBLIC_BEHAVIOR_PRESET_ORDER = (
    "chalk_heavy",
    "balanced_public",
    "contrarian_public",
    "future_aware_public",
    "naive_public",
)

DEFAULT_PUBLIC_BEHAVIOR_CONFIG = PUBLIC_BEHAVIOR_PRESETS["balanced_public"]


def get_public_behavior_config(name: str) -> PublicBehaviorConfig:
    """Return a named public behavior preset."""

    key = str(name).strip().lower().replace("-", "_")
    try:
        return PUBLIC_BEHAVIOR_PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(PUBLIC_BEHAVIOR_PRESET_ORDER)
        raise ValueError(f"unknown public behavior preset '{name}'. Choose one of: {choices}.") from exc


def resolve_public_behavior_config(
    value: PublicBehaviorConfig | str | None,
) -> PublicBehaviorConfig:
    """Coerce a preset name or config object into a config."""

    if value is None:
        return DEFAULT_PUBLIC_BEHAVIOR_CONFIG
    if isinstance(value, PublicBehaviorConfig):
        return value
    return get_public_behavior_config(str(value))
