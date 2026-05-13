"""Canonical NFL team identity helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping


CANONICAL_TEAMS: tuple[str, ...] = (
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LV",
    "LAC",
    "LAR",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
)


_TEAM_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ARI": ("ARI", "AZ", "Arizona", "Cardinals", "Arizona Cardinals"),
    "ATL": ("ATL", "Atlanta", "Falcons", "Atlanta Falcons"),
    "BAL": ("BAL", "Baltimore", "Ravens", "Baltimore Ravens"),
    "BUF": ("BUF", "Buffalo", "Bills", "Buffalo Bills"),
    "CAR": ("CAR", "Carolina", "Panthers", "Carolina Panthers"),
    "CHI": ("CHI", "Chicago", "Bears", "Chicago Bears"),
    "CIN": ("CIN", "Cincinnati", "Bengals", "Cincinnati Bengals"),
    "CLE": ("CLE", "Cleveland", "Browns", "Cleveland Browns"),
    "DAL": ("DAL", "Dallas", "Cowboys", "Dallas Cowboys"),
    "DEN": ("DEN", "Denver", "Broncos", "Denver Broncos"),
    "DET": ("DET", "Detroit", "Lions", "Detroit Lions"),
    "GB": ("GB", "G.B.", "Green Bay", "Packers", "Green Bay Packers"),
    "HOU": ("HOU", "Houston", "Texans", "Houston Texans"),
    "IND": ("IND", "Indianapolis", "Colts", "Indianapolis Colts"),
    "JAX": (
        "JAX",
        "JAC",
        "Jacksonville",
        "Jaguars",
        "Jacksonville Jaguars",
    ),
    "KC": (
        "KC",
        "K.C.",
        "Kansas City",
        "Chiefs",
        "Kansas City Chiefs",
    ),
    "LV": (
        "LV",
        "L.V.",
        "Las Vegas",
        "Raiders",
        "Las Vegas Raiders",
        "Oakland Raiders",
    ),
    "LAC": (
        "LAC",
        "L.A. Chargers",
        "LA Chargers",
        "Los Angeles Chargers",
        "Chargers",
    ),
    "LAR": (
        "LAR",
        "L.A. Rams",
        "LA Rams",
        "Los Angeles Rams",
        "Rams",
        "St. Louis Rams",
        "STL Rams",
    ),
    "MIA": ("MIA", "Miami", "Dolphins", "Miami Dolphins"),
    "MIN": ("MIN", "Minnesota", "Vikings", "Minnesota Vikings"),
    "NE": (
        "NE",
        "N.E.",
        "New England",
        "Patriots",
        "New England Patriots",
    ),
    "NO": ("NO", "N.O.", "New Orleans", "Saints", "New Orleans Saints"),
    "NYG": (
        "NYG",
        "N.Y. Giants",
        "NY Giants",
        "New York Giants",
        "Giants",
    ),
    "NYJ": (
        "NYJ",
        "N.Y. Jets",
        "NY Jets",
        "New York Jets",
        "Jets",
    ),
    "PHI": ("PHI", "Philadelphia", "Eagles", "Philadelphia Eagles"),
    "PIT": ("PIT", "Pittsburgh", "Steelers", "Pittsburgh Steelers"),
    "SEA": ("SEA", "Seattle", "Seahawks", "Seattle Seahawks"),
    "SF": (
        "SF",
        "S.F.",
        "San Francisco",
        "49ers",
        "Forty Niners",
        "Niners",
        "San Francisco 49ers",
    ),
    "TB": (
        "TB",
        "T.B.",
        "Tampa Bay",
        "Buccaneers",
        "Bucs",
        "Tampa Bay Buccaneers",
    ),
    "TEN": ("TEN", "Tennessee", "Titans", "Tennessee Titans"),
    "WAS": (
        "WAS",
        "WSH",
        "Washington",
        "Commanders",
        "Washington Commanders",
        "WFT",
        "Washington Football Team",
    ),
}


def normalize_team_name(value: object) -> str:
    """Return the canonical abbreviation for an NFL team alias.

    The lookup is case-insensitive and ignores punctuation differences, so
    inputs such as ``"K.C."``, ``"Kansas City Chiefs"``, and ``" kc "`` all
    normalize to ``"KC"``.
    """
    cleaned = _clean_alias(value)
    compact = cleaned.replace(" ", "")

    for key in (cleaned, compact):
        if key in _ALIAS_TO_TEAM:
            return _ALIAS_TO_TEAM[key]

    raise ValueError(f"Unknown NFL team alias: {value!r}")


def is_valid_team(value: object) -> bool:
    """Return whether ``value`` can be normalized to a canonical NFL team."""
    try:
        normalize_team_name(value)
    except ValueError:
        return False
    return True


def get_all_team_aliases() -> dict[str, str]:
    """Return the accepted normalized alias keys mapped to canonical teams."""
    return dict(sorted(_ALIAS_TO_TEAM.items()))


def _clean_alias(value: object) -> str:
    if value is None:
        raise ValueError("Team alias cannot be blank.")

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "<na>"}:
        raise ValueError("Team alias cannot be blank.")

    text = text.upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for team, aliases in _TEAM_ALIASES.items():
        if team not in CANONICAL_TEAMS:
            raise ValueError(f"Alias table contains unknown canonical team: {team}")
        for alias in aliases:
            cleaned = _clean_alias(alias)
            for key in {cleaned, cleaned.replace(" ", "")}:
                existing = alias_map.get(key)
                if existing is not None and existing != team:
                    raise ValueError(
                        f"Team alias {alias!r} maps to both {existing} and {team}.",
                    )
                alias_map[key] = team
    return alias_map


_ALIAS_TO_TEAM = _build_alias_map()
