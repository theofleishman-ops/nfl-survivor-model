"""The Odds API provider adapter.

The adapter fetches the current NFL odds endpoint, flattens provider responses
into Phase 8 raw odds records, then delegates schema coercion, schedule joins,
and probability calculations to :mod:`survivor.odds_ingestion`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from survivor.odds_ingestion import normalize_odds_records
from survivor.teams import normalize_team_name


API_KEY_ENV_VAR = "THE_ODDS_API_KEY"
DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
SUPPORTED_MARKETS = ("h2h", "spreads", "totals")

HttpGet = Callable[[str, Mapping[str, str], float], Any]


class TheOddsAPIProvider:
    """Provider adapter for The Odds API's NFL odds endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        sport: str = NFL_SPORT_KEY,
        season: int | None = None,
        week: int | None = None,
        regions: str | Sequence[str] = "us",
        markets: str | Sequence[str] = SUPPORTED_MARKETS,
        odds_format: str = "american",
        bookmakers: str | Sequence[str] | None = None,
        timeout: float = 30.0,
        base_url: str = DEFAULT_BASE_URL,
        http_get: HttpGet | None = None,
        save_raw_path: str | Path | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv(API_KEY_ENV_VAR)
        self.sport = sport
        self.season = season
        self.week = week
        self.regions = _csv_text(regions)
        self.markets, self.unsupported_markets = _normalize_markets(markets)
        self.odds_format = odds_format.strip().lower()
        self.bookmakers = _csv_text(bookmakers) if bookmakers is not None else None
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.http_get = http_get or _default_http_get
        self.save_raw_path = Path(save_raw_path) if save_raw_path is not None else None

    def fetch_events(self) -> Iterable[Mapping[str, object]]:
        """Fetch NFL event metadata from The Odds API."""
        self._require_api_key()
        payload = self.http_get(
            self._url(f"/sports/{self.sport}/events"),
            {"apiKey": str(self.api_key), "dateFormat": "iso"},
            self.timeout,
        )
        return _ensure_event_list(payload)

    def fetch_odds(self) -> Iterable[Mapping[str, object]]:
        """Fetch current NFL odds from The Odds API."""
        self._require_api_key()
        if not self.markets:
            raise ValueError(
                "No supported The Odds API markets requested. Supported markets: "
                f"{', '.join(SUPPORTED_MARKETS)}.",
            )
        if self.odds_format not in {"american", "decimal"}:
            raise ValueError(
                "The Odds API odds format must be 'american' or 'decimal'. "
                f"Got {self.odds_format!r}.",
            )

        params = {
            "apiKey": str(self.api_key),
            "regions": self.regions,
            "markets": ",".join(self.markets),
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
        }
        if self.bookmakers:
            params["bookmakers"] = self.bookmakers

        payload = self.http_get(
            self._url(f"/sports/{self.sport}/odds/"),
            params,
            self.timeout,
        )
        if self.save_raw_path is not None:
            _save_raw_payload(payload, self.save_raw_path)
        return _ensure_event_list(payload)

    def normalize(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Return The Odds API odds normalized to the survivor odds schema."""
        records = self.records_from_events(self.fetch_odds(), schedule_df=schedule_df)
        if not records:
            raise ValueError(
                "The Odds API returned no supported odds rows matching the supplied "
                "schedule. Confirm NFL odds are posted for the requested season/week "
                "and that the requested markets or sportsbooks are available.",
            )
        return normalize_odds_records(records, schedule_df)

    def records_from_events(
        self,
        events: Iterable[Mapping[str, object]],
        *,
        schedule_df: pd.DataFrame | None = None,
    ) -> list[dict[str, object]]:
        """Flatten The Odds API events into provider-agnostic raw odds records."""
        schedule_lookup = _schedule_lookup(schedule_df, season=self.season, week=self.week)
        records: list[dict[str, object]] = []
        for event in events:
            records.extend(self._records_from_event(event, schedule_lookup))
        return records

    def _records_from_event(
        self,
        event: Mapping[str, object],
        schedule_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    ) -> list[dict[str, object]]:
        home_team = _normalize_provider_team(event.get("home_team"))
        away_team = _normalize_provider_team(event.get("away_team"))
        schedule_game = schedule_lookup.get((home_team, away_team))
        if schedule_lookup and schedule_game is None:
            return []

        game_context = {
            "season": _context_value(schedule_game, "season", self.season),
            "week": _context_value(schedule_game, "week", self.week),
            "game_id": _context_value(schedule_game, "game_id", None),
            "home_team": home_team,
            "away_team": away_team,
        }
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            return []

        records: list[dict[str, object]] = []
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, Mapping):
                continue
            records.extend(self._records_from_bookmaker(bookmaker, game_context))
        return records

    def _records_from_bookmaker(
        self,
        bookmaker: Mapping[str, object],
        game_context: Mapping[str, object],
    ) -> list[dict[str, object]]:
        markets = bookmaker.get("markets")
        if not isinstance(markets, list):
            return []

        records: list[dict[str, object]] = []
        for market in markets:
            if not isinstance(market, Mapping):
                continue
            market_key = str(market.get("key", "")).strip().lower()
            if market_key not in SUPPORTED_MARKETS:
                continue
            if market_key not in self.markets:
                continue

            if market_key == "totals":
                records.extend(
                    self._total_records_from_market(bookmaker, market, game_context),
                )
            else:
                records.extend(
                    self._team_records_from_market(bookmaker, market, game_context),
                )
        return records

    def _team_records_from_market(
        self,
        bookmaker: Mapping[str, object],
        market: Mapping[str, object],
        game_context: Mapping[str, object],
    ) -> list[dict[str, object]]:
        market_key = str(market.get("key", "")).strip().lower()
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            return []

        records: list[dict[str, object]] = []
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                continue
            team = _provider_team_or_none(outcome.get("name"))
            if team is None:
                continue
            opponent = _opponent_for(team, game_context)
            if opponent is None:
                continue

            record = self._base_record(bookmaker, market, game_context)
            record["market_type"] = market_key
            record["team"] = team
            record["opponent"] = opponent
            price = _price_to_american(outcome.get("price"), self.odds_format)
            if market_key == "h2h":
                record["moneyline"] = price
            elif market_key == "spreads":
                record["spread"] = _float_or_none(outcome.get("point"))
                record["spread_price"] = price
            records.append(record)
        return records

    def _total_records_from_market(
        self,
        bookmaker: Mapping[str, object],
        market: Mapping[str, object],
        game_context: Mapping[str, object],
    ) -> list[dict[str, object]]:
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            return []

        over = _find_total_outcome(outcomes, "over")
        under = _find_total_outcome(outcomes, "under")
        total = _first_total_point(over, under)
        records: list[dict[str, object]] = []

        if over is not None:
            record = self._base_record(bookmaker, market, game_context)
            record["market_type"] = "totals"
            record["team"] = game_context["home_team"]
            record["opponent"] = game_context["away_team"]
            record["total"] = total
            record["total_price"] = _price_to_american(over.get("price"), self.odds_format)
            record["source"] = "the_odds_api:totals_over"
            records.append(record)

        if under is not None:
            record = self._base_record(bookmaker, market, game_context)
            record["market_type"] = "totals"
            record["team"] = game_context["away_team"]
            record["opponent"] = game_context["home_team"]
            record["total"] = total
            record["total_price"] = _price_to_american(
                under.get("price"),
                self.odds_format,
            )
            record["source"] = "the_odds_api:totals_under"
            records.append(record)

        return records

    def _base_record(
        self,
        bookmaker: Mapping[str, object],
        market: Mapping[str, object],
        game_context: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            **dict(game_context),
            "sportsbook": _sportsbook_key(bookmaker),
            "pulled_at": _text_or_default(
                market.get("last_update") or bookmaker.get("last_update"),
                "",
            ),
            "source": "the_odds_api",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                f"{API_KEY_ENV_VAR} is not set. Set it in the runtime environment "
                "before using --provider the-odds-api. Never commit API keys or "
                "raw secret-bearing command lines to the repo.",
            )


def _default_http_get(url: str, params: Mapping[str, str], timeout: float) -> Any:
    query = urlencode(params)
    request = Request(
        f"{url}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "nfl-survivor-model/0.1",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "The Odds API request failed with "
            f"HTTP {exc.code}: {_provider_error_text(body)}",
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"The Odds API request failed: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The Odds API returned invalid JSON.") from exc


def _ensure_event_list(payload: Any) -> list[Mapping[str, object]]:
    if not isinstance(payload, list):
        raise ValueError("The Odds API response must be a JSON list of events.")
    events: list[Mapping[str, object]] = []
    for event in payload:
        if not isinstance(event, Mapping):
            raise ValueError(f"The Odds API event must be an object: {event!r}")
        events.append(event)
    return events


def _normalize_markets(
    markets: str | Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requested = _csv_parts(markets)
    supported: list[str] = []
    unsupported: list[str] = []
    for market in requested:
        normalized = _market_alias(market)
        if normalized is None:
            unsupported.append(market)
            continue
        if normalized not in supported:
            supported.append(normalized)
    return tuple(supported), tuple(unsupported)


def _market_alias(value: str) -> str | None:
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    aliases = {
        "h2h": "h2h",
        "headtohead": "h2h",
        "moneyline": "h2h",
        "ml": "h2h",
        "spread": "spreads",
        "spreads": "spreads",
        "pointspread": "spreads",
        "totals": "totals",
        "total": "totals",
        "overunder": "totals",
    }
    return aliases.get(normalized)


def _csv_parts(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = []
        for part in value:
            items.extend(str(part).split(","))
    return [item.strip() for item in items if item.strip()]


def _csv_text(value: str | Sequence[str]) -> str:
    return ",".join(_csv_parts(value))


def _schedule_lookup(
    schedule_df: pd.DataFrame | None,
    *,
    season: int | None,
    week: int | None,
) -> dict[tuple[str, str], Mapping[str, object]]:
    if schedule_df is None or schedule_df.empty:
        return {}

    schedule = schedule_df.copy()
    if "season" in schedule.columns and season is not None:
        seasons = pd.to_numeric(schedule["season"], errors="coerce")
        schedule = schedule[seasons == int(season)]
    if week is not None:
        weeks = pd.to_numeric(schedule["week"], errors="coerce")
        schedule = schedule[weeks == int(week)]

    lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for _, row in schedule.iterrows():
        home = _normalize_provider_team(row.get("home_team"))
        away = _normalize_provider_team(row.get("away_team"))
        key = (home, away)
        if key in lookup:
            continue
        lookup[key] = {
            "season": _int_or_none(row.get("season")),
            "week": _int_or_none(row.get("week")),
            "game_id": _text_or_default(row.get("game_id"), ""),
        }
    return lookup


def _normalize_provider_team(value: object) -> str:
    try:
        return normalize_team_name(value)
    except ValueError as exc:
        raise ValueError(f"Could not normalize The Odds API team {value!r}.") from exc


def _provider_team_or_none(value: object) -> str | None:
    try:
        return normalize_team_name(value)
    except ValueError:
        return None


def _opponent_for(
    team: str,
    game_context: Mapping[str, object],
) -> str | None:
    home_team = str(game_context["home_team"])
    away_team = str(game_context["away_team"])
    if team == home_team:
        return away_team
    if team == away_team:
        return home_team
    return None


def _find_total_outcome(
    outcomes: Sequence[object],
    side: str,
) -> Mapping[str, object] | None:
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        name = str(outcome.get("name", "")).strip().lower()
        if name == side:
            return outcome
    return None


def _first_total_point(
    over: Mapping[str, object] | None,
    under: Mapping[str, object] | None,
) -> float | None:
    for outcome in (over, under):
        if outcome is not None:
            value = _float_or_none(outcome.get("point"))
            if value is not None:
                return value
    return None


def _price_to_american(value: object, odds_format: str) -> float | None:
    price = _float_or_none(value)
    if price is None:
        return None
    if odds_format == "american":
        return price
    if price <= 1:
        raise ValueError(f"Decimal odds must be greater than 1. Got {value!r}.")
    if price >= 2:
        return float(round((price - 1) * 100))
    return float(round(-100 / (price - 1)))


def _float_or_none(value: object) -> float | None:
    if _is_blank(value):
        return None
    return float(pd.to_numeric(value, errors="raise"))


def _int_or_none(value: object) -> int | None:
    if _is_blank(value):
        return None
    return int(pd.to_numeric(value, errors="raise"))


def _text_or_default(value: object, default: str) -> str:
    if _is_blank(value):
        return default
    return str(value).strip()


def _context_value(
    mapping: Mapping[str, object] | None,
    key: str,
    default: object,
) -> object:
    if mapping is None or _is_blank(mapping.get(key)):
        return default
    return mapping[key]


def _sportsbook_key(bookmaker: Mapping[str, object]) -> str:
    return _text_or_default(bookmaker.get("key"), _text_or_default(bookmaker.get("title"), "UNKNOWN"))


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and value.strip() == ""


def _provider_error_text(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:500] or "no response body"
    if isinstance(payload, Mapping):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if value:
                return str(value)[:500]
    return body.strip()[:500] or "no response body"


def _save_raw_payload(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as raw_file:
        json.dump(payload, raw_file, indent=2, sort_keys=True)
        raw_file.write("\n")
