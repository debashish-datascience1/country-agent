"""REST Countries API client."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_REST_COUNTRIES_URL = "https://restcountries.com/v3.1/name/{country}"

# Fields we care about — avoids fetching the full 50+ field payload
_FIELDS = ",".join([
    "name",
    "capital",
    "population",
    "area",
    "currencies",
    "languages",
    "flags",
    "region",
    "subregion",
    "timezones",
    "borders",
    "cca2",
    "cca3",
    "latlng",
    "continents",
    "independent",
    "unMember",
])


class CountryNotFoundError(Exception):
    pass


class CountryAPIError(Exception):
    pass


async def fetch_country(country_name: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """
    Fetch country data from the REST Countries API.

    Strategy:
      1. Try fullText=true (exact name match) — correctly returns China, India, etc.
      2. If not found, fall back to partial search and pick the closest match
         by scoring each result's common name against the query.

    Raises:
        CountryNotFoundError: if the country cannot be found by either strategy.
        CountryAPIError: on network error, timeout, or unexpected HTTP status.
    """
    # --- Step 1: exact match ---
    try:
        results = await _get_results(country_name, client, full_text=True)
        return results[0]
    except CountryNotFoundError:
        pass  # fall through to partial search

    # --- Step 2: partial search + best-match selection ---
    results = await _get_results(country_name, client, full_text=False)
    return _best_match(results, country_name)


async def _get_results(
    country_name: str,
    client: httpx.AsyncClient,
    full_text: bool,
) -> list[dict[str, Any]]:
    """Raw API call — returns the full list of matching results."""
    url = _REST_COUNTRIES_URL.format(country=country_name)
    params: dict[str, Any] = {"fields": _FIELDS}
    if full_text:
        params["fullText"] = "true"

    try:
        response = await client.get(url, params=params)
    except httpx.TimeoutException as exc:
        logger.warning("Timeout fetching '%s': %s", country_name, exc)
        raise CountryAPIError(f"Request timed out for '{country_name}'") from exc
    except httpx.RequestError as exc:
        logger.warning("Network error fetching '%s': %s", country_name, exc)
        raise CountryAPIError(f"Network error for '{country_name}'") from exc

    if response.status_code == 404:
        raise CountryNotFoundError(f"Country '{country_name}' not found")

    if response.status_code != 200:
        logger.warning(
            "Unexpected HTTP %d from REST Countries API for '%s'",
            response.status_code,
            country_name,
        )
        raise CountryAPIError(
            f"REST Countries API returned HTTP {response.status_code} for '{country_name}'"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise CountryAPIError("Failed to parse API response as JSON") from exc

    if not data or not isinstance(data, list):
        raise CountryNotFoundError(f"No data returned for '{country_name}'")

    return data


def _best_match(results: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """
    Pick the result whose common name best matches the search query.

    Scoring (highest wins):
      3 — exact case-insensitive match  (query == common name)
      2 — common name starts with query
      1 — query is a substring of common name
      0 — no match (keep as fallback)
    """
    query_lower = query.strip().lower()

    def score(country: dict) -> int:
        common = (country.get("name") or {}).get("common", "").lower()
        if common == query_lower:
            return 3
        if common.startswith(query_lower):
            return 2
        if query_lower in common:
            return 1
        return 0

    best = max(results, key=score)
    logger.info(
        "Best match for '%s': '%s' (score %d)",
        query,
        (best.get("name") or {}).get("common"),
        score(best),
    )
    return best
