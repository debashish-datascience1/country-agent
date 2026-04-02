"""Unit tests for the REST Countries API client."""
import pytest
import httpx
import respx

from app.agent.tools import fetch_country, CountryNotFoundError, CountryAPIError

GERMANY_PAYLOAD = [
    {
        "name": {"common": "Germany", "official": "Federal Republic of Germany"},
        "capital": ["Berlin"],
        "population": 83240525,
        "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
        "languages": {"deu": "German"},
        "region": "Europe",
        "flags": {"svg": "https://example.com/de.svg"},
    }
]


@pytest.mark.asyncio
@respx.mock
async def test_fetch_country_success():
    respx.get("https://restcountries.com/v3.1/name/Germany").mock(
        return_value=httpx.Response(200, json=GERMANY_PAYLOAD)
    )
    async with httpx.AsyncClient() as client:
        result = await fetch_country("Germany", client)
    assert result["name"]["common"] == "Germany"
    assert result["population"] == 83240525


@pytest.mark.asyncio
@respx.mock
async def test_fetch_country_not_found():
    respx.get("https://restcountries.com/v3.1/name/Xlandia").mock(
        return_value=httpx.Response(404, json={"status": 404, "message": "Not Found"})
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(CountryNotFoundError):
            await fetch_country("Xlandia", client)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_country_api_error():
    respx.get("https://restcountries.com/v3.1/name/Germany").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(CountryAPIError):
            await fetch_country("Germany", client)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_country_timeout():
    respx.get("https://restcountries.com/v3.1/name/Germany").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(CountryAPIError, match="timed out"):
            await fetch_country("Germany", client)
