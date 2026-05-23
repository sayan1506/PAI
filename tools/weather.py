"""
tools/weather.py

WeatherTool — fetches current weather from OpenWeatherMap's free API.

Single action: get_weather
  Required param: location (city name, "City,CountryCode", or ZIP)
  Optional param: units ("metric" | "imperial") — overrides config default

API docs: https://openweathermap.org/current
Free tier: 60 calls/minute, 1 000 000 calls/month — sufficient for personal use.
"""

import requests

import config
from tools.base import BaseTool, ToolResult
from core.logger import logger

_OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

# Human-readable unit labels
_TEMP_UNIT = {"metric": "°C", "imperial": "°F"}
_SPEED_UNIT = {"metric": "m/s", "imperial": "mph"}


class WeatherTool(BaseTool):

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Get the current weather for any city. "
            "Returns temperature, weather conditions, humidity, and wind speed "
            "in a single sentence suitable for speaking aloud. "
            "Use action='get_weather' with a 'location' parameter. "
            "Location can be a city name ('London'), 'City,CountryCode' "
            "('London,GB'), or a US ZIP code ('10001'). "
            "Optionally pass units='metric' (°C) or units='imperial' (°F) "
            "to override the configured default."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_weather"],
                    "description": "Must be 'get_weather'.",
                },
                "location": {
                    "type": "string",
                    "description": (
                        "City name, 'City,CountryCode', or US ZIP code. "
                        "Example: 'Tokyo', 'Paris,FR', '10001'."
                    ),
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": (
                        "Temperature units. Defaults to config WEATHER_UNITS."
                    ),
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "")
        if action != "get_weather":
            return ToolResult(
                success=False,
                error=f"Unknown action: {action!r}. Only 'get_weather' is supported.",
            )
        return self._get_weather(
            location=kwargs.get("location", ""),
            units=kwargs.get("units", config.WEATHER_UNITS),
        )

    def _get_weather(self, location: str, units: str) -> ToolResult:
        # 1. Check API key
        if not config.WEATHER_API_KEY:
            return ToolResult(
                success=False,
                error=(
                    "WEATHER_API_KEY is not set. Get a free key at "
                    "https://openweathermap.org/api and add it to your .env file."
                ),
            )

        # 2. Resolve location
        loc = location.strip() or config.WEATHER_DEFAULT_LOCATION.strip()
        if not loc:
            return ToolResult(
                success=False,
                error=(
                    "No location provided and WEATHER_DEFAULT_LOCATION is not set. "
                    "Provide a city name (e.g. 'London')."
                ),
            )

        # 3. Fetch
        try:
            resp = requests.get(
                _OWM_URL,
                params={"q": loc, "appid": config.WEATHER_API_KEY, "units": units},
                timeout=8,
            )
        except requests.exceptions.Timeout:
            return ToolResult(success=False, error="Weather request timed out.")
        except requests.exceptions.RequestException as e:
            return ToolResult(success=False, error=f"Weather request failed: {e}")

        # 4. HTTP errors
        if resp.status_code == 401:
            return ToolResult(
                success=False,
                error="Invalid WEATHER_API_KEY. Check your key at openweathermap.org.",
            )
        if resp.status_code == 404:
            return ToolResult(
                success=False,
                error=f"Location not found: '{loc}'. Try 'City,CountryCode' format.",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"OpenWeatherMap returned HTTP {resp.status_code}.",
            )

        # 5. Parse
        try:
            data = resp.json()
            city = data["name"]
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"WeatherTool: unexpected response structure: {e}")
            return ToolResult(
                success=False,
                error="Could not parse weather response. Unexpected API format.",
            )

        # 6. Format
        t_unit = _TEMP_UNIT.get(units, "°C")
        s_unit = _SPEED_UNIT.get(units, "m/s")
        output = (
            f"Currently in {city}: {description}, "
            f"{temp:.1f}{t_unit}, "
            f"humidity {humidity}%, "
            f"wind {wind_speed:.1f} {s_unit}."
        )
        logger.info(f"WeatherTool: {output}")
        return ToolResult(success=True, output=output)
