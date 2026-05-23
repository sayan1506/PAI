"""
tests/tools/test_weather.py

Unit tests for WeatherTool — all HTTP calls mocked.
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from tools.weather import WeatherTool
from tools.base import ToolResult


@pytest.fixture
def tool():
    return WeatherTool()


# ─── 1. Metadata ─────────────────────────────────────────────────────────────

def test_weather_tool_name_and_description(tool):
    assert tool.name == "weather"
    assert "weather" in tool.description.lower()
    assert "get_weather" in tool.description


# ─── 2. Unknown action ───────────────────────────────────────────────────────

def test_weather_unknown_action_returns_failure(tool):
    result = tool.execute(action="forecast")
    assert result.success is False
    assert "Unknown action" in result.error


# ─── 3. Missing API key ─────────────────────────────────────────────────────

@patch("tools.weather.config")
def test_weather_missing_api_key_returns_helpful_error(mock_config, tool):
    mock_config.WEATHER_API_KEY = ""
    mock_config.WEATHER_UNITS = "metric"
    result = tool.execute(action="get_weather", location="London")
    assert result.success is False
    assert "WEATHER_API_KEY" in result.error
    assert "openweathermap.org" in result.error


# ─── 4. Missing location and default ────────────────────────────────────────

@patch("tools.weather.config")
def test_weather_missing_location_and_default_returns_error(mock_config, tool):
    mock_config.WEATHER_API_KEY = "test"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""
    result = tool.execute(action="get_weather", location="")
    assert result.success is False
    assert "location" in result.error.lower()


# ─── 5. Uses default location ───────────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_uses_default_location_when_none_provided(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = "Berlin"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": "Berlin",
        "main": {"temp": 15.0, "humidity": 50},
        "weather": [{"description": "cloudy"}],
        "wind": {"speed": 2.0},
    }
    mock_get.return_value = mock_resp

    tool.execute(action="get_weather", location="")

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
    assert params["q"] == "Berlin"


# ─── 6. Formatted sentence — metric ─────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_returns_formatted_sentence_metric(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": "Paris",
        "main": {"temp": 18.3, "humidity": 65},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.2},
    }
    mock_get.return_value = mock_resp

    result = tool.execute(action="get_weather", location="Paris")
    assert result.success is True
    assert "Paris" in result.output
    assert "18.3°C" in result.output
    assert "clear sky" in result.output
    assert "65%" in result.output
    assert "3.2 m/s" in result.output


# ─── 7. Formatted sentence — imperial ───────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_returns_formatted_sentence_imperial(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "imperial"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": "Paris",
        "main": {"temp": 18.3, "humidity": 65},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.2},
    }
    mock_get.return_value = mock_resp

    result = tool.execute(action="get_weather", location="Paris", units="imperial")
    assert result.success is True
    assert "°F" in result.output
    assert "mph" in result.output


# ─── 8. HTTP 404 ─────────────────────────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_404_returns_location_not_found_error(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    result = tool.execute(action="get_weather", location="Xyzzyville")
    assert result.success is False
    assert "not found" in result.error.lower()


# ─── 9. HTTP 401 ─────────────────────────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_401_returns_invalid_key_error(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp

    result = tool.execute(action="get_weather", location="London")
    assert result.success is False
    assert "WEATHER_API_KEY" in result.error


# ─── 10. Timeout ─────────────────────────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_timeout_returns_timeout_error(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    result = tool.execute(action="get_weather", location="London")
    assert result.success is False
    assert "timed out" in result.error.lower()


# ─── 11. Unexpected HTTP status ──────────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_unexpected_http_status_returns_error(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_get.return_value = mock_resp

    result = tool.execute(action="get_weather", location="London")
    assert result.success is False
    assert "500" in result.error


# ─── 12. Malformed JSON ──────────────────────────────────────────────────────

@patch("tools.weather.requests.get")
@patch("tools.weather.config")
def test_weather_malformed_json_returns_parse_error(mock_config, mock_get, tool):
    mock_config.WEATHER_API_KEY = "test-key"
    mock_config.WEATHER_UNITS = "metric"
    mock_config.WEATHER_DEFAULT_LOCATION = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": "London",
        "main": {"temp": 10.0, "humidity": 80},
        # "weather" key is missing — triggers parse error
        "wind": {"speed": 5.0},
    }
    mock_get.return_value = mock_resp

    result = tool.execute(action="get_weather", location="London")
    assert result.success is False
    assert "parse" in result.error.lower() or "format" in result.error.lower()
