"""
Weather service using Open-Meteo (free, no API key required).
- Historical: open-meteo archive API (2021-present)
- Forecast: open-meteo forecast API (today + 7 days)
- Weather code → human label mapping included
"""

import requests
from datetime import date, timedelta
from typing import Optional

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}

def wmo_label(code: int) -> str:
    return WMO_CODES.get(code, f"Code {code}")

def weather_impact(code: int, temp_max: float) -> str:
    """Classify demand impact from weather."""
    if code in (95, 96, 99):    return "very_low"
    if code in (65, 82, 86, 99): return "low"
    if code in (61, 63, 80, 81): return "slight_down"
    if temp_max is not None and temp_max >= 44: return "low"
    if temp_max is not None and temp_max >= 40: return "slight_down"
    return "neutral"

def get_historical_weather(lat: float, lon: float,
                           start_date: str, end_date: str) -> list[dict]:
    """
    Returns daily weather for a lat/lon from archive.
    start_date / end_date: 'YYYY-MM-DD'
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weathercode",
        ],
        "timezone": "Asia/Kolkata",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=15)
        if r.status_code == 403:
            return [{"error": "Open-Meteo blocked in this environment. Will work on your local server.", "date": start_date}]
        r.raise_for_status()
        data = r.json()["daily"]
        return [
            {
                "date":          data["time"][i],
                "type":          "historical",
                "temp_max_c":    round(data["temperature_2m_max"][i], 1) if data["temperature_2m_max"][i] is not None else None,
                "temp_min_c":    round(data["temperature_2m_min"][i], 1) if data["temperature_2m_min"][i] is not None else None,
                "precipitation_mm": round(data["precipitation_sum"][i], 1) if data["precipitation_sum"][i] is not None else None,
                "weather_code":  data["weathercode"][i],
                "weather_label": wmo_label(data["weathercode"][i]) if data["weathercode"][i] is not None else "Unknown",
                "demand_impact": weather_impact(
                    data["weathercode"][i] or 0,
                    data["temperature_2m_max"][i]
                ),
            }
            for i in range(len(data["time"]))
        ]
    except Exception as e:
        return [{"error": str(e), "start_date": start_date, "end_date": end_date}]


def get_forecast_weather(lat: float, lon: float, days: int = 8) -> list[dict]:
    """
    Returns daily forecast for today + next `days` days.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weathercode",
            "precipitation_probability_max",
        ],
        "forecast_days": days,
        "timezone": "Asia/Kolkata",
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=15)
        if r.status_code == 403:
            return [{"error": "Open-Meteo blocked in this environment. Will work on your local server.", "date": "today"}]
        r.raise_for_status()
        data = r.json()["daily"]
        return [
            {
                "date":           data["time"][i],
                "type":           "forecast",
                "temp_max_c":     round(data["temperature_2m_max"][i], 1) if data["temperature_2m_max"][i] is not None else None,
                "temp_min_c":     round(data["temperature_2m_min"][i], 1) if data["temperature_2m_min"][i] is not None else None,
                "precipitation_mm": round(data["precipitation_sum"][i], 1) if data["precipitation_sum"][i] is not None else None,
                "rain_probability_pct": data["precipitation_probability_max"][i],
                "weather_code":   data["weathercode"][i],
                "weather_label":  wmo_label(data["weathercode"][i]) if data["weathercode"][i] is not None else "Unknown",
                "demand_impact":  weather_impact(
                    data["weathercode"][i] or 0,
                    data["temperature_2m_max"][i]
                ),
            }
            for i in range(len(data["time"]))
        ]
    except Exception as e:
        return [{"error": str(e)}]


def get_full_weather_timeline(lat: float, lon: float,
                               historical_start: str = "2023-04-01") -> dict:
    """
    Returns:
      - historical: Apr 2023 → yesterday
      - forecast:   today → today+7
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today     = date.today().isoformat()

    hist = get_historical_weather(lat, lon, historical_start, yesterday)
    fcst = get_forecast_weather(lat, lon, days=8)

    return {
        "historical_start": historical_start,
        "historical_end":   yesterday,
        "forecast_start":   today,
        "forecast_end":     (date.today() + timedelta(days=7)).isoformat(),
        "historical":       hist,
        "forecast":         fcst,
    }
