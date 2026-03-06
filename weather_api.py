"""
utils/weather_api.py
OpenWeather API integration for Gujarat cities.
Falls back to realistic demo weather if no key / API fails.
"""

import requests
from datetime import datetime
from typing import Optional

# ──────────────────────────────────────────────
# GUJARAT CITIES SUPPORTED
# ──────────────────────────────────────────────
GUJARAT_CITIES = [
    "Ahmedabad", "Vadodara", "Anand", "Surat", "Rajkot",
    "Gandhinagar", "Nadiad", "Bharuch", "Mehsana", "Bhavnagar",
]

# Your OpenWeather API key (hardcoded as fallback)
DEFAULT_API_KEY = "15676aee0807a886792185eae38f1b42"

# ──────────────────────────────────────────────
# DEMO FALLBACK (time-of-day based, Gujarat-realistic)
# ──────────────────────────────────────────────
def get_demo_weather(city: str = "Ahmedabad") -> dict:
    """
    Returns realistic Gujarat weather based on current hour.
    Used when API key is missing or API call fails.
    """
    hour = datetime.now().hour
    month = datetime.now().month

    # Season-aware base
    if month in [4, 5, 6]:          # Summer
        if 6 <= hour < 10:   temp, hum = 32, 55
        elif 10 <= hour < 14: temp, hum = 40, 35
        elif 14 <= hour < 18: temp, hum = 42, 30
        elif 18 <= hour < 22: temp, hum = 36, 45
        else:                 temp, hum = 30, 60
    elif month in [7, 8, 9]:         # Monsoon
        if 6 <= hour < 10:   temp, hum = 29, 80
        elif 10 <= hour < 18: temp, hum = 33, 75
        elif 18 <= hour < 22: temp, hum = 30, 85
        else:                 temp, hum = 27, 88
    elif month in [11, 12, 1]:       # Winter
        if 4 <= hour < 8:    temp, hum = 12, 50
        elif 8 <= hour < 14: temp, hum = 20, 40
        elif 14 <= hour < 18: temp, hum = 22, 35
        elif 18 <= hour < 22: temp, hum = 16, 45
        else:                 temp, hum = 13, 55
    else:                             # Spring (Feb, Mar, Oct)
        if 6 <= hour < 10:   temp, hum = 25, 55
        elif 10 <= hour < 18: temp, hum = 33, 45
        elif 18 <= hour < 22: temp, hum = 28, 50
        else:                 temp, hum = 22, 60

    return {
        "outdoor_temp":      float(temp),
        "humidity":          float(hum),
        "feels_like":        float(temp + 2),
        "weather_condition": "demo",
        "description":       "Demo Weather (Gujarat Estimate)",
        "city":              city,
        "timestamp":         datetime.now(),
        "is_demo":           True,
    }


# ──────────────────────────────────────────────
# LIVE CURRENT WEATHER
# ──────────────────────────────────────────────
def get_current_weather(api_key: Optional[str] = None, city: str = "Ahmedabad") -> dict:
    """
    Fetch current weather from OpenWeather API.
    Returns demo values if API call fails.
    """
    key = api_key or DEFAULT_API_KEY
    if not key or key.strip() == "":
        print("[DEMO] No API key — using demo weather.")
        return get_demo_weather(city)

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q":     f"{city},IN",
            "appid": key.strip(),
            "units": "metric",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        return {
            "outdoor_temp":      round(data["main"]["temp"], 1),
            "humidity":          round(data["main"]["humidity"], 1),
            "feels_like":        round(data["main"]["feels_like"], 1),
            "weather_condition": data["weather"][0]["main"].lower(),
            "description":       data["weather"][0]["description"].title(),
            "city":              data["name"],
            "timestamp":         datetime.now(),
            "is_demo":           False,
        }
    except requests.exceptions.HTTPError as e:
        print(f"[WARN] OpenWeather HTTP error: {e}. Falling back to demo.")
        return get_demo_weather(city)
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Network error: {e}. Falling back to demo.")
        return get_demo_weather(city)
    except Exception as e:
        print(f"[WARN] Unexpected error: {e}. Falling back to demo.")
        return get_demo_weather(city)


# ──────────────────────────────────────────────
# 3-HOUR FORECAST → INTERPOLATED TO 15-MIN SLOTS
# ──────────────────────────────────────────────
def get_weather_forecast_3hrs(api_key: Optional[str] = None,
                               city: str = "Ahmedabad") -> list:
    """
    Returns weather forecast interpolated to 15-minute slots
    for the next 3 hours (12 slots).
    Falls back to demo if API fails.
    """
    key = api_key or DEFAULT_API_KEY
    slots = []

    try:
        if not key or key.strip() == "":
            raise ValueError("No API key")

        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": f"{city},IN", "appid": key.strip(), "units": "metric", "cnt": 2}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("list", [])
        if len(items) < 2:
            raise ValueError("Insufficient forecast data")

        t0 = items[0]["main"]["temp"]
        h0 = items[0]["main"]["humidity"]
        t1 = items[1]["main"]["temp"]
        h1 = items[1]["main"]["humidity"]

        now = datetime.now()
        # Interpolate 12 slots (3 hours) between t0 and t1
        for i in range(12):
            frac = i / 11.0
            slot_time = now.replace(second=0, microsecond=0)
            slot_time = slot_time.replace(minute=(slot_time.minute // 15) * 15)
            from datetime import timedelta
            slot_time = slot_time + timedelta(minutes=i * 15)

            slots.append({
                "time":     slot_time.strftime("%H:%M"),
                "temp":     round(t0 + (t1 - t0) * frac, 1),
                "humidity": round(h0 + (h1 - h0) * frac, 1),
                "is_demo":  False,
            })

    except Exception as e:
        print(f"[WARN] Forecast fallback: {e}")
        # Build demo forecast around current hour
        now = datetime.now()
        base = get_demo_weather(city)
        from datetime import timedelta
        for i in range(12):
            st = now + timedelta(minutes=i * 15)
            # Slight temperature drift: -0.2°C per slot after 2 PM, else +0.1
            drift = -0.2 if now.hour >= 14 else 0.1
            slots.append({
                "time":     st.strftime("%H:%M"),
                "temp":     round(base["outdoor_temp"] + drift * i, 1),
                "humidity": round(base["humidity"] + 0.3 * i, 1),
                "is_demo":  True,
            })

    return slots


# ──────────────────────────────────────────────
# 7-DAY DAILY FORECAST
# ──────────────────────────────────────────────
def get_7day_forecast(api_key: Optional[str] = None, city: str = "Ahmedabad") -> list:
    """
    Returns 7-day daily forecast (max available from free tier is 5 days/40 slots).
    Each entry: {day, date, min_temp, max_temp, avg_humidity, condition, icon, is_demo}
    Falls back to realistic Gujarat demo data on failure.
    """
    key = api_key or DEFAULT_API_KEY
    WEATHER_ICONS = {
        "clear sky": "☀️", "few clouds": "🌤️", "scattered clouds": "⛅",
        "broken clouds": "☁️", "overcast clouds": "☁️",
        "light rain": "🌦️", "moderate rain": "🌧️", "heavy intensity rain": "⛈️",
        "thunderstorm": "⛈️", "snow": "❄️", "mist": "🌫️", "haze": "🌫️",
        "fog": "🌫️", "drizzle": "🌦️",
    }
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    try:
        if not key or key.strip() == "":
            raise ValueError("No API key")

        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": f"{city},IN", "appid": key.strip(), "units": "metric", "cnt": 40}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        from collections import defaultdict
        daily = defaultdict(lambda: {"temps": [], "humidities": [], "conditions": []})
        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            day_key = dt.strftime("%Y-%m-%d")
            daily[day_key]["temps"].append(item["main"]["temp"])
            daily[day_key]["humidities"].append(item["main"]["humidity"])
            daily[day_key]["conditions"].append(item["weather"][0]["description"])

        result = []
        for day_str, vals in sorted(daily.items())[:7]:
            dt = datetime.strptime(day_str, "%Y-%m-%d")
            cond = max(set(vals["conditions"]), key=vals["conditions"].count)
            icon = WEATHER_ICONS.get(cond, "🌡️")
            result.append({
                "day":          DAYS[dt.weekday()],
                "date":         dt.strftime("%b %d"),
                "min_temp":     round(min(vals["temps"]), 1),
                "max_temp":     round(max(vals["temps"]), 1),
                "avg_humidity": round(sum(vals["humidities"]) / len(vals["humidities"]), 0),
                "condition":    cond.title(),
                "icon":         icon,
                "is_demo":      False,
            })
        return result

    except Exception as e:
        print(f"[WARN] 7-day forecast fallback: {e}")
        # Demo fallback — Gujarat realistic March weather
        from datetime import timedelta
        base = get_demo_weather(city)
        result = []
        DEMO_CONDITIONS = [
            ("Clear Sky", "☀️", 2), ("Few Clouds", "🌤️", 1),
            ("Scattered Clouds", "⛅", 1), ("Clear Sky", "☀️", 1),
            ("Haze", "🌫️", 1), ("Clear Sky", "☀️", 1), ("Few Clouds", "🌤️", 1),
        ]
        today = datetime.now()
        for i in range(7):
            dt = today + timedelta(days=i)
            cond_name, icon, _ = DEMO_CONDITIONS[i % len(DEMO_CONDITIONS)]
            variation = (i % 3) - 1  # slight daily variation
            result.append({
                "day":          DAYS[dt.weekday()],
                "date":         dt.strftime("%b %d"),
                "min_temp":     round(base["outdoor_temp"] - 4 + variation, 1),
                "max_temp":     round(base["outdoor_temp"] + 3 + variation, 1),
                "avg_humidity": round(base["humidity"] + variation * 2, 0),
                "condition":    cond_name,
                "icon":         icon,
                "is_demo":      True,
            })
        return result



# ──────────────────────────────────────────────
# APPLIANCE WEATHER IMPACT DESCRIPTION
# ──────────────────────────────────────────────
def get_appliance_weather_impact(weather: dict, appliance: str) -> str:
    """
    Returns a human-readable weather impact message for the appliance.
    """
    temp = weather.get("outdoor_temp", 30)
    hum  = weather.get("humidity", 60)

    impacts = {
        "AC": (
            "🔴 VERY HIGH load — extreme heat day"    if temp > 40 else
            "🟠 HIGH load — hot day, AC working hard" if temp > 35 else
            "🟡 MODERATE load — pleasant conditions"  if temp > 28 else
            "🟢 LOW load — cool weather, minimal AC needed"
        ),
        "Geyser": (
            "🔴 HIGH demand — cold morning, water heating long" if temp < 15 else
            "🟡 MODERATE demand"                                if temp < 22 else
            "🟢 LOW demand — warm weather, quick heating"
        ),
        "Refrigerator": (
            "🔴 Compressor under HIGH stress — keep door closed" if temp > 38 else
            "🟡 Moderate compressor load"                         if temp > 30 else
            "🟢 Normal operation"
        ),
        "Ceiling Fan": (
            "🔴 Running at full speed — very hot"   if temp > 37 else
            "🟡 Running at medium-high speed"        if temp > 30 else
            "🟢 Light usage — comfortable weather"
        ),
        "Washing Machine": "🟢 Weather has minimal impact on washing machine",
        "Microwave":       "🟢 Weather has minimal impact on microwave",
        "LED TV":          "🟢 Weather has minimal impact on TV",
        "Desktop PC": (
            "🟡 Ensure ventilation — high ambient temp" if temp > 35 else "🟢 Normal operation"
        ),
        "Electric Iron":  "🟢 Weather has minimal impact on iron",
        "LED Bulb":       "🟢 Weather has minimal impact on LED bulbs",
        "Laptop": (
            "🟡 Laptop may throttle — high ambient temp" if temp > 35 else "🟢 Normal operation"
        ),
        "Wi-Fi Router":   "🟢 Weather has minimal impact on router",
    }
    return impacts.get(appliance, "🟢 No significant weather impact")


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing Weather API ===")
    for city in ["Ahmedabad", "Anand", "Vadodara"]:
        w = get_current_weather(DEFAULT_API_KEY, city)
        demo_tag = "[DEMO]" if w["is_demo"] else "[LIVE]"
        print(f"{demo_tag} {city}: {w['outdoor_temp']}°C, {w['humidity']}% humidity, {w['description']}")

    print("\n=== Forecast (next 3 hrs, 15-min slots) ===")
    forecast = get_weather_forecast_3hrs(DEFAULT_API_KEY, "Ahmedabad")
    for s in forecast[:6]:
        print(f"  {s['time']} → {s['temp']}°C, {s['humidity']}% hum")
