"""
Built-in skills: time/date, weather (open-meteo, no API key), and countdown
timers. Each registers itself via @skill and returns a JSON-serializable dict
that the model turns into a spoken reply.
"""
import json
import time
import threading
import datetime
import urllib.parse
import urllib.request

import config
from modules.skills import skill


# --------------------------------------------------------------------------
# Time / date
# --------------------------------------------------------------------------
@skill("get_time", "Get the current local date and time.")
def _get_time(args):
    now = datetime.datetime.now()
    return {
        "time": now.strftime("%-I:%M %p"),
        "date": now.strftime("%A, %B %-d, %Y"),
    }


# --------------------------------------------------------------------------
# Weather (open-meteo: free, no key)
# --------------------------------------------------------------------------
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
    81: "showers", 82: "violent showers", 85: "snow showers",
    86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

_geo_cache: dict = {}


def _get_json(url: str, timeout: int = 8) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "voice-assistant"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _geocode(place: str):
    """Resolve a place name to (lat, lon, label) via open-meteo geocoding. The
    geocoder wants a bare name, so 'Dallas, Texas' falls back to 'Dallas'."""
    key = place.strip().lower()
    if key in _geo_cache:
        return _geo_cache[key]
    # Try the full string, then progressively shorter leading parts.
    candidates = [place.strip()]
    if "," in place:
        candidates.append(place.split(",", 1)[0].strip())
    for cand in candidates:
        if not cand:
            continue
        url = ("https://geocoding-api.open-meteo.com/v1/search?name="
               + urllib.parse.quote(cand) + "&count=1")
        results = _get_json(url).get("results") or []
        if results:
            r = results[0]
            label = ", ".join(x for x in (r.get("name"), r.get("admin1"),
                                          r.get("country_code")) if x)
            out = (r["latitude"], r["longitude"], label)
            _geo_cache[key] = out
            return out
    return None


def _resolve_location(arg_location):
    """Pick a location: explicit arg (geocoded) or the configured home location."""
    if arg_location:
        geo = _geocode(str(arg_location))
        if geo:
            return geo
    if config.WEATHER_LAT is not None and config.WEATHER_LON is not None:
        return (config.WEATHER_LAT, config.WEATHER_LON,
                config.WEATHER_PLACE or "your area")
    if config.WEATHER_PLACE:
        geo = _geocode(config.WEATHER_PLACE)
        if geo:
            return geo
    return None


@skill("get_weather",
       "Get current weather for a place.",
       {"location": "city name (optional; defaults to the configured home location)"})
def _get_weather(args):
    loc = _resolve_location(args.get("location"))
    if not loc:
        return {"error": "no location set — configure a home location in the control panel"}
    lat, lon, label = loc
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
           f"&temperature_unit={config.WEATHER_UNITS}&wind_speed_unit=mph")
    data = _get_json(url)
    cur = data.get("current", {})
    code = int(cur.get("weather_code", -1))
    unit = "F" if config.WEATHER_UNITS == "fahrenheit" else "C"
    return {
        "location": label,
        "conditions": _WMO.get(code, "unknown"),
        "temperature": f"{round(cur.get('temperature_2m'))}°{unit}",
        "feels_like": f"{round(cur.get('apparent_temperature'))}°{unit}",
        "wind": f"{round(cur.get('wind_speed_10m'))} mph",
    }


# --------------------------------------------------------------------------
# Timers
# --------------------------------------------------------------------------
_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
}


def _parse_duration(text: str) -> int:
    """Parse '10 minutes', '1 hr 30 min', '90s' → total seconds (0 if none)."""
    import re
    total = 0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", text.lower()):
        if unit in _UNIT_SECONDS:
            total += int(float(num) * _UNIT_SECONDS[unit])
    return total


def _chime():
    """A short two-tone beep, synthesized and played."""
    try:
        import numpy as np
        from modules import audio
        sr = 16000
        beep = []
        for freq in (880, 1320):
            t = np.linspace(0, 0.18, int(sr * 0.18), endpoint=False)
            tone = 0.4 * np.sin(2 * np.pi * freq * t)
            env = np.minimum(1, np.minimum(t * 40, (0.18 - t) * 40))   # fade in/out
            beep.append((tone * env).astype(np.float32))
        pcm = (np.concatenate(beep) * 32767).astype("<i2").tobytes()
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm)
        audio.play_audio(buf.getvalue())
    except Exception:
        pass


def _fire_timer(label):
    from modules import tts
    _chime()
    what = f"Your {label} timer is up." if label else "Your timer is up."
    tts.speak(what)


@skill("set_timer",
       "Start a countdown timer that chimes when it finishes.",
       {"duration": "e.g. '10 minutes', '90 seconds', '1 hour 30 min'",
        "label": "optional name for the timer"})
def _set_timer(args):
    seconds = _parse_duration(str(args.get("duration", "")))
    if seconds <= 0:
        return {"error": "couldn't understand the duration"}
    label = (args.get("label") or "").strip() or None
    timer = threading.Timer(seconds, _fire_timer, args=[label])
    timer.daemon = True
    timer.start()
    mins, secs = divmod(seconds, 60)
    pretty = (f"{mins} min" + (f" {secs} sec" if secs else "")) if mins else f"{secs} sec"
    return {"ok": True, "duration": pretty, "label": label}
