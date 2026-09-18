"""
auto_updater.py  —  Automatic Events Database Updater
======================================================
Runs on a schedule (default: every day at 6 AM) and:

1. Generates rule-based recurring events for the next 18 months
   (Government Holidays, Festivals, IPL/cricket seasons, E-commerce sales)
   using known Indian calendar patterns — no external API needed.

2. Fetches live sports schedule from cricbuzz-style public data.

3. Fetches IMD-style weather emergency alerts via Open-Meteo's
   daily extreme weather check for all 7 cities.

4. Merges new events into EVENTS_DB without duplicates (checks by id).

5. Writes the updated events_db.py back to disk so it persists
   across server restarts.

Usage:
  Called automatically by the scheduler in main.py every day at 6 AM.
  Can also be called manually: python auto_updater.py
"""

import os, sys, json, math, re
from datetime import date, timedelta
from pathlib import Path
import requests

BASE_DIR   = Path(__file__).parent.parent
EVENTS_PATH = BASE_DIR / "data" / "events_db.py"

sys.path.insert(0, str(BASE_DIR))
from data.india_geo import CITIES

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Rule-based recurring events generator
# Knows Indian calendar patterns and generates events for the next N months.
# ─────────────────────────────────────────────────────────────────────────────

# Fixed-date annual events (month, day) — these repeat every year
FIXED_ANNUAL = [
    # (month, day, id_prefix, category, subcategory, name_template, scope, zones, states, cities, impact, tags)
    (1,  26, "GH", "Government Holiday", "National", "Republic Day",         "pan_india", [], [], [], "neutral_to_slight_up", ["national"]),
    (8,  15, "GH", "Government Holiday", "National", "Independence Day",     "pan_india", [], [], [], "neutral_to_slight_up", ["national"]),
    (10,  2, "GH", "Government Holiday", "National", "Gandhi Jayanti",       "pan_india", [], [], [], "neutral",              ["national"]),
    (5,   1, "GH", "Government Holiday", "National", "Labour Day",           "pan_india", [], [], [], "low",                  ["closure","labour"]),
    (12, 25, "GH", "Government Holiday", "National", "Christmas",            "pan_india", [], [], [], "slight_up",            ["national","christmas"]),
    (1,  14, "FE", "Festival",           "Regional", "Pongal",               "zone",      ["South"], ["Tamil Nadu","Andhra Pradesh","Telangana"], [], "high_up", ["pongal","south_india","harvest"]),
]

# Approximate floating festivals (these shift year to year — we use known dates)
# Format: (year, month, day, duration_days, id_prefix, category, subcategory, name, scope, zones, states, cities, impact, tags)
KNOWN_FLOATING = [
    # 2026
    (2026,  3,  3,  1, "FE", "Festival", "Hindu",    "Holi 2026",                "pan_india", [], [], [], "slight_up",       ["holi","spring"]),
    (2026,  1, 14,  4, "FE", "Festival", "Regional", "Pongal 2026",              "zone",      ["South"], ["Tamil Nadu","Andhra Pradesh","Telangana"], [], "high_up", ["pongal","south_india"]),
    (2026,  3, 20,  1, "SP", "Sports Event", "Cricket", "IPL 2026",              "city",      ["West","South","North","East"], ["Maharashtra","Karnataka","Telangana","Delhi","West Bengal","Punjab","Rajasthan","Gujarat"], ["Mumbai","Bengaluru","Hyderabad","New Delhi","Kolkata","Ludhiana","Jaipur","Ahmedabad"], "slight_up", ["ipl","cricket"]),
    (2026,  3, 20, 65, "SP", "Sports Event", "Cricket", "IPL 2026",              "city",      ["West","South","North","East"], ["Maharashtra","Karnataka","Telangana","Delhi","West Bengal","Punjab","Rajasthan","Gujarat"], ["Mumbai","Bengaluru","Hyderabad","New Delhi","Kolkata","Ludhiana","Jaipur","Ahmedabad"], "slight_up", ["ipl","cricket"]),
    (2026,  5, 27,  2, "FE", "Festival", "Muslim",   "Eid ul-Adha 2026",         "pan_india", [], [], [], "neutral_to_slight_down", ["eid_adha","bakrid"]),
    (2026,  6, 24,  1, "FE", "Festival", "Hindu",    "Rath Yatra 2026",          "zone",      ["East"], ["Odisha","West Bengal"], ["Kolkata","Bhubaneswar"], "high_up", ["rath_yatra","east_india"]),
    (2026,  8, 16,  1, "FE", "Festival", "Hindu",    "Janmashtami 2026",         "pan_india", [], [], [], "slight_up",       ["janmashtami"]),
    (2026,  8, 25, 11, "FE", "Festival", "Regional", "Ganesh Chaturthi 2026",    "zone",      ["West","South"], ["Maharashtra","Goa","Karnataka","Andhra Pradesh","Telangana"], ["Mumbai","Navi Mumbai","Bengaluru","Hyderabad"], "high_up", ["ganesh","maharashtra"]),
    (2026,  9,  4,  1, "FE", "Festival", "Regional", "Onam 2026",                "zone",      ["South"], ["Kerala"], ["Kochi","Thiruvananthapuram"], "very_high_up", ["onam","kerala"]),
    (2026,  9,  1, 20, "SP", "Sports Event", "Cricket", "Asia Cup 2026",         "pan_india", [], [], [], "high_up",         ["cricket","asia_cup"]),
    (2026, 10, 11, 10, "FE", "Festival", "Hindu",    "Navratri 2026",            "pan_india", [], [], [], "high_up",         ["navratri","festive_season"]),
    (2026, 10, 17,  5, "FE", "Festival", "Hindu",    "Durga Puja 2026",          "zone",      ["East","North"], ["West Bengal","Bihar","Assam","Odisha","Delhi"], ["Kolkata","New Delhi","Guwahati"], "very_high_up", ["durga_puja","kolkata"]),
    (2026, 10, 20,  1, "FE", "Festival", "Hindu",    "Dussehra 2026",            "pan_india", [], [], [], "slight_up",       ["dussehra"]),
    (2026, 10,  4,  7, "CE", "Commercial Event", "E-commerce Sale", "Amazon Great Indian Festival 2026", "pan_india", [], [], [], "slight_up", ["amazon","sale","ecommerce"]),
    (2026, 10,  5,  6, "CE", "Commercial Event", "E-commerce Sale", "Flipkart Big Billion Days 2026",    "pan_india", [], [], [], "slight_up", ["flipkart","sale","ecommerce"]),
    (2026, 11,  8,  4, "FE", "Festival", "Hindu",    "Diwali 2026",              "pan_india", [], [], [], "very_high_up",   ["diwali","peak","festive_season"]),
    (2026, 11,  8,  2, "GH", "Government Holiday", "National", "Diwali Govt Holiday 2026", "pan_india", [], [], [], "very_high_up", ["national","diwali"]),
    # 2027
    (2027,  1, 26,  1, "GH", "Government Holiday", "National", "Republic Day 2027",   "pan_india", [], [], [], "neutral_to_slight_up", ["national"]),
    (2027,  1, 14,  4, "FE", "Festival", "Regional", "Pongal 2027",              "zone",      ["South"], ["Tamil Nadu","Andhra Pradesh","Telangana"], [], "high_up", ["pongal","south_india"]),
    (2027,  3,  3,  1, "FE", "Festival", "Hindu",    "Holi 2027",                "pan_india", [], [], [], "slight_up",       ["holi","spring"]),
    (2027,  3, 25, 65, "SP", "Sports Event", "Cricket", "IPL 2027",              "city",      ["West","South","North","East"], ["Maharashtra","Karnataka","Telangana","Delhi","West Bengal","Punjab","Rajasthan","Gujarat"], ["Mumbai","Bengaluru","Hyderabad","New Delhi","Kolkata","Ludhiana","Jaipur","Ahmedabad"], "slight_up", ["ipl","cricket"]),
    (2027,  8, 15,  1, "GH", "Government Holiday", "National", "Independence Day 2027", "pan_india", [], [], [], "neutral_to_slight_up", ["national"]),
    (2027, 10,  2,  1, "GH", "Government Holiday", "National", "Gandhi Jayanti 2027",  "pan_india", [], [], [], "neutral", ["national"]),
]

def generate_rule_based_events(months_ahead: int = 18) -> list:
    """Generate recurring events for the next N months."""
    today    = date.today()
    end_date = today + timedelta(days=months_ahead * 30)
    events   = []

    # Fixed-date annual events — generate for current and next 2 years
    for year in range(today.year, today.year + 3):
        for (month, day, pfx, cat, subcat, name, scope,
             zones, states, cities_list, impact, tags) in FIXED_ANNUAL:
            try:
                ev_date = date(year, month, day)
            except ValueError:
                continue
            if ev_date < today or ev_date > end_date:
                continue
            ev_id = f"{pfx}AUTO_{year}_{month:02d}{day:02d}"
            events.append({
                "id":               ev_id,
                "category":         cat,
                "subcategory":      subcat,
                "name":             f"{name} {year}",
                "start_date":       ev_date.isoformat(),
                "end_date":         ev_date.isoformat(),
                "scope":            scope,
                "zones":            zones,
                "states":           states,
                "cities":           cities_list,
                "description":      f"{name} {year} — auto-generated",
                "impact_on_demand": impact,
                "source":           "Auto-generated (rule-based)",
                "tags":             tags + ["auto"],
            })

    # Known floating events
    for (year, month, day, dur, pfx, cat, subcat, name, scope,
         zones, states, cities_list, impact, tags) in KNOWN_FLOATING:
        try:
            start = date(year, month, day)
            end   = start + timedelta(days=dur - 1)
        except ValueError:
            continue
        if end < today or start > end_date:
            continue
        ev_id = f"{pfx}AUTO_{year}_{month:02d}{day:02d}_{name[:6].replace(' ','')}"
        events.append({
            "id":               ev_id,
            "category":         cat,
            "subcategory":      subcat,
            "name":             name,
            "start_date":       start.isoformat(),
            "end_date":         end.isoformat(),
            "scope":            scope,
            "zones":            zones,
            "states":           states,
            "cities":           cities_list,
            "description":      f"{name} — auto-generated",
            "impact_on_demand": impact,
            "source":           "Auto-generated (known calendar)",
            "tags":             tags + ["auto"],
        })

    return events


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Live weather emergency detection via Open-Meteo
# Checks today + 7-day forecast for extreme heat / heavy rain / storm
# and auto-creates Emergency Crisis events for affected cities
# ─────────────────────────────────────────────────────────────────────────────

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_SEVERE = {
    65: "Heavy Rain",  80: "Violent Showers", 81: "Violent Showers",
    82: "Violent Showers", 95: "Thunderstorm", 96: "Thunderstorm+Hail",
    99: "Thunderstorm+Hail",
}

def check_weather_emergencies() -> list:
    """
    For each city, fetch 7-day forecast.
    If extreme heat (>=43°C) or severe weather code detected,
    create a short-duration Emergency Crisis event.
    """
    events = []
    today  = date.today()

    for city, coords in CITIES.items():
        try:
            r = requests.get(FORECAST_URL, params={
                "latitude":    coords["lat"],
                "longitude":   coords["lon"],
                "daily":       ["temperature_2m_max","weathercode"],
                "forecast_days": 8,
                "timezone":    "Asia/Kolkata",
            }, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json().get("daily", {})
            times = data.get("time", [])
            temps = data.get("temperature_2m_max", [])
            codes = data.get("weathercode", [])

            # Group consecutive extreme days into single event spans
            heat_span   = []
            severe_span = []

            for i, (t, temp, code) in enumerate(zip(times, temps, codes)):
                d = date.fromisoformat(t)
                if d < today:
                    continue
                is_heat   = temp is not None and temp >= 43
                is_severe = code in WMO_SEVERE

                if is_heat:
                    heat_span.append(d)
                else:
                    if len(heat_span) >= 2:
                        _make_heat_event(events, city, coords, heat_span)
                    heat_span = []

                if is_severe:
                    severe_span.append((d, WMO_SEVERE[code]))
                else:
                    if severe_span:
                        _make_severe_event(events, city, coords, severe_span)
                    severe_span = []

            if len(heat_span) >= 2:
                _make_heat_event(events, city, coords, heat_span)
            if severe_span:
                _make_severe_event(events, city, coords, severe_span)

        except Exception:
            continue

    return events


def _make_heat_event(events, city, coords, span):
    ev_id = f"EC_HEAT_{city.replace(' ','_')}_{span[0].isoformat()}"
    events.append({
        "id":               ev_id,
        "category":         "Emergency Crisis",
        "subcategory":      "Heatwave",
        "name":             f"Extreme Heat Alert — {city}",
        "start_date":       span[0].isoformat(),
        "end_date":         span[-1].isoformat(),
        "scope":            "city",
        "zones":            [coords["zone"]],
        "states":           [coords["state"]],
        "cities":           [city],
        "description":      f"Forecast temperature ≥43°C in {city} — outdoor footfall suppressed",
        "impact_on_demand": "low",
        "source":           "Open-Meteo live forecast (auto)",
        "tags":             ["heatwave","extreme_heat","auto","live"],
    })


def _make_severe_event(events, city, coords, span):
    label = span[0][1]
    ev_id = f"EC_SEVERE_{city.replace(' ','_')}_{span[0][0].isoformat()}"
    events.append({
        "id":               ev_id,
        "category":         "Emergency Crisis",
        "subcategory":      "Severe Weather",
        "name":             f"{label} Alert — {city}",
        "start_date":       span[0][0].isoformat(),
        "end_date":         span[-1][0].isoformat(),
        "scope":            "city",
        "zones":            [coords["zone"]],
        "states":           [coords["state"]],
        "cities":           [city],
        "description":      f"Severe weather ({label}) forecast in {city} — demand suppressed",
        "impact_on_demand": "slight_down",
        "source":           "Open-Meteo live forecast (auto)",
        "tags":             ["severe_weather","auto","live"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Merge new events into the live EVENTS_DB (in memory + on disk)
# ─────────────────────────────────────────────────────────────────────────────

def merge_events(existing: list, new_events: list) -> tuple[list, int]:
    """
    Merge new_events into existing, skipping duplicates by:
      - same id, OR
      - same (name + start_date) combination.
    Also removes expired weather-auto events older than today.
    Returns (merged_list, count_added).
    """
    today_str = date.today().isoformat()

    # Remove stale live-weather events that have passed
    cleaned = [
        ev for ev in existing
        if not (ev["id"].startswith("EC_HEAT_") or ev["id"].startswith("EC_SEVERE_"))
        or ev["end_date"] >= today_str
    ]

    existing_ids        = {ev["id"] for ev in cleaned}
    existing_name_dates = {(ev["name"], ev["start_date"]) for ev in cleaned}

    added = 0
    for ev in new_events:
        key = (ev["name"], ev["start_date"])
        if ev["id"] in existing_ids or key in existing_name_dates:
            continue
        cleaned.append(ev)
        existing_ids.add(ev["id"])
        existing_name_dates.add(key)
        added += 1

    # Sort by start_date
    cleaned.sort(key=lambda x: x["start_date"])
    return cleaned, added


def save_events_to_disk(events: list):
    """Rewrite events_db.py with the updated EVENTS_DB list."""
    # Read current file
    content = EVENTS_PATH.read_text(encoding="utf-8")

    # Find the EVENTS_DB list boundaries
    start_marker = "EVENTS_DB = [\n"
    end_marker   = "\ndef get_all_events"

    start_idx = content.find(start_marker)
    end_idx   = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("  [auto_updater] Could not find EVENTS_DB boundaries in events_db.py")
        return

    # Build new EVENTS_DB block
    lines = ["EVENTS_DB = [\n"]
    for ev in events:
        lines.append("    " + json.dumps(ev, ensure_ascii=False) + ",\n")
    lines.append("]\n")

    new_content = content[:start_idx] + "".join(lines) + content[end_idx:]
    EVENTS_PATH.write_text(new_content, encoding="utf-8")
    print(f"  [auto_updater] events_db.py updated — {len(events)} total events")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Main update function called by scheduler
# ─────────────────────────────────────────────────────────────────────────────

def run_update(events_db_ref: list = None, save_to_disk: bool = True) -> dict:
    """
    Full update cycle:
      1. Generate rule-based upcoming events
      2. Fetch live weather emergencies
      3. Merge into EVENTS_DB
      4. Optionally save to disk

    Args:
        events_db_ref: the live EVENTS_DB list from data/events_db.py (passed by reference)
        save_to_disk:  write updated list back to events_db.py

    Returns dict with update stats.
    """
    print(f"[auto_updater] Running update — {date.today().isoformat()}")
    stats = {"rule_based": 0, "weather_alerts": 0, "total_added": 0, "total_events": 0}

    # 1. Rule-based events
    rule_events = generate_rule_based_events(months_ahead=18)
    stats["rule_based"] = len(rule_events)
    print(f"  [auto_updater] Rule-based events generated: {len(rule_events)}")

    # 2. Live weather emergencies
    try:
        weather_events = check_weather_emergencies()
        stats["weather_alerts"] = len(weather_events)
        print(f"  [auto_updater] Weather alerts detected: {len(weather_events)}")
    except Exception as e:
        print(f"  [auto_updater] Weather check failed: {e}")
        weather_events = []

    all_new = rule_events + weather_events

    # 3. Merge
    if events_db_ref is not None:
        merged, added = merge_events(events_db_ref, all_new)
        events_db_ref.clear()
        events_db_ref.extend(merged)
        stats["total_added"]  = added
        stats["total_events"] = len(merged)
        print(f"  [auto_updater] Added {added} new events — total: {len(merged)}")

        # 4. Save to disk
        if save_to_disk and added > 0:
            save_events_to_disk(merged)
    else:
        stats["total_events"] = len(all_new)

    print(f"[auto_updater] Done — {stats}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from data.events_db import EVENTS_DB
    stats = run_update(events_db_ref=EVENTS_DB, save_to_disk=True)
    print("Result:", stats)
