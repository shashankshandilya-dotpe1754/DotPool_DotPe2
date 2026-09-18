"""
India Events & Weather Centralized API
=======================================
FastAPI application exposing:
  /events/*   — historical & upcoming events (7 categories)
  /weather/*  — historical + live 7-day forecast
  /timeline/* — combined date-range view (events + weather merged)
  /calendar/* — day-level full context card

Run:
  cd /home/claude/events_api
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, timedelta
from typing import Optional, List
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.events_db  import (get_all_events, get_events_by_date,
                               get_events_by_range, EVENTS_DB)
from data.india_geo  import CITIES, ZONES, STATE_ZONE, CITY_LIST, STATE_LIST, ZONE_LIST
from utils.weather_service import (get_historical_weather, get_forecast_weather,
                                    get_full_weather_timeline)

# ─── Scheduler setup ─────────────────────────────────────────────────────────
from apscheduler.schedulers.background import BackgroundScheduler
from utils.auto_updater import run_update

def _scheduled_update():
    """Called by scheduler every day at 6 AM — updates EVENTS_DB in memory + disk."""
    run_update(events_db_ref=EVENTS_DB, save_to_disk=True)

scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.add_job(_scheduled_update, "cron", hour=6, minute=0,
                  id="daily_event_update", replace_existing=True)
scheduler.start()
print("[scheduler] Daily event update job started — runs every day at 6:00 AM IST")

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="India Events & Weather — Centralized API",
    description="""
Centralised reference API for demand forecasting context across Pan India.

**Event categories covered:**
- Government Holidays (national, state, city)
- Festivals (Hindu, Muslim, Christian, Regional)
- Public Events (elections, civic gatherings)
- Commercial Events (sales, promotions, trade fairs)
- Sports Events (IPL, World Cup, ISL)
- Government Orders (policy changes affecting demand)
- Emergency Crisis (floods, heatwaves, cyclones, AQI alerts)

**Weather:**
- Historical daily weather: Apr 2023 → yesterday (Open-Meteo archive, no API key)
- Live 7-day forecast: today → today+7 (Open-Meteo forecast)
- Fields: temp_max, temp_min, precipitation, weather_code, weather_label, demand_impact

**Geography filters available for all endpoints:**
- scope: pan_india | zone | state | city
- zones: North, South, East, West, Central, Northeast
- states: any Indian state
- cities: 35 major cities with lat/lon
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─── Health ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "api": "India Events & Weather Centralized API v1.0",
        "today": date.today().isoformat(),
        "endpoints": {
            "events_today":     "/events/today",
            "events_range":     "/events/range?start_date=2025-04-01&end_date=2025-05-31",
            "events_category":  "/events/category/{category}",
            "weather_city":     "/weather/city/{city_name}",
            "weather_city_hist":"/weather/city/{city_name}/historical",
            "weather_city_fcst":"/weather/city/{city_name}/forecast",
            "timeline":         "/timeline?start_date=2025-04-01&end_date=2025-04-30&city=New Delhi",
            "day_context":      "/calendar/{date}?city=Bengaluru",
            "geo_cities":       "/geo/cities",
            "geo_zones":        "/geo/zones",
            "forecast_day":     "/forecast/day?city=Bengaluru",
            "forecast_range":   "/forecast/range?city=Mumbai&start_date=2026-10-01&end_date=2026-10-31",
            "forecast_compare": "/forecast/compare?city=New Delhi&forecast_date=2026-10-20&restaurant_type=PBCL",
            "ml_model_info":    "/ml/model-info",
            "chat":             "POST /chat",
            "ingest_pending":   "/ingest/pending",
        },
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": date.today().isoformat()}


# ─── Geo endpoints ────────────────────────────────────────────────────────────
@app.get("/geo/cities", tags=["Geography"])
def list_cities():
    """All supported cities with state, zone, lat, lon."""
    return {"count": len(CITIES), "cities": CITIES}

@app.get("/geo/zones", tags=["Geography"])
def list_zones():
    """Zone → states mapping."""
    return {"zones": ZONES}

@app.get("/geo/states", tags=["Geography"])
def list_states():
    """All states with their zone."""
    return {"states": STATE_ZONE}


# ─── Events endpoints ─────────────────────────────────────────────────────────
CATEGORIES = [
    "Government Holiday",
    "Festival",
    "Public Event",
    "Commercial Event",
    "Sports Event",
    "Government Order",
    "Emergency Crisis",
]

@app.get("/events/today", tags=["Events"])
def events_today(
    city:  Optional[str] = Query(None, description="Filter by city name"),
    state: Optional[str] = Query(None, description="Filter by state name"),
    zone:  Optional[str] = Query(None, description="Filter by zone (North/South/East/West/Central/Northeast)"),
):
    """All events active today across all categories."""
    today = date.today().isoformat()
    evts  = get_events_by_date(today, city=city, state=state, zone=zone)
    return {
        "date": today,
        "filters": {"city": city, "state": state, "zone": zone},
        "count": len(evts),
        "events": evts,
    }

@app.get("/events/range", tags=["Events"])
def events_range(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date:   str = Query(..., description="YYYY-MM-DD"),
    category:   Optional[str] = Query(None, description="Filter by category"),
    city:       Optional[str] = Query(None),
    state:      Optional[str] = Query(None),
    zone:       Optional[str] = Query(None),
):
    """All events within a date range with optional geo and category filters."""
    evts = get_events_by_range(start_date, end_date,
                                category=category, city=city,
                                state=state, zone=zone)
    return {
        "start_date": start_date,
        "end_date":   end_date,
        "filters": {"category": category, "city": city, "state": state, "zone": zone},
        "count":  len(evts),
        "events": evts,
    }

@app.get("/events/upcoming", tags=["Events"])
def events_upcoming(
    days: int = Query(30, ge=1, le=365, description="How many days ahead"),
    city:  Optional[str] = None,
    state: Optional[str] = None,
    zone:  Optional[str] = None,
):
    """Upcoming events from today for the next N days."""
    today = date.today()
    end   = today + timedelta(days=days)
    evts  = get_events_by_range(today.isoformat(), end.isoformat(),
                                 city=city, state=state, zone=zone)
    return {
        "from": today.isoformat(),
        "to":   end.isoformat(),
        "count": len(evts),
        "events": evts,
    }

@app.get("/events/category/{category}", tags=["Events"])
def events_by_category(
    category:   str,
    start_date: Optional[str] = Query("2023-04-01"),
    end_date:   Optional[str] = Query(None),
    city:       Optional[str] = None,
    state:      Optional[str] = None,
    zone:       Optional[str] = None,
):
    """Events filtered by category. Category must be one of the 7 supported types."""
    if category not in CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Choose from: {CATEGORIES}"
        )
    if end_date is None:
        end_date = (date.today() + timedelta(days=365)).isoformat()
    evts = get_events_by_range(start_date, end_date,
                                category=category, city=city,
                                state=state, zone=zone)
    return {
        "category":   category,
        "start_date": start_date,
        "end_date":   end_date,
        "count":  len(evts),
        "events": evts,
    }

@app.get("/events/all", tags=["Events"])
def all_events(
    category: Optional[str] = None,
    scope:    Optional[str] = Query(None, description="pan_india / zone / state / city"),
):
    """Full events database dump with optional filters."""
    evts = get_all_events()
    if category:
        evts = [e for e in evts if e["category"] == category]
    if scope:
        evts = [e for e in evts if e["scope"] == scope]
    return {"count": len(evts), "events": evts}

@app.get("/events/{event_id}", tags=["Events"])
def get_event(event_id: str):
    """Get a single event by its ID."""
    for ev in get_all_events():
        if ev["id"] == event_id:
            return ev
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found")


# ─── Weather endpoints ────────────────────────────────────────────────────────
def _resolve_city(city_name: str):
    if city_name not in CITIES:
        close = [c for c in CITIES if city_name.lower() in c.lower()]
        raise HTTPException(
            status_code=404,
            detail=f"City '{city_name}' not found. "
                   + (f"Did you mean: {close}?" if close else f"Available: {CITY_LIST}")
        )
    return CITIES[city_name]

@app.get("/weather/city/{city_name}", tags=["Weather"])
def weather_city_full(city_name: str):
    """
    Full weather timeline for a city:
    - Historical: Apr 2023 → yesterday
    - Forecast: today → today+7
    """
    c    = _resolve_city(city_name)
    data = get_full_weather_timeline(c["lat"], c["lon"])
    data["city"]  = city_name
    data["state"] = c["state"]
    data["zone"]  = c["zone"]
    data["lat"]   = c["lat"]
    data["lon"]   = c["lon"]
    return data

@app.get("/weather/city/{city_name}/historical", tags=["Weather"])
def weather_city_historical(
    city_name:  str,
    start_date: str = Query("2023-04-01", description="YYYY-MM-DD"),
    end_date:   Optional[str] = Query(None, description="YYYY-MM-DD, defaults to yesterday"),
):
    """Historical daily weather for a city (temp, precipitation, weather label)."""
    c = _resolve_city(city_name)
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()
    data = get_historical_weather(c["lat"], c["lon"], start_date, end_date)
    return {
        "city": city_name, "state": c["state"], "zone": c["zone"],
        "lat": c["lat"], "lon": c["lon"],
        "start_date": start_date, "end_date": end_date,
        "count": len(data),
        "daily": data,
    }

@app.get("/weather/city/{city_name}/forecast", tags=["Weather"])
def weather_city_forecast(
    city_name: str,
    days: int = Query(8, ge=1, le=16, description="Forecast days (max 16)"),
):
    """Live weather forecast for a city — today + next N days."""
    c    = _resolve_city(city_name)
    data = get_forecast_weather(c["lat"], c["lon"], days=days)
    return {
        "city": city_name, "state": c["state"], "zone": c["zone"],
        "lat": c["lat"], "lon": c["lon"],
        "forecast_from": date.today().isoformat(),
        "forecast_to":   (date.today() + timedelta(days=days-1)).isoformat(),
        "count": len(data),
        "daily": data,
    }

@app.get("/weather/zone/{zone_name}/forecast", tags=["Weather"])
def weather_zone_forecast(
    zone_name: str,
    days: int = Query(8, ge=1, le=16),
):
    """Live forecast for all cities in a zone."""
    if zone_name not in ZONES:
        raise HTTPException(status_code=404, detail=f"Zone not found. Available: {ZONE_LIST}")
    zone_cities = {c: v for c, v in CITIES.items() if v["zone"] == zone_name}
    result = {}
    for city, coords in zone_cities.items():
        fcst = get_forecast_weather(coords["lat"], coords["lon"], days=days)
        result[city] = {"state": coords["state"], "forecast": fcst}
    return {"zone": zone_name, "cities_count": len(zone_cities), "data": result}


# ─── Timeline — merged events + weather ──────────────────────────────────────
@app.get("/timeline", tags=["Timeline"])
def timeline(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date:   str = Query(..., description="YYYY-MM-DD"),
    city:       Optional[str] = Query(None),
    state:      Optional[str] = Query(None),
    zone:       Optional[str] = Query(None),
    include_weather: bool = Query(True, description="Attach weather to each day (uses Open-Meteo)"),
):
    """
    Master date-range view: for each day in range, returns
    all active events + weather (historical or forecast).
    This is the primary endpoint for the demand forecast feature pipeline.
    """
    from datetime import datetime

    s = date.fromisoformat(start_date)
    e = date.fromisoformat(end_date)
    if (e - s).days > 366:
        raise HTTPException(status_code=400, detail="Max range is 366 days")

    evts_all = get_events_by_range(start_date, end_date,
                                    city=city, state=state, zone=zone)

    evts_by_date: dict = {}
    for ev in evts_all:
        ev_s = date.fromisoformat(ev["start_date"])
        ev_e = date.fromisoformat(ev["end_date"])
        cur  = max(s, ev_s)
        while cur <= min(e, ev_e):
            evts_by_date.setdefault(cur.isoformat(), []).append({
                "id": ev["id"], "category": ev["category"],
                "name": ev["name"], "scope": ev["scope"],
                "impact_on_demand": ev["impact_on_demand"],
                "tags": ev["tags"],
            })
            cur += timedelta(days=1)

    weather_by_date: dict = {}
    if include_weather and city and city in CITIES:
        coords    = CITIES[city]
        today_iso = date.today().isoformat()
        hist_end  = min((date.today() - timedelta(days=1)).isoformat(), end_date)
        fcst_start= max(today_iso, start_date)

        if start_date <= hist_end:
            for w in get_historical_weather(coords["lat"], coords["lon"],
                                             start_date, hist_end):
                weather_by_date[w["date"]] = w

        if fcst_start <= end_date:
            days_ahead = (date.fromisoformat(end_date) - date.today()).days + 1
            for w in get_forecast_weather(coords["lat"], coords["lon"],
                                           days=min(max(days_ahead, 1), 16)):
                weather_by_date[w["date"]] = w

    days_out = []
    cur = s
    while cur <= e:
        d_iso = cur.isoformat()
        day_events = evts_by_date.get(d_iso, [])
        categories = list({e["category"] for e in day_events})
        entry = {
            "date":        d_iso,
            "weekday":     cur.strftime("%A"),
            "is_weekend":  cur.weekday() >= 5,
            "event_count": len(day_events),
            "categories":  categories,
            "events":      day_events,
        }
        if include_weather and d_iso in weather_by_date:
            entry["weather"] = weather_by_date[d_iso]
        days_out.append(entry)
        cur += timedelta(days=1)

    return {
        "start_date": start_date,
        "end_date":   end_date,
        "filters": {"city": city, "state": state, "zone": zone},
        "total_days":  len(days_out),
        "total_events": len(evts_all),
        "timeline":    days_out,
    }


# ─── Day context card ─────────────────────────────────────────────────────────
@app.get("/calendar/{date_str}", tags=["Calendar"])
def day_context(
    date_str: str,
    city: Optional[str] = Query(None),
):
    """
    Full context for a single date: all events active that day + weather.
    Ideal for the demand forecast pipeline to enrich a single prediction date.
    """
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    evts = get_events_by_date(date_str, city=city)
    summary_flags = {
        "has_festival":           any(e["category"] == "Festival"           for e in evts),
        "has_gov_holiday":        any(e["category"] == "Government Holiday" for e in evts),
        "has_public_event":       any(e["category"] == "Public Event"       for e in evts),
        "has_commercial_event":   any(e["category"] == "Commercial Event"   for e in evts),
        "has_sports_event":       any(e["category"] == "Sports Event"       for e in evts),
        "has_government_order":   any(e["category"] == "Government Order"   for e in evts),
        "has_emergency_crisis":   any(e["category"] == "Emergency Crisis"   for e in evts),
        "is_weekend":             d.weekday() >= 5,
        "weekday":                d.strftime("%A"),
    }

    # Weather
    weather = None
    if city and city in CITIES:
        coords = CITIES[city]
        today  = date.today()
        if d < today:
            hist = get_historical_weather(coords["lat"], coords["lon"],
                                          date_str, date_str)
            weather = hist[0] if hist else None
        else:
            days = (d - today).days + 2
            fcst = get_forecast_weather(coords["lat"], coords["lon"],
                                         days=min(days, 16))
            weather = next((w for w in fcst if w.get("date") == date_str), None)

    # Overall demand signal
    impacts = [e["impact_on_demand"] for e in evts]
    if "very_low" in impacts or (weather and weather.get("demand_impact") == "very_low"):
        overall = "very_low"
    elif "low" in impacts or (weather and weather.get("demand_impact") == "low"):
        overall = "low"
    elif "very_high_up" in impacts:
        overall = "very_high_up"
    elif "high_up" in impacts:
        overall = "high_up"
    elif "slight_up" in impacts:
        overall = "slight_up"
    elif "slight_down" in impacts:
        overall = "slight_down"
    else:
        overall = "neutral"

    return {
        "date":           date_str,
        "city":           city,
        "weekday":        d.strftime("%A"),
        "summary_flags":  summary_flags,
        "overall_demand_signal": overall,
        "event_count":    len(evts),
        "events":         evts,
        "weather":        weather,
    }


# ─── Stats / summary ──────────────────────────────────────────────────────────
@app.get("/stats/events", tags=["Stats"])
def event_stats():
    """Count of events by category and scope in the database."""
    all_ev = get_all_events()
    by_cat   = {}
    by_scope = {}
    for ev in all_ev:
        by_cat[ev["category"]]  = by_cat.get(ev["category"], 0) + 1
        by_scope[ev["scope"]]   = by_scope.get(ev["scope"], 0) + 1
    return {
        "total": len(all_ev),
        "by_category": by_cat,
        "by_scope":    by_scope,
        "date_range":  {
            "earliest": min(ev["start_date"] for ev in all_ev),
            "latest":   max(ev["end_date"]   for ev in all_ev),
        },
    }


# ─── Auto-update endpoints ────────────────────────────────────────────────────
@app.post("/admin/update-events", tags=["Admin"])
def trigger_update_now():
    """
    Manually trigger an immediate events database update.
    Adds upcoming events for next 18 months + live weather alerts.
    Normally runs automatically every day at 6:00 AM IST.
    """
    stats = run_update(events_db_ref=EVENTS_DB, save_to_disk=True)
    return {
        "message": "Events database updated successfully",
        "stats":   stats,
        "total_events_now": len(EVENTS_DB),
    }


@app.get("/admin/update-status", tags=["Admin"])
def update_status():
    """Show the current scheduler status and next scheduled run time."""
    jobs = scheduler.get_jobs()
    job_info = []
    for job in jobs:
        job_info.append({
            "id":       job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger":  str(job.trigger),
        })
    return {
        "scheduler_running": scheduler.running,
        "total_events_in_db": len(EVENTS_DB),
        "jobs": job_info,
        "note": "Auto-update runs every day at 6:00 AM IST. POST /admin/update-events to run now.",
    }


# ─── ML demand-forecast endpoints ─────────────────────────────────────────────
from utils.restaurant_impact import (          # noqa: E402
    predict_impact, predict_date_range, compare_engines, engine_status, ALL_RT,
)


def _weather_lookup(city: str, start: str, end: str) -> dict:
    """Weather keyed by date for a city — archive for past days, forecast ahead."""
    geo = CITIES.get(city)
    if not geo:
        return {}
    out, today = {}, date.today()
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    try:
        if s < today:
            for r in get_historical_weather(geo["lat"], geo["lon"],
                                            s.isoformat(), min(e, today - timedelta(days=1)).isoformat()):
                if "error" not in r:
                    out[r["date"]] = r
        if e >= today:
            for r in get_forecast_weather(geo["lat"], geo["lon"], days=8):
                if "error" not in r:
                    out[r["date"]] = r
    except Exception:
        pass
    return out


@app.get("/ml/model-info", tags=["Forecast"])
def ml_model_info():
    """Which engine is live, when it was trained, and how it scored on holdout."""
    return engine_status()


@app.get("/forecast/day", tags=["Forecast"])
def forecast_day(
    city: str = Query(..., description="City name, e.g. Bengaluru"),
    forecast_date: Optional[str] = Query(None, description="YYYY-MM-DD (default: today)"),
    restaurant_type: Optional[str] = Query(None, description=f"One of {ALL_RT}"),
):
    """ML demand forecast for one day — all restaurant types unless one is named."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city '{city}'")
    d = date.fromisoformat(forecast_date) if forecast_date else date.today()
    wx = _weather_lookup(city, d.isoformat(), d.isoformat()).get(d.isoformat(), {})
    types = [restaurant_type] if restaurant_type else ALL_RT
    for t in types:
        if t not in ALL_RT:
            raise HTTPException(400, f"Unknown restaurant_type '{t}'. Use one of {ALL_RT}")
    return {
        "city": city, "date": d.isoformat(),
        "weather": wx or None,
        "forecasts": [
            predict_impact(city, d, t,
                           temp_max=wx.get("temp_max_c"),
                           precip_mm=wx.get("precipitation_mm"),
                           weather_code=wx.get("weather_code"),
                           temp_min=wx.get("temp_min_c"), drivers=True)
            for t in types
        ],
    }


@app.get("/forecast/range", tags=["Forecast"])
def forecast_range(
    city: str = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
):
    """ML demand forecast across a date range for every restaurant type."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city '{city}'")
    s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if (e - s).days > 120:
        raise HTTPException(400, "Range too large — max 120 days")
    wx = _weather_lookup(city, start_date, end_date)
    return {
        "city": city, "start_date": start_date, "end_date": end_date,
        "engine": engine_status()["active_engine"],
        "predictions": predict_date_range(city, s, e, wx),
    }


@app.get("/forecast/compare", tags=["Forecast"])
def forecast_compare(
    city: str = Query(...),
    forecast_date: str = Query(...),
    restaurant_type: str = Query(...),
):
    """ML model vs the legacy multiplier rules for the same day."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city '{city}'")
    d = date.fromisoformat(forecast_date)
    wx = _weather_lookup(city, forecast_date, forecast_date).get(forecast_date, {})
    return compare_engines(city, d, restaurant_type,
                           temp_max=wx.get("temp_max_c"),
                           precip_mm=wx.get("precipitation_mm"),
                           weather_code=wx.get("weather_code"))


# ─── Chat, ingestion & health endpoints ───────────────────────────────────────
from fastapi import Body                                            # noqa: E402


@app.post("/chat", tags=["Assistant"])
def chat_endpoint(question: str = Body(..., embed=True)):
    """Ask a demand question in plain language.

    The assistant answers by calling the trained model — `tool_trace` shows
    exactly which forecasts backed the reply. Needs an LLM key
    (ANTHROPIC_API_KEY or OPENAI_API_KEY); without one it returns 503.
    """
    from ml.chat_agent import answer
    from ml.llm import LLMUnavailable

    try:
        res = answer(question)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc))
    return {"question": question, "answer": res["text"], "tool_trace": res["tool_trace"]}


@app.get("/llm/status", tags=["Assistant"])
def llm_status():
    """Whether the LLM layer is configured, and with which provider."""
    from ml.llm import status
    return status()


@app.get("/ingest/pending", tags=["Events"])
def ingest_pending():
    """Candidate events proposed by the LLM and awaiting human approval."""
    from ml.event_ingest import load_pending
    pending = load_pending()
    return {"count": len(pending), "candidates": pending}


@app.post("/ingest/propose", tags=["Events"])
def ingest_propose(year: Optional[int] = Body(None, embed=True),
                   text: Optional[str] = Body(None, embed=True)):
    """Propose calendar events for a year, or extract them from pasted text.

    Nothing is written to the events database here — candidates land in the
    review queue and must be approved.
    """
    from ml.event_ingest import propose_for_year, propose_from_text
    from ml.llm import LLMUnavailable

    if not year and not text:
        raise HTTPException(400, "supply either `year` or `text`")
    try:
        return propose_for_year(year) if year else propose_from_text(text)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc))


@app.post("/ingest/approve", tags=["Events"])
def ingest_approve(ids: List[str] = Body(..., embed=True)):
    """Merge approved candidate events into the events database."""
    from ml.event_ingest import approve_ids
    return approve_ids(ids)


@app.post("/ingest/reject", tags=["Events"])
def ingest_reject(ids: List[str] = Body(..., embed=True)):
    """Drop candidate events from the review queue."""
    from ml.event_ingest import reject_ids
    return reject_ids(ids)
