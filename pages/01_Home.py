"""
India Events & Weather Dashboard — Self-Contained Version
=========================================================
Works on Streamlit Cloud with NO separate API server needed.
- Weather: calls Open-Meteo directly
- Events: reads events_db.py directly
Run: streamlit run dashboard.py
"""

import streamlit as st
import requests
import sys
import os
from datetime import date, timedelta, datetime
import pytz

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.india_geo import CITIES, ZONES
from data.events_db import EVENTS_DB, _event_applies_to_geo
from utils.restaurant_impact import predict_impact, ALL_RT, EVENT_MULTIPLIERS, BASELINE_AOV
from components.branding import APP_NAME, TAGLINE, page_icon, inject_theme, sidebar_header, page_logo, INK, MUTE, LINE, PANEL, RED

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
page_logo()

# ── Open-Meteo ─────────────────────────────────────────────────────────────────
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

WMO_EMOJI = {
    0:("","Clear Sky"),      1:("","Mainly Clear"),  2:("","Partly Cloudy"),
    3:("","Overcast"),       45:("","Fog"),           48:("","Icy Fog"),
    51:("","Light Drizzle"),53:("","Drizzle"),       55:("","Heavy Drizzle"),
    61:("","Light Rain"),   63:("","Moderate Rain"), 65:("","Heavy Rain"),
    80:("","Showers"),      81:("","Showers"),       82:("","Violent Showers"),
    95:("","Thunderstorm"), 96:("","Thunderstorm"),  99:("","Severe Storm"),
}

def wmo(code):
    return WMO_EMOJI.get(code, ("","Unknown"))

@st.cache_data(ttl=900, show_spinner=False)
def get_forecast(city):
    coords = CITIES.get(city, {})
    if not coords: return []
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": coords["lat"], "longitude": coords["lon"],
            "daily": ["temperature_2m_max","temperature_2m_min",
                      "precipitation_sum","weathercode","precipitation_probability_max"],
            "forecast_days": 8, "timezone": "Asia/Kolkata",
        }, timeout=10)
        if r.status_code != 200: return []
        d = r.json()["daily"]
        return [{"date": d["time"][i],
                 "temp_max_c": d["temperature_2m_max"][i],
                 "temp_min_c": d["temperature_2m_min"][i],
                 "precipitation_mm": d["precipitation_sum"][i] or 0,
                 "rain_probability_pct": d["precipitation_probability_max"][i],
                 "weather_code": d["weathercode"][i],
                 } for i in range(len(d["time"]))]
    except: return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical(city, days_back=30):
    coords = CITIES.get(city, {})
    if not coords: return []
    today    = date.today()
    start_dt = (today - timedelta(days=days_back)).isoformat()
    end_dt   = today.isoformat()          # include today so archive covers it
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": coords["lat"], "longitude": coords["lon"],
            "daily": ["temperature_2m_max","temperature_2m_min",
                      "precipitation_sum","weathercode"],
            "start_date": start_dt, "end_date": end_dt,
            "timezone": "Asia/Kolkata",
        }, timeout=15)
        if r.status_code != 200: return []
        d = r.json()["daily"]
        return [{"date": d["time"][i],
                 "temp_max_c": d["temperature_2m_max"][i],
                 "temp_min_c": d["temperature_2m_min"][i],
                 "precipitation_mm": d["precipitation_sum"][i] or 0,
                 "weather_code": d["weathercode"][i],
                 } for i in range(len(d["time"]))]
    except: return []

def get_events(start_str, end_str, city=None, category=None):
    from datetime import date as dt_date
    s = dt_date.fromisoformat(start_str)
    e = dt_date.fromisoformat(end_str)
    out = []
    for ev in EVENTS_DB:
        if ev["category"] == "Weather Event": continue  # weather stays backend-only
        ev_s = dt_date.fromisoformat(ev["start_date"])
        ev_e = dt_date.fromisoformat(ev["end_date"])
        if ev_e < s or ev_s > e: continue
        if category and category != "All" and ev["category"] != category: continue
        if city and not _event_applies_to_geo(ev, city=city): continue
        out.append(ev)
    return out

def get_today_context(city):
    today_str = date.today().isoformat()
    evts = get_events(today_str, today_str, city=city)
    flags = {
        "has_festival":         any(e["category"]=="Festival" for e in evts),
        "has_gov_holiday":      any(e["category"]=="Government Holiday" for e in evts),
        "has_sports_event":     any(e["category"]=="Sports Event" for e in evts),
        "has_commercial_event": any(e["category"]=="Commercial Event" for e in evts),
        "has_public_event":     any(e["category"]=="Public Event" for e in evts),
        "has_emergency_crisis": any(e["category"]=="Emergency Crisis" for e in evts),
        "is_weekend":           date.today().weekday() >= 5,
    }
    impacts = [e["impact_on_demand"] for e in evts]
    if   "very_low"    in impacts: signal = "very_low"
    elif "low"         in impacts: signal = "low"
    elif "very_high_up"in impacts: signal = "very_high_up"
    elif "high_up"     in impacts: signal = "high_up"
    elif "slight_up"   in impacts: signal = "slight_up"
    elif "slight_down" in impacts: signal = "slight_down"
    else:                          signal = "neutral"
    return {"events": evts, "flags": flags, "signal": signal}

# ── Style ──────────────────────────────────────────────────────────────────────
IMPACT_COLOR = {
    "very_high_up":"#00C853","high_up":"#43A047","slight_up":"#1B5E20",
    "neutral_positive":"#78909C","neutral":"#78909C","neutral_to_slight_down":"#FFB74D",
    "slight_down":"#D84315","low":"#C62828","very_low":"#B71C1C",
}
IMPACT_LABEL = {
    "very_high_up":"Very High Demand","high_up":"High Demand",
    "slight_up":"↑ Slight Uplift","neutral_positive":"+ Neutral-Positive",
    "neutral":"-> Neutral","neutral_to_slight_down":"↘ Slight Suppression",
    "slight_down":"↓ Suppressed","low":"Low Demand","very_low":"Very Low Demand",
}
CAT_EMOJI = {
    "Festival":"","Government Holiday":"","Sports Event":"",
    "Commercial Event":"","Public Event":"",
    "Government Order":"","Emergency Crisis":"!",
}
CAT_COLOR = {
    "Festival":"#FF6F00","Government Holiday":"#1565C0","Sports Event":"#6A1B9A",
    "Commercial Event":"#00695C","Public Event":"#2E7D32",
    "Government Order":"#37474F","Emergency Crisis":"#B71C1C",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
# Note: page-wide theme (canvas/sidebar/buttons) comes from inject_theme() above.
# This block only defines the page-specific card classes actually used below —
# dead classes from an earlier design (weather-hero, metric-mini) and the
# duplicate section-hdr/sidebar rules that used to fight inject_theme() have
# been removed.
st.markdown(f"""
<style>
.forecast-card{{background:{PANEL};border:1px solid {LINE};
  border-radius:16px;padding:14px 8px;text-align:center;color:{INK}}}
.forecast-day{{font-size:15px;font-weight:600;opacity:.8}}
.forecast-emoji{{font-size:31px;margin:8px 0}}
.forecast-hi{{font-size:20px;font-weight:600}}
.forecast-lo{{font-size:15px;opacity:.55}}
.forecast-rain{{font-size:14px;color:#0D47A1;margin-top:4px}}
.event-card{{border-radius:14px;padding:14px 16px;margin-bottom:10px;
  border-left:4px solid;color:{INK}}}
.event-name{{font-size:17px;font-weight:600}}
.event-dates{{font-size:14px;opacity:.7;margin-top:2px}}
.context-card{{background:{PANEL};border:1px solid {LINE};
  border-radius:16px;padding:18px 20px;color:{INK}}}
.signal-badge{{display:inline-block;padding:5px 14px;border-radius:20px;
  font-size:16px;font-weight:600;margin-top:6px}}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
# Categories shown/filterable to the board — "Weather Event" stays in the DB
# and continues to feed the backend demand model, but is never surfaced as
# a browsable event category (see requirement: hide raw weather, show impact).
DISPLAY_CATS = ["Festival","Government Holiday","Sports Event",
                 "Commercial Event","Public Event","Government Order",
                 "Emergency Crisis"]

sidebar_header("Home")
with st.sidebar:
    city = st.selectbox("Select City", sorted(CITIES.keys()),
                        index=sorted(CITIES.keys()).index("New Delhi"))
    st.markdown("---")
    st.markdown("### Date Range")
    IST = pytz.timezone("Asia/Kolkata")
    today = datetime.now(IST).date()
    c1,c2 = st.columns(2)
    with c1: start_date = st.date_input("From", today)
    with c2: end_date   = st.date_input("To",   today+timedelta(days=30))
    st.markdown("---")
    st.markdown("###  Filter Events")
    cats = ["All"] + DISPLAY_CATS
    selected_cat = st.selectbox("Category", cats)
    st.markdown("---")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.success("Self-contained mode")
    # Values computed outside the f-string — nested same-type quotes inside an
    # f-string only parse on Python 3.12+, and Streamlit Cloud may run 3.11.
    _n_tracked = len([e for e in EVENTS_DB if e["category"] in DISPLAY_CATS])
    _ist_now = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%d %b %Y %I:%M %p")
    st.markdown(
        f"<div style='color:rgba(0,0,0,.5);font-size:14px'>"
        f" {_n_tracked} events tracked<br>IST: {_ist_now}<br>Date: {today.strftime('%d %b %Y')}</div>",
        unsafe_allow_html=True)

# ── Fetch data (weather is fetched for the backend model only — never shown) ────
with st.spinner("Loading demand signals..."):
    forecast  = get_forecast(city)
    hist_data = get_historical(city, days_back=30)
    today_ctx = get_today_context(city)

# Try forecast first, fall back to archive if today missing
today_w = next((f for f in forecast if f.get("date") == today.isoformat()), {})
if not today_w or today_w.get("temp_max_c") is None:
    arch_today = next((h for h in hist_data if h.get("date") == today.isoformat()), {})
    if arch_today.get("temp_max_c") is not None:
        today_w = arch_today
        if forecast:
            today_w["rain_probability_pct"] = forecast[0].get("rain_probability_pct", "--")

def _classify_weather_impact(day):
    """Turn a raw weather-forecast day into a board-friendly demand signal.
    Deliberately never surfaces °C / mm figures — business impact only."""
    code   = day.get("weather_code")
    precip = day.get("precipitation_mm") or 0
    tmax   = day.get("temp_max_c")
    if code in (95, 96, 99):
        return ("Severe Weather Disruption", "#B71C1C",
                "Thunderstorms likely — dine-in & PBCL footfall drop sharply, delivery demand spikes.")
    if precip >= 20:
        return ("Delivery Surge Expected", "#1565C0",
                "Heavy rain likely — dine-in visits fall, QSR & Cloud Kitchen delivery orders rise sharply.")
    if precip >= 8:
        return ("Moderate Delivery Lift", "#1976D2",
                "Rain likely — modest shift from dine-in to delivery across formats.")
    if tmax is not None and tmax >= 42:
        return ("Heat Suppression Risk", "#E65100",
                "Extreme heat forecast — outdoor and daytime footfall likely to dip.")
    if tmax is not None and tmax <= 18:
        return ("Cosy Dining Boost", "#00838F",
                "Cooler weather — dine-in and café visits typically rise.")
    return ("Normal Demand Day", "#43A047",
            "No major weather-driven demand shift expected.")

# ── TODAY'S DEMAND SNAPSHOT (board-facing — no raw weather figures) ────────────
sig    = today_ctx["signal"]
flags  = today_ctx["flags"]
evc    = len(today_ctx["events"])
sigc   = IMPACT_COLOR.get(sig,"#78909C")
sigl   = IMPACT_LABEL.get(sig,sig)
fm     = {"has_festival":("","Festival"),"has_gov_holiday":("","Holiday"),
          "has_sports_event":("","Sports"),"has_commercial_event":("","Commercial"),
          "has_public_event":("","Public"),"has_emergency_crisis":("!","Crisis"),
          "is_weekend":("","Weekend")}
active = [f"{i} {l}" for k,(i,l) in fm.items() if flags.get(k)]
today_wx_label, today_wx_color, today_wx_note = _classify_weather_impact(today_w)
if today_wx_label != "Normal Demand Day":
    active.append(today_wx_label)
badges = " &nbsp;".join(
    [f"<span style='background:rgba(0,0,0,.15);padding:3px 10px;border-radius:12px;font-size:14px'>{f}</span>"
     for f in active]
) if active else "<span style='opacity:.45;font-size:15px'>No special demand factors today</span>"

st.markdown(f"""
<div class="context-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px">
    <div>
      <div style="font-size:23px;font-weight:600"> {city} &nbsp;·&nbsp; Today's Demand Snapshot</div>
      <div style="font-size:15px;opacity:.6;margin-top:4px">{today.strftime('%A, %d %B %Y')}</div>
      <div class="signal-badge" style="background:{sigc}22;border:1px solid {sigc};color:{sigc}">{sigl}</div>
    </div>
    <div style="text-align:right;min-width:220px">
      <div style="font-size:14px;font-weight:500;opacity:.6;margin-bottom:6px">WEATHER IMPACT TODAY</div>
      <div style="font-size:17px;font-weight:600;color:{today_wx_color}">{today_wx_label}</div>
      <div style="font-size:14px;opacity:.6;margin-top:4px;max-width:260px">{today_wx_note}</div>
    </div>
  </div>
  <div style="margin-top:16px;font-size:14px;font-weight:500;opacity:.6;margin-bottom:8px">ACTIVE FACTORS</div>
  <div style="line-height:2">{badges}</div>
  <div style="margin-top:10px;font-size:15px;opacity:.5">{evc} event(s) active today</div>
</div>""", unsafe_allow_html=True)

# ── UPCOMING WEATHER IMPACT (7 days) ────────────────────────────────────────────
st.markdown('<div class="section-hdr"> Upcoming Weather Impact — Next 7 Days</div>', unsafe_allow_html=True)
if forecast:
    cols = st.columns(min(len(forecast),8))
    for i,day in enumerate(forecast[:8]):
        with cols[i]:
            d   = date.fromisoformat(day["date"])
            dn  = "Today" if d==today else ("Tomorrow" if d==today+timedelta(1) else d.strftime("%a"))
            label, color, note = _classify_weather_impact(day)
            bdr = "border:1.5px solid rgba(0,0,0,.5);" if d==today else ""
            st.markdown(f"""
            <div class="forecast-card" style="{bdr};text-align:left;padding:14px">
              <div class="forecast-day" style="text-align:center">{dn}<br>{d.strftime('%d %b')}</div>
              <div style="margin-top:10px;font-size:15px;font-weight:600;color:{color};text-align:center">{label}</div>
              <div style="font-size:13px;opacity:.55;margin-top:6px;line-height:1.5">{note}</div>
            </div>""", unsafe_allow_html=True)
else:
    st.info("Weather-driven demand signal unavailable right now — event-based demand factors below are unaffected.")

# ── EVENTS ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr"> Events & Holidays</div>', unsafe_allow_html=True)

try:
    from ml.product_advisor import recommend as _pa_recommend
    from data.dotpe_products import contextual_pitch as _pa_contextual_pitch
    _PA_AVAIL = True
except Exception:
    _PA_AVAIL = False

# Build weather lookup from already-fetched forecast + historical
_wmap = {}
for _d in (hist_data or []):
    if _d.get("date"): _wmap[_d["date"]] = _d
for _d in (forecast or []):
    if _d.get("date"): _wmap[_d["date"]] = _d

def _factor_badge(text, bg, fc):
    return (f"<span style='background:{bg};color:{fc};border:1px solid {fc}44;"
            f"border-radius:8px;padding:3px 9px;font-size:13.5px;font-weight:500;"
            f"white-space:nowrap;display:inline-block'>{text}</span>")

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_day_impact(city_: str, d_iso: str, rt: str,
                       temp_max, precip_mm, weather_code) -> dict | None:
    """One (city, date, restaurant_type) model call, cached at the finest
    grain so overlapping festivals — Navratri, Dussehra, Durga Puja and
    Diwali all sit in the same October window — share cache hits instead of
    each event independently re-running the model over the same calendar
    days. This is what actually drove the CPU throttle: without this, N
    overlapping events cost N× the compute for identical days."""
    from datetime import date as _dt
    try:
        return predict_impact(city_, _dt.fromisoformat(d_iso), rt,
                              temp_max=temp_max, precip_mm=precip_mm,
                              weather_code=weather_code, explain=False)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _event_model_impact(city_: str, ev_start: str, ev_end: str, wmap_: dict) -> dict:
    """Real predict_impact() per restaurant type for an event, averaged over
    every day in the event's span that has weather data (falls back to a
    no-weather call for days outside the ~38-day window we fetch weather for).
    This is the same trained model / same weather feed the rest of the app
    uses — not the static EVENT_MULTIPLIERS lookup table."""
    from datetime import date as _dt, timedelta as _td
    s, e = _dt.fromisoformat(ev_start), _dt.fromisoformat(ev_end)
    # Cap very long events (e.g. month-long observances) to 14 sampled days
    # so this stays cheap; a festival's demand effect doesn't need every day.
    days = []
    cur = s
    while cur <= e and len(days) < 14:
        days.append(cur); cur += _td(days=1)

    out = {}
    for rt in ALL_RT:
        pct_vals, aov_vals, had_weather = [], [], False
        for d in days:
            w = wmap_.get(d.isoformat(), {})
            if w:
                had_weather = True
            p = _cached_day_impact(city_, d.isoformat(), rt,
                                   w.get("temp_max_c"), w.get("precipitation_mm"),
                                   w.get("weather_code"))
            if p is None:
                continue
            pct_vals.append(p["pct_change"])
            if p.get("predicted_aov"):
                aov_vals.append(p["predicted_aov"])
        if pct_vals:
            out[rt] = {
                "pct": sum(pct_vals) / len(pct_vals),
                "peak_pct": max(pct_vals, key=abs),
                "aov": (sum(aov_vals) / len(aov_vals)) if aov_vals else None,
                "has_weather": had_weather,
            }
    return out


def _signal_for_pct(pct: float) -> str:
    """Same thresholds ml/restaurant_impact uses internally, applied here so
    signal stays consistent whichever code path computed pct."""
    if pct >= 30:  return "very_high_up"
    if pct >= 15:  return "high_up"
    if pct >= 5:   return "slight_up"
    if pct <= -35: return "very_low"
    if pct <= -20: return "low"
    if pct <= -8:  return "slight_down"
    return "neutral"

def _all_factors(ev):
    from datetime import date as _dt, timedelta as _td
    ev_s = _dt.fromisoformat(ev["start_date"])
    ev_e = _dt.fromisoformat(ev["end_date"])

    # ── Row 1: Restaurant impact — this event's OWN multiplier only ──────────
    # Deliberately the simple, single-event lookup (this event's category and
    # subcategory against the hand-audited EVENT_MULTIPLIERS table) rather
    # than the full model averaged across every day of the event's span.
    # The fuller model-based version compounds EVERY event active on a given
    # day together (Navratri + Durga Puja + a 6-month sports season + a
    # commercial sale, all multiplying at once) plus the weekend baseline
    # bump, which is how a card ended up claiming a 178% swing — technically
    # traceable, but far more than this one event actually drives, and not
    # what a rep should read as "this festival's effect." This card is about
    # this one event's own, standalone effect.
    cat = ev.get("category", ""); sub = ev.get("subcategory", "*")
    cm = EVENT_MULTIPLIERS.get(cat, {})
    sm = cm.get(sub, cm.get("*", {}))
    model_impact = {}
    for rt in ALL_RT:
        mult = sm.get(rt, 1.0)
        model_impact[rt] = {
            "pct": (mult - 1) * 100,
            "aov": BASELINE_AOV.get(rt, 0) * mult,
            "has_weather": False,
        }

    rt_badges = []
    for rt,ico in [("QSR",""),("Fine Dining",""),("PBCL",""),("Cloud Kitchen",""),("Cafe","")]:
        mi = model_impact.get(rt)
        pct = round(mi["pct"]) if mi else 0
        if pct>0:   txt,bg,fc = f"{ico} {rt} ↑+{pct}%","rgba(67,160,71,.25)","#1B5E20"
        elif pct<0: txt,bg,fc = f"{ico} {rt} ↓{pct}%","rgba(239,83,80,.25)","#B71C1C"
        else:       txt,bg,fc = f"{ico} {rt} →0%","rgba(120,144,156,.18)","#455A64"
        rt_badges.append(_factor_badge(txt,bg,fc))

    # ── Weather summary for this event's span — computed here (moved up)
    # so the Sales Playbook pitch lines below can reference it too, not just
    # the Weather badge row further down. ────────────────────────────────────
    _temps=[]; _precips=[]; _codes=[]
    _cur = ev_s
    while _cur <= ev_e:
        _w = _wmap.get(_cur.isoformat(),{})
        if _w.get("temp_max_c") is not None: _temps.append(_w["temp_max_c"])
        if _w.get("precipitation_mm") is not None: _precips.append(_w["precipitation_mm"] or 0)
        if _w.get("weather_code") is not None: _codes.append(_w["weather_code"])
        _cur += _td(days=1)
    weather_note = None   # only set for weather notable enough to change a pitch
    if _temps and max(_temps) >= 40:
        weather_note = f"forecast heat of {max(_temps):.0f}°C will also cut outdoor/dine-in footfall"
    if _precips and sum(_precips) >= 8:
        weather_note = "forecast rain will push delivery up and dine-in down further"
    if _codes:
        from collections import Counter as _Counter
        if _Counter(_codes).most_common(1)[0][0] in (95,96,99):
            weather_note = "thunderstorms in the forecast will suppress footfall sharply"

    # ── Row 1b: Sales playbook — ALL relevant products, deduped to families ──
    # Driven by the same model_impact figures above (event + weather). Every
    # restaurant type returns up to 3 ranked products from product_advisor
    # (e.g. surge_delivery → SOVA, QR Ordering, Rista bundle); previously only
    # the #1 pick per type was shown, which meant WABA / QR Ordering / Horizon
    # never surfaced even though they're valid secondary pitches. Now every
    # returned product is collected, deduped by family, and each family notes
    # which restaurant types it's the TOP pick for vs. which it's a fit for.
    # The pitch line itself is built fresh per event from contextual_pitch() —
    # naming this event and its peak %, not a static description — rather
    # than the generic family_pitch that came back from product_advisor.
    sales_families = {}   # family -> {"primary": [...], "secondary": [...], "peak_pct": float, "up": bool}
    if _PA_AVAIL:
        for rt,ico in [("QSR",""),("Fine Dining",""),("PBCL",""),("Cloud Kitchen",""),("Cafe","")]:
            mi = model_impact.get(rt)
            if not mi:
                continue
            pct = mi["pct"]
            sig = _signal_for_pct(pct)
            try:
                rec = _pa_recommend(rt, sig, pct, mi.get("aov"))
                prods = rec["recommended_products"]
            except Exception:
                prods = []
            is_up = sig in ("very_high_up","high_up","slight_up")
            for idx, prod in enumerate(prods):
                fam = prod.get("family", prod["name"])
                entry = sales_families.setdefault(fam, {"primary": [], "secondary": [],
                                                          "peak_pct": pct, "up": True})
                (entry["primary"] if idx == 0 else entry["secondary"]).append((ico, rt))
                entry["up"] = entry["up"] and is_up
                if abs(pct) > abs(entry["peak_pct"]):
                    entry["peak_pct"] = pct

    sales_blocks = []
    # Families with a #1 pick somewhere first, then families that only ever
    # showed up as a secondary/also-fits pick.
    ordered = sorted(sales_families.items(), key=lambda kv: (not kv[1]["primary"],))
    for fam, info in ordered:
        primary_types  = ", ".join(f"{ico} {rt}" for ico,rt in info["primary"])
        secondary_types = ", ".join(f"{ico} {rt}" for ico,rt in info["secondary"]
                                     if (ico,rt) not in info["primary"])
        fit_lines = []
        if primary_types:
            fit_lines.append(f"<b>Top pick for:</b> {primary_types}")
        if secondary_types:
            fit_lines.append(f"<b>Also fits:</b> {secondary_types}")
        pitch = _pa_contextual_pitch(fam, info["peak_pct"], [ev["name"]], weather_note) if _PA_AVAIL else ""
        bg, fc, bd = ("rgba(67,160,71,.12)", "#1B5E20", "#43A047") if info["up"] else ("rgba(255,143,0,.12)", "#E65100", "#FF8F00")
        sales_blocks.append(f"""
        <div style="background:{bg};border-left:3px solid {bd};border-radius:8px;
          padding:8px 12px;margin-bottom:6px">
          <div style="font-size:15px;font-weight:700;color:{fc}">- {fam}</div>
          <div style="font-size:13px;opacity:.55;margin-top:1px">{" &nbsp;·&nbsp; ".join(fit_lines)}</div>
          <div style="font-size:13.5px;opacity:.75;margin-top:4px;line-height:1.4">{pitch}</div>
        </div>""")

    # ── Row 2: Weather factors — reuses _temps/_precips/_codes computed above
    wx_badges = []
    if _temps:
        mx = max(_temps); av = sum(_temps)/len(_temps)
        if mx>=44:    wx_badges.append(_factor_badge("Extreme Heat — footfall ↓↓","rgba(183,28,28,.3)","#B71C1C"))
        elif mx>=40:  wx_badges.append(_factor_badge("Hot Weather — outdoor footfall ↓","rgba(230,81,0,.25)","#E65100"))
        elif mx>=30:  wx_badges.append(_factor_badge("Warm Weather — normal footfall","rgba(46,125,50,.2)","#1B5E20"))
        else:         wx_badges.append(_factor_badge("Mild Weather — good for dining","rgba(21,101,192,.2)","#0D47A1"))
    if _precips:
        total_p = sum(_precips)
        if total_p>=25:   wx_badges.append(_factor_badge("Heavy Rain — delivery ↑↑ dine-in ↓↓","rgba(13,71,161,.3)","#0D47A1"))
        elif total_p>=8:  wx_badges.append(_factor_badge("Rain — delivery ↑ dine-in ↓","rgba(21,101,192,.25)","#01579B"))
        elif total_p>=2:  wx_badges.append(_factor_badge("Light Drizzle — minor impact","rgba(21,101,192,.15)","#0277BD"))
        else:             wx_badges.append(_factor_badge("Dry weather","rgba(46,125,50,.15)","#2E7D32"))
    if _codes:
        from collections import Counter
        mc = Counter(_codes).most_common(1)[0][0]
        if mc in (95,96,99): wx_badges.append(_factor_badge("Thunderstorms — severe suppression","rgba(183,28,28,.35)","#C62828"))
    if not wx_badges:
        wx_badges.append(_factor_badge("Future dates — forecast not available yet","rgba(120,144,156,.15)","#78909C"))

    # ── Row 3: Weekday / Weekend ──────────────────────────────────────────────
    days_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    all_days=[]; cur=ev_s
    while cur<=ev_e: all_days.append(cur); cur+=_td(days=1)
    wkdays=[d for d in all_days if d.weekday()<5]
    wkends=[d for d in all_days if d.weekday()>=5]
    day_badges=[]
    if wkdays:
        names=", ".join(days_names[d.weekday()] for d in wkdays[:3])+("…" if len(wkdays)>3 else "")
        day_badges.append(_factor_badge(f" {len(wkdays)} Weekday{'s' if len(wkdays)>1 else ''} ({names}) — normal base","rgba(21,101,192,.2)","#0D47A1"))
    if wkends:
        names=", ".join(days_names[d.weekday()] for d in wkends[:3])+("…" if len(wkends)>3 else "")
        day_badges.append(_factor_badge(f" {len(wkends)} Weekend day{'s' if len(wkends)>1 else ''} ({names}) — higher base demand","rgba(255,143,0,.25)","#F57F17"))
    if not day_badges:
        day_badges.append(_factor_badge("Single day event","rgba(120,144,156,.15)","#78909C"))

    def _row(label, badges):
        inner = " ".join(badges)
        return (f"<div style='margin-top:8px'>"
                f"<div style='font-size:12.5px;font-weight:700;opacity:.4;letter-spacing:.08em;"
                f"text-transform:uppercase;margin-bottom:5px'>{label}</div>"
                f"<div style='display:flex;gap:5px;flex-wrap:wrap'>{inner}</div></div>")

    out = _row("Restaurant Impact", rt_badges)
    if sales_blocks:
        all_types = [t for info in sales_families.values() for t in info["primary"] + info["secondary"]]
        no_wx = any(not model_impact.get(rt, {}).get("has_weather", True) for _, rt in all_types)
        # One-line framing so it's explicit which situation this event puts a
        # restaurant in — growth/cross-sell when demand is rising, recovery
        # when it's falling — before the specific product cards below.
        up_count = sum(1 for mi in model_impact.values() if mi["pct"] > 0)
        down_count = sum(1 for mi in model_impact.values() if mi["pct"] < 0)
        if up_count >= down_count:
            frame = ("<b>Demand is rising</b> — the opportunity here is to sell more, "
                     "not just one thing: layer a delivery/ordering product with a "
                     "growth or CRM tool to convert the extra footfall into repeat customers.")
        else:
            frame = ("<b>Demand is falling</b> — this is a recovery situation, not a "
                     "growth pitch: lead with win-back, loyalty and re-engagement "
                     "tools to bring volume back up, not higher-value bundles.")
        out += (f"<div style='margin-top:8px'>"
                f"<div style='font-size:12.5px;font-weight:700;opacity:.4;letter-spacing:.08em;"
                f"text-transform:uppercase;margin-bottom:5px'>- Sales Playbook — what to pitch</div>"
                f"<div style='font-size:13.5px;opacity:.75;margin-bottom:8px;line-height:1.5'>{frame}</div>"
                f"{''.join(sales_blocks)}</div>")
        if no_wx:
            out += ("<div style='font-size:12px;opacity:.4;margin-top:-2px;margin-bottom:6px'>"
                     "* based on event effect for date(s) outside the ~5-week live weather window</div>")
    out += _row("Weather", wx_badges) + _row("Day Type", day_badges)
    return out

events = get_events(start_date.isoformat(), end_date.isoformat(),
                    city=city, category=selected_cat)
if events:
    c1,c2 = st.columns(2)
    for i,ev in enumerate(events):
        col   = c1 if i%2==0 else c2
        cat   = ev.get("category","")
        ico   = CAT_EMOJI.get(cat,"")
        color = CAT_COLOR.get(cat,"#546E7A")
        imp   = ev.get("impact_on_demand","neutral")
        impl  = IMPACT_LABEL.get(imp,imp)
        impc  = IMPACT_COLOR.get(imp,"#455A64")
        scope = ev.get("scope","").replace("_"," ").title()
        sd    = ev.get("start_date","")
        ed    = ev.get("end_date","")
        ds    = sd if sd==ed else f"{sd} → {ed}"
        desc  = ev.get("description","")
        dh    = f"<div style='font-size:14.5px;opacity:.6;margin-top:8px;line-height:1.6;border-top:1px solid rgba(0,0,0,.08);padding-top:8px'>{desc}</div>" if desc else ""
        factors = _all_factors(ev)
        with col:
            st.markdown(f"""
            <div class="event-card" style="background:{color}22;border-left-color:{color}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div class="event-name">{ico} {ev.get('name','')}</div>
                  <div class="event-dates"> {ds} &nbsp;·&nbsp;  {scope}</div>
                </div>
                <span style="background:{color}33;border:1px solid {color}66;color:{color};
                  font-size:13px;padding:2px 8px;border-radius:10px;white-space:nowrap;
                  margin-left:8px">{cat}</span>
              </div>
              <div style="font-size:14px;color:{impc};margin-top:6px;font-weight:600">{impl}</div>
              {factors}
              {dh}
            </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:rgba(0,0,0,.05);border-radius:16px;padding:32px;
      text-align:center;color:rgba(0,0,0,.4)">
       No events found for the selected filters and date range.
    </div>""", unsafe_allow_html=True)
# ── HISTORICAL DEMAND INDEX (order + price-adjusted revenue) ───────────────────
st.markdown('<div class="section-hdr"> Historical Demand Index — Last 30 Days</div>',
            unsafe_allow_html=True)
hist = hist_data  # already fetched above (weather stays backend-only, feeds this model)
if hist:
    try:
        import plotly.graph_objects as go

        @st.cache_data(ttl=1800, show_spinner=False)
        def _composite_index_series(city_: str, hist_: list) -> tuple:
            """Composite order/revenue index over the history window. Uses the
            same per-day cache as the event Sales Playbook above, so a date
            that's both in the last-30-days window and inside an active
            event's span is only ever scored once."""
            dts_, oi_, ri_ = [], [], []
            for d in hist_:
                dd_iso = d["date"]
                per_type = [_cached_day_impact(city_, dd_iso, rt,
                                d.get("temp_max_c"), d.get("precipitation_mm"),
                                d.get("weather_code"))
                            for rt in ALL_RT]
                per_type = [p for p in per_type if p is not None]
                if not per_type:
                    continue
                dts_.append(dd_iso)
                oi_.append(sum(p["predicted_index"] for p in per_type)/len(per_type))
                ri_.append(sum(p["predicted_revenue_index"] for p in per_type)/len(per_type))
            return dts_, oi_, ri_

        dts, order_idx, rev_idx = _composite_index_series(city, hist)
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=dts,y=order_idx,name="Composite Order Index",
            line=dict(color="#1565C0",width=2.5),
            fill="tozeroy",fillcolor="rgba(66,165,245,.1)"))
        fig.add_trace(go.Scatter(x=dts,y=rev_idx,name="Composite Revenue Index (price-adjusted)",
            line=dict(color="#F57F17",width=2,dash="dash")))
        fig.add_hline(y=100,line_dash="dot",line_color="rgba(0,0,0,.2)",
                      annotation_text="Baseline",annotation_font_color="rgba(0,0,0,.35)")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,.04)",
            font=dict(color="white",family="Inter"),height=280,
            margin=dict(l=0,r=0,t=10,b=0),hovermode="x unified",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(0,0,0,.08)",tickfont=dict(color="rgba(0,0,0,.6)")),
            yaxis=dict(title="Index (100 = baseline)",gridcolor="rgba(0,0,0,.08)",
                       tickfont=dict(color="rgba(0,0,0,.6)")),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Chart unavailable: {e}")
else:
    st.info("Historical demand data unavailable.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:rgba(0,0,0,.25);font-size:14px;
  margin-top:32px;padding-top:16px;border-top:1px solid rgba(0,0,0,.08)">
   {APP_NAME} — {TAGLINE}
  &nbsp;·&nbsp; {len([e for e in EVENTS_DB if e['category'] in DISPLAY_CATS])} events tracked &nbsp;·&nbsp; {city}
</div>""", unsafe_allow_html=True)
