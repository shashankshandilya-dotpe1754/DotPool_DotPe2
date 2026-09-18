"""
ml/features.py
==============
The single source of truth for feature engineering.

Both the training pipeline (ml/train.py) and the live inference layer
(ml/predict.py) call `build_row()` / `rows_to_frame()` from here, so a feature
can never drift between training and serving.

Design notes
------------
* No hand-tuned demand multipliers appear anywhere in this file. The events DB
  contributes *descriptive* features only — which categories are active, how
  many, how long they run, how far away the next festival is. What those
  features do to demand is learned by the model.
* No lag/rolling-window targets are used. The app forecasts arbitrary future
  dates (and whole date ranges) where yesterday's orders are unknown, so the
  model is a pure covariate regression on calendar + event + weather + geo.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from data.events_db import EVENTS_DB, _event_applies_to_geo
from data.india_geo import CITIES, CITY_LIST, STATE_LIST, ZONE_LIST
from ml.config import (
    EVENT_CATEGORIES,
    EVENT_SUBCATEGORIES,
    IMPACT_ORDINAL,
    RESTAURANT_TYPES,
    city_tier,
)

# ── Column naming helpers ─────────────────────────────────────────────────────
def _cat_col(c: str) -> str:
    return "evt_cat_" + c.lower().replace(" ", "_")


def _sub_col(s: str) -> str:
    return "evt_sub_" + s.lower().replace(" ", "_").replace("-", "_")


CATEGORY_COLS    = [_cat_col(c) for c in EVENT_CATEGORIES]
SUBCATEGORY_COLS = [_sub_col(s) for s in EVENT_SUBCATEGORIES]

CATEGORICAL_COLUMNS = ["restaurant_type", "city", "state", "zone"]

CALENDAR_COLS = [
    "tier", "lat", "lon",
    "dow", "is_weekend", "is_saturday", "is_sunday", "is_friday",
    "month", "day_of_month", "week_of_year",
    "doy_sin", "doy_cos", "month_sin", "month_cos",
    "is_payday_window", "is_month_end", "days_from_month_start",
    "year_index", "t_index",
]

EVENT_META_COLS = [
    "n_events", "impact_sum", "impact_max", "impact_min",
    "max_event_span", "is_multiday_event",
    "days_to_next_festival", "days_since_last_festival",
    "is_festival_eve", "is_festival_day_after",
    "n_pan_india_events", "n_local_events",
    "is_long_weekend", "is_holiday_adjacent",
]

WEATHER_COLS = [
    "temp_max", "temp_min", "temp_range", "precip_mm", "log_precip",
    "weather_code", "is_rain", "is_heavy_rain", "is_thunder", "is_fog",
    "is_extreme_heat", "is_high_heat", "is_cool", "is_pleasant",
    "has_weather_data",
]

FEATURE_COLUMNS = (
    CATEGORICAL_COLUMNS
    + CALENDAR_COLS
    + CATEGORY_COLS
    + SUBCATEGORY_COLS
    + EVENT_META_COLS
    + WEATHER_COLS
)

_EPOCH = date(2023, 1, 1)


# ── Event index (built once) ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _parsed_events():
    out = []
    for ev in EVENTS_DB:
        try:
            s = date.fromisoformat(ev["start_date"])
            e = date.fromisoformat(ev["end_date"])
        except Exception:
            continue
        out.append((s, e, ev))
    return out


@lru_cache(maxsize=300000)
def active_events(city: str, d: date) -> tuple:
    """All events from the DB active on `d` and applicable to `city`."""
    hits = []
    for s, e, ev in _parsed_events():
        if s <= d <= e and _event_applies_to_geo(ev, city=city):
            hits.append(ev)
    return tuple(hits)


@lru_cache(maxsize=300000)
def _festival_dates(city: str) -> tuple:
    """Sorted start dates of festival / major-holiday events for a city."""
    ds = set()
    for s, e, ev in _parsed_events():
        if ev["category"] not in ("Festival", "Government Holiday"):
            continue
        if not _event_applies_to_geo(ev, city=city):
            continue
        cur = s
        while cur <= e:
            ds.add(cur)
            cur += timedelta(days=1)
    return tuple(sorted(ds))


def _festival_proximity(city: str, d: date) -> tuple:
    """(days_to_next, days_since_last) capped at 60."""
    ds = _festival_dates(city)
    nxt, prv = 60, 60
    for f in ds:
        delta = (f - d).days
        if 0 <= delta < nxt:
            nxt = delta
        if 0 < -delta < prv:
            prv = -delta
    return nxt, prv


def _is_holiday_on(city: str, d: date) -> bool:
    for ev in active_events(city, d):
        if ev["category"] in ("Government Holiday", "Festival"):
            return True
    return False


# ── Feature blocks ────────────────────────────────────────────────────────────
def calendar_features(city: str, d: date) -> dict:
    geo = CITIES.get(city, {})
    doy = d.timetuple().tm_yday
    dow = d.weekday()
    dim = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return {
        "tier": city_tier(city),
        "lat": geo.get("lat", np.nan),
        "lon": geo.get("lon", np.nan),
        "dow": dow,
        "is_weekend": int(dow >= 5),
        "is_saturday": int(dow == 5),
        "is_sunday": int(dow == 6),
        "is_friday": int(dow == 4),
        "month": d.month,
        "day_of_month": d.day,
        "week_of_year": d.isocalendar()[1],
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "month_sin": math.sin(2 * math.pi * d.month / 12),
        "month_cos": math.cos(2 * math.pi * d.month / 12),
        # Indian salary cycle: payday spend burst on the 1st–5th, dip end-month
        "is_payday_window": int(d.day <= 5),
        "is_month_end": int(d.day >= dim.day - 3),
        "days_from_month_start": d.day - 1,
        "year_index": d.year - _EPOCH.year,
        "t_index": (d - _EPOCH).days,   # lets the model learn a slow trend
    }


def event_features(city: str, d: date, evs=None, proximity: bool = True) -> dict:
    """Event features for `city` on `d`.

    `evs` may be passed explicitly to score counterfactuals — e.g. the same day
    with one event (or all events) removed, which is how ml/predict.py measures
    each event's individual contribution.
    """
    evs = active_events(city, d) if evs is None else tuple(evs)
    f = {c: 0 for c in CATEGORY_COLS}
    f.update({c: 0 for c in SUBCATEGORY_COLS})

    impacts, spans, pan, local = [], [], 0, 0
    for ev in evs:
        col = _cat_col(ev["category"])
        if col in f:
            f[col] += 1
        sub = _sub_col(ev.get("subcategory") or "")
        if sub in f:
            f[sub] = 1
        impacts.append(IMPACT_ORDINAL.get(ev.get("impact_on_demand", "neutral"), 0))
        try:
            spans.append(
                (date.fromisoformat(ev["end_date"]) - date.fromisoformat(ev["start_date"])).days + 1
            )
        except Exception:
            spans.append(1)
        if ev.get("scope") == "pan_india":
            pan += 1
        else:
            local += 1

    if proximity:
        nxt, prv = _festival_proximity(city, d)
        dow = d.weekday()
        adj_holiday = _is_holiday_on(city, d - timedelta(days=1)) or _is_holiday_on(
            city, d + timedelta(days=1)
        )
        long_weekend = int(
            (dow == 0 and _is_holiday_on(city, d))       # Monday holiday
            or (dow == 4 and _is_holiday_on(city, d))    # Friday holiday
            or (dow >= 5 and adj_holiday)
        )
    else:
        # "no events at all" counterfactual — also strip calendar proximity
        nxt = prv = 60
        adj_holiday = False
        long_weekend = 0

    f.update({
        "n_events": len(evs),
        "impact_sum": float(sum(impacts)) if impacts else 0.0,
        "impact_max": float(max(impacts)) if impacts else 0.0,
        "impact_min": float(min(impacts)) if impacts else 0.0,
        "max_event_span": float(max(spans)) if spans else 0.0,
        "is_multiday_event": int(bool(spans) and max(spans) > 1),
        "days_to_next_festival": nxt,
        "days_since_last_festival": prv,
        "is_festival_eve": int(nxt == 1),
        "is_festival_day_after": int(prv == 1),
        "n_pan_india_events": pan,
        "n_local_events": local,
        "is_long_weekend": long_weekend,
        "is_holiday_adjacent": int(adj_holiday),
    })
    return f


def weather_features(temp_max=None, temp_min=None, precip_mm=None,
                     weather_code=None) -> dict:
    def _f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        return np.nan if math.isnan(v) else v

    tmax, tmin, pr = _f(temp_max), _f(temp_min), _f(precip_mm)
    _c = _f(weather_code)
    code = -1 if np.isnan(_c) else int(_c)
    has = int(not (np.isnan(tmax) and np.isnan(pr)))
    return {
        "temp_max": tmax,
        "temp_min": tmin,
        "temp_range": (tmax - tmin) if not (np.isnan(tmax) or np.isnan(tmin)) else np.nan,
        "precip_mm": pr,
        "log_precip": math.log1p(pr) if not np.isnan(pr) else np.nan,
        "weather_code": code,
        "is_rain": int(code in (51, 53, 55, 61, 63, 65, 80, 81, 82)) if code >= 0 else 0,
        "is_heavy_rain": int((not np.isnan(pr) and pr >= 20) or code in (65, 82)),
        "is_thunder": int(code in (95, 96, 99)),
        "is_fog": int(code in (45, 48)),
        "is_extreme_heat": int(not np.isnan(tmax) and tmax >= 44),
        "is_high_heat": int(not np.isnan(tmax) and 40 <= tmax < 44),
        "is_cool": int(not np.isnan(tmax) and tmax <= 15),
        "is_pleasant": int(not np.isnan(tmax) and 18 <= tmax <= 30 and (np.isnan(pr) or pr < 1)),
        "has_weather_data": has,
    }


def build_row(city: str, d: date, rest_type: str,
              temp_max=None, temp_min=None, precip_mm=None,
              weather_code=None, evs=None, proximity: bool = True) -> dict:
    """One fully-featured row, ready for `rows_to_frame`.

    `evs` / `proximity` exist for counterfactual scoring (see ml/predict.py)."""
    geo = CITIES.get(city, {})
    row = {
        "restaurant_type": rest_type,
        "city": city,
        "state": geo.get("state", "Unknown"),
        "zone": geo.get("zone", "Unknown"),
    }
    row.update(calendar_features(city, d))
    row.update(event_features(city, d, evs=evs, proximity=proximity))
    row.update(weather_features(temp_max, temp_min, precip_mm, weather_code))
    return row


# ── Frame assembly with stable categorical dtypes ─────────────────────────────
_CAT_LEVELS = {
    "restaurant_type": RESTAURANT_TYPES,
    "city": CITY_LIST + ["Unknown"],
    "state": STATE_LIST + ["Unknown"],
    "zone": ZONE_LIST + ["Unknown"],
}


def rows_to_frame(rows) -> pd.DataFrame:
    """Build a model-ready DataFrame with columns in a fixed, known order."""
    df = pd.DataFrame(list(rows))
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    df = df[FEATURE_COLUMNS].copy()
    for col, levels in _CAT_LEVELS.items():
        df[col] = pd.Categorical(
            df[col].where(df[col].isin(levels), "Unknown"), categories=levels
        )
    for c in FEATURE_COLUMNS:
        if c not in _CAT_LEVELS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df
