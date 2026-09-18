"""
ml/simulate.py
==============
Bootstrap training data generator.

WHY THIS EXISTS
---------------
A supervised demand model needs a history of (day, city, restaurant type) →
(orders, AOV). Until real POS/aggregator history is available, this module
synthesises a defensible stand-in: a latent data-generating process with
city-level heterogeneity, seasonality, payday cycles, a growth trend,
non-linear weather response, event effects with build-up/decay, and
multiplicative noise — then the model has to *learn* that structure back from
raw features, exactly as it will have to learn it from real data.

IMPORTANT / HONEST CAVEAT
-------------------------
A model trained on this data reproduces the assumptions encoded here. It is a
working pipeline and a sane prior — not evidence about real Indian restaurant
demand. Replace `data/training/real_history.csv` with true history and rerun
`python -m ml.train` to get a model that means something empirically.
See ml/README.md for the required CSV schema.

Weather
-------
By default weather is simulated from a per-city climatology (monsoon window,
latitude-driven temperature curve, heat waves, thunderstorms, winter fog) so
training works offline. Pass `--real-weather` to `ml/train.py` to pull actual
Open-Meteo archive data for the sampled cities instead.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.india_geo import CITIES, CITY_LIST
from ml.config import (
    BASELINE_AOV,
    RANDOM_SEED,
    RESTAURANT_TYPES,
    SYNTH_CITY_SAMPLE,
    TRAIN_END,
    TRAIN_START,
    city_tier,
)
from ml.features import active_events

# ══════════════════════════════════════════════════════════════════════════════
# 1. LATENT TRUTH PARAMETERS
#    These are the "real world" the model must rediscover. They are deliberately
#    richer and different from the legacy hand-written multiplier table.
# ══════════════════════════════════════════════════════════════════════════════

# Day-of-week log-effects per restaurant type (Mon..Sun)
DOW_EFFECT = {
    "QSR":           [0.00, -0.02, 0.00, 0.03, 0.14, 0.38, 0.27],
    "Fine Dining":   [-0.06, -0.05, -0.02, 0.05, 0.26, 0.52, 0.44],
    "PBCL":          [-0.12, -0.10, 0.02, 0.10, 0.34, 0.57, 0.18],
    "Casual Dining": [-0.04, -0.03, 0.00, 0.05, 0.22, 0.45, 0.37],
    "Cloud Kitchen": [0.02, 0.00, 0.01, 0.04, 0.12, 0.27, 0.23],
    "Cafe":          [0.00, -0.01, 0.01, 0.04, 0.13, 0.34, 0.30],
}

TIER_EFFECT = {1: 0.30, 2: 0.10, 3: -0.12}          # log orders
TIER_AOV_EFFECT = {1: 0.16, 2: 0.04, 3: -0.10}      # log AOV

# Annual seasonality amplitude / phase per type (peak around Oct–Dec wedding +
# festive season for dine-in; flatter for delivery)
SEASON = {
    "QSR":           (0.07, 300),
    "Fine Dining":   (0.15, 320),
    "PBCL":          (0.13, 350),
    "Casual Dining": (0.11, 315),
    "Cloud Kitchen": (0.05, 200),
    "Cafe":          (0.09, 340),
}

# Annual compound growth in orders (delivery-led categories grow faster)
GROWTH = {
    "QSR": 0.09, "Fine Dining": 0.04, "PBCL": 0.05,
    "Casual Dining": 0.05, "Cloud Kitchen": 0.18, "Cafe": 0.12,
}

# Latent event log-effects: (category, subcategory) → per-type effect.
# "*" is the fallback subcategory.
EVENT_EFFECT = {
    ("Festival", "Hindu"):        {"QSR": 0.20, "Fine Dining": 0.34, "PBCL": 0.04, "Casual Dining": 0.28, "Cloud Kitchen": 0.19, "Cafe": 0.12},
    ("Festival", "Muslim"):       {"QSR": 0.16, "Fine Dining": 0.31, "PBCL": -0.32, "Casual Dining": 0.25, "Cloud Kitchen": 0.24, "Cafe": 0.05},
    ("Festival", "Christian"):    {"QSR": 0.14, "Fine Dining": 0.38, "PBCL": 0.30, "Casual Dining": 0.27, "Cloud Kitchen": 0.13, "Cafe": 0.22},
    ("Festival", "Sikh"):         {"QSR": 0.12, "Fine Dining": 0.18, "PBCL": -0.10, "Casual Dining": 0.17, "Cloud Kitchen": 0.12, "Cafe": 0.07},
    ("Festival", "Jain"):         {"QSR": -0.08, "Fine Dining": -0.05, "PBCL": -0.12, "Casual Dining": -0.04, "Cloud Kitchen": 0.02, "Cafe": 0.01},
    ("Festival", "Buddhist"):     {"QSR": 0.05, "Fine Dining": 0.08, "PBCL": -0.02, "Casual Dining": 0.07, "Cloud Kitchen": 0.05, "Cafe": 0.04},
    ("Festival", "Regional"):     {"QSR": 0.18, "Fine Dining": 0.36, "PBCL": 0.02, "Casual Dining": 0.30, "Cloud Kitchen": 0.16, "Cafe": 0.10},
    ("Festival", "*"):            {"QSR": 0.17, "Fine Dining": 0.32, "PBCL": 0.03, "Casual Dining": 0.27, "Cloud Kitchen": 0.16, "Cafe": 0.10},

    ("Government Holiday", "*"):  {"QSR": 0.10, "Fine Dining": 0.26, "PBCL": 0.29, "Casual Dining": 0.22, "Cloud Kitchen": 0.14, "Cafe": 0.18},

    ("Sports Event", "Cricket"):  {"QSR": 0.17, "Fine Dining": -0.24, "PBCL": 0.43, "Casual Dining": -0.11, "Cloud Kitchen": 0.38, "Cafe": 0.09},
    ("Sports Event", "Football"): {"QSR": 0.11, "Fine Dining": -0.17, "PBCL": 0.34, "Casual Dining": -0.09, "Cloud Kitchen": 0.30, "Cafe": 0.07},
    ("Sports Event", "*"):        {"QSR": 0.13, "Fine Dining": -0.16, "PBCL": 0.36, "Casual Dining": -0.08, "Cloud Kitchen": 0.31, "Cafe": 0.07},

    ("Commercial Event", "E-commerce Sale"): {"QSR": 0.09, "Fine Dining": 0.05, "PBCL": 0.02, "Casual Dining": 0.07, "Cloud Kitchen": 0.14, "Cafe": 0.06},
    ("Commercial Event", "*"):    {"QSR": 0.07, "Fine Dining": 0.13, "PBCL": 0.05, "Casual Dining": 0.11, "Cloud Kitchen": 0.05, "Cafe": 0.09},

    ("Public Event", "*"):        {"QSR": 0.14, "Fine Dining": 0.04, "PBCL": 0.09, "Casual Dining": 0.07, "Cloud Kitchen": 0.09, "Cafe": 0.11},
    ("Public Event", "Political"):{"QSR": 0.05, "Fine Dining": -0.10, "PBCL": -0.18, "Casual Dining": -0.05, "Cloud Kitchen": 0.06, "Cafe": -0.02},

    ("Government Order", "Liquor Ban"):        {"QSR": -0.02, "Fine Dining": -0.06, "PBCL": -1.02, "Casual Dining": -0.09, "Cloud Kitchen": 0.01, "Cafe": 0.00},
    ("Government Order", "Traffic Regulation"):{"QSR": -0.04, "Fine Dining": -0.17, "PBCL": -0.14, "Casual Dining": -0.14, "Cloud Kitchen": 0.03, "Cafe": -0.09},
    ("Government Order", "Operating Hours"):   {"QSR": 0.04, "Fine Dining": 0.03, "PBCL": 0.12, "Casual Dining": 0.04, "Cloud Kitchen": 0.00, "Cafe": 0.02},
    ("Government Order", "Environmental"):     {"QSR": -0.03, "Fine Dining": -0.01, "PBCL": -0.01, "Casual Dining": -0.02, "Cloud Kitchen": -0.05, "Cafe": -0.02},
    ("Government Order", "Digital Mandate"):   {"QSR": 0.04, "Fine Dining": 0.01, "PBCL": 0.00, "Casual Dining": 0.02, "Cloud Kitchen": 0.06, "Cafe": 0.02},
    ("Government Order", "Food Safety"):       {"QSR": 0.00, "Fine Dining": 0.00, "PBCL": 0.00, "Casual Dining": 0.00, "Cloud Kitchen": -0.01, "Cafe": 0.00},
    ("Government Order", "Labour Regulation"): {"QSR": -0.02, "Fine Dining": 0.00, "PBCL": 0.00, "Casual Dining": -0.01, "Cloud Kitchen": -0.03, "Cafe": 0.00},
    ("Government Order", "*"):                 {"QSR": -0.10, "Fine Dining": -0.28, "PBCL": -0.35, "Casual Dining": -0.22, "Cloud Kitchen": -0.04, "Cafe": -0.16},

    ("Emergency Crisis", "Heatwave"):       {"QSR": -0.28, "Fine Dining": -0.50, "PBCL": -0.42, "Casual Dining": -0.42, "Cloud Kitchen": -0.14, "Cafe": -0.35},
    ("Emergency Crisis", "Flood"):          {"QSR": -0.92, "Fine Dining": -1.38, "PBCL": -1.55, "Casual Dining": -1.20, "Cloud Kitchen": -0.68, "Cafe": -1.05},
    ("Emergency Crisis", "Cyclone"):        {"QSR": -1.05, "Fine Dining": -1.60, "PBCL": -1.85, "Casual Dining": -1.38, "Cloud Kitchen": -0.90, "Cafe": -1.20},
    ("Emergency Crisis", "Air Quality"):    {"QSR": -0.24, "Fine Dining": -0.58, "PBCL": -0.66, "Casual Dining": -0.47, "Cloud Kitchen": -0.09, "Cafe": -0.42},
    ("Emergency Crisis", "Natural Disaster"):{"QSR": -0.60, "Fine Dining": -0.95, "PBCL": -1.05, "Casual Dining": -0.85, "Cloud Kitchen": -0.40, "Cafe": -0.70},
    ("Emergency Crisis", "*"):              {"QSR": -0.42, "Fine Dining": -0.68, "PBCL": -0.78, "Casual Dining": -0.58, "Cloud Kitchen": -0.24, "Cafe": -0.52},

    ("Weather Event", "Monsoon"):        {"QSR": 0.03, "Fine Dining": -0.09, "PBCL": -0.07, "Casual Dining": -0.07, "Cloud Kitchen": 0.16, "Cafe": -0.03},
    ("Weather Event", "Summer Heatwave"):{"QSR": -0.14, "Fine Dining": -0.26, "PBCL": -0.20, "Casual Dining": -0.22, "Cloud Kitchen": -0.05, "Cafe": -0.18},
    ("Weather Event", "Winter"):         {"QSR": 0.05, "Fine Dining": 0.14, "PBCL": 0.11, "Casual Dining": 0.12, "Cloud Kitchen": 0.07, "Cafe": 0.19},
    ("Weather Event", "*"):              {"QSR": 0.00, "Fine Dining": -0.04, "PBCL": -0.03, "Casual Dining": -0.03, "Cloud Kitchen": 0.05, "Cafe": -0.01},
}

# AOV responds to the same events but with damped, partly different signs.
AOV_EVENT_DAMPING = {
    "QSR": 0.30, "Fine Dining": 0.45, "PBCL": 0.40,
    "Casual Dining": 0.38, "Cloud Kitchen": 0.28, "Cafe": 0.30,
}

# Weather sensitivity (per restaurant type) for the latent process
RAIN_SENS = {
    "QSR": 0.09, "Fine Dining": -0.42, "PBCL": -0.40,
    "Casual Dining": -0.34, "Cloud Kitchen": 0.34, "Cafe": -0.22,
}
HEAT_SENS = {
    "QSR": -0.030, "Fine Dining": -0.055, "PBCL": -0.045,
    "Casual Dining": -0.048, "Cloud Kitchen": -0.016, "Cafe": -0.040,
}
COLD_SENS = {
    "QSR": 0.010, "Fine Dining": 0.026, "PBCL": 0.020,
    "Casual Dining": 0.022, "Cloud Kitchen": 0.014, "Cafe": 0.032,
}


def _event_effect(cat: str, sub: str, rt: str) -> float:
    tbl = EVENT_EFFECT.get((cat, sub)) or EVENT_EFFECT.get((cat, "*")) or {}
    return tbl.get(rt, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYNTHETIC CLIMATOLOGY
# ══════════════════════════════════════════════════════════════════════════════
MONSOON_WINDOW = {          # zone → (start doy, end doy, intensity)
    "South":     (152, 288, 1.15),
    "West":      (158, 273, 1.35),
    "East":      (152, 280, 1.25),
    "Northeast": (135, 285, 1.55),
    "Central":   (172, 268, 1.00),
    "North":     (180, 262, 0.85),
}


def simulate_weather(city: str, d: date, rng: np.random.Generator) -> dict:
    geo = CITIES.get(city, {})
    lat = geo.get("lat", 22.0)
    zone = geo.get("zone", "Central")
    doy = d.timetuple().tm_yday

    # Base annual temperature curve: hotter at lower latitudes, peak ~ mid-May
    annual_mean = 31.5 - 0.28 * (lat - 12.0)
    amplitude = 3.0 + 0.42 * (lat - 12.0)
    tmax = annual_mean + amplitude * math.sin(2 * math.pi * (doy - 105) / 365.25)

    m_start, m_end, m_int = MONSOON_WINDOW.get(zone, (165, 270, 1.0))
    in_monsoon = m_start <= doy <= m_end
    if in_monsoon:
        tmax -= 4.5 * m_int

    tmax += rng.normal(0, 2.1)

    # Pre-monsoon heat spikes (Apr–Jun) in North / Central / East plains
    if 90 <= doy <= 175 and zone in ("North", "Central", "East", "West"):
        if rng.random() < 0.09:
            tmax += rng.uniform(3.0, 7.5)

    # Precipitation
    if in_monsoon:
        p_rain = 0.52 * m_int
        scale = 11.0 * m_int
    elif zone == "South" and 288 < doy <= 350:      # NE monsoon (TN/Kerala)
        p_rain, scale = 0.34, 9.0
    else:
        p_rain, scale = 0.06, 3.0

    precip = 0.0
    if rng.random() < min(p_rain, 0.95):
        precip = float(rng.exponential(scale))

    # Weather code
    if precip >= 25 and rng.random() < 0.35:
        code = int(rng.choice([95, 96, 99], p=[0.8, 0.14, 0.06]))
    elif precip >= 20:
        code = 65
    elif precip >= 8:
        code = 63
    elif precip >= 2:
        code = 61
    elif precip > 0:
        code = 51
    elif zone == "North" and (doy <= 25 or doy >= 340) and rng.random() < 0.30:
        code = 45
    else:
        code = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.25, 0.10]))

    tmin = tmax - rng.uniform(6.5, 13.0) - (2.5 if in_monsoon else 0.0)
    return {
        "temp_max_c": round(float(tmax), 1),
        "temp_min_c": round(float(tmin), 1),
        "precipitation_mm": round(float(precip), 1),
        "weather_code": code,
    }


def fetch_real_weather(city: str, start: str, end: str) -> dict:
    """Optional: pull true Open-Meteo archive weather for a city."""
    from utils.weather_service import get_historical_weather

    geo = CITIES.get(city, {})
    rows = get_historical_weather(geo.get("lat"), geo.get("lon"), start, end)
    out = {}
    for r in rows:
        if "error" in r:
            return {}
        out[r["date"]] = {
            "temp_max_c": r.get("temp_max_c"),
            "temp_min_c": r.get("temp_min_c"),
            "precipitation_mm": r.get("precipitation_mm"),
            "weather_code": r.get("weather_code"),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. PANEL GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def pick_cities(n: int = SYNTH_CITY_SAMPLE, seed: int = RANDOM_SEED) -> list:
    """Representative national sample: all tier-1, then a spread across zones."""
    rng = np.random.default_rng(seed)
    t1 = [c for c in CITY_LIST if city_tier(c) == 1]
    rest = [c for c in CITY_LIST if c not in t1]
    by_zone = {}
    for c in rest:
        by_zone.setdefault(CITIES[c]["zone"], []).append(c)
    picked = list(t1)
    zones = sorted(by_zone)
    i = 0
    while len(picked) < n and any(by_zone.values()):
        z = zones[i % len(zones)]
        if by_zone[z]:
            picked.append(by_zone[z].pop(rng.integers(len(by_zone[z]))))
        i += 1
    return picked[:n]


def generate_panel(start: str = TRAIN_START, end: str = TRAIN_END,
                   cities=None, use_real_weather: bool = False,
                   seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = cities or pick_cities()
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)

    # Persistent per-city and per-city×type latent effects
    city_fx = {c: rng.normal(0, 0.13) for c in cities}
    ct_fx = {(c, rt): rng.normal(0, 0.08) for c in cities for rt in RESTAURANT_TYPES}
    city_aov_fx = {c: rng.normal(0, 0.09) for c in cities}

    real_weather = {}
    if use_real_weather:
        for c in cities:
            real_weather[c] = fetch_real_weather(c, start, min(end, date.today().isoformat()))

    records = []
    for c in cities:
        tier = city_tier(c)
        d = d0
        while d <= d1:
            key = d.isoformat()
            w = (real_weather.get(c, {}) or {}).get(key) or simulate_weather(c, d, rng)
            tmax = w["temp_max_c"]
            precip = w["precipitation_mm"] or 0.0
            code = w["weather_code"]
            evs = active_events(c, d)
            doy = d.timetuple().tm_yday
            years = (d - date(2023, 1, 1)).days / 365.25

            for rt in RESTAURANT_TYPES:
                log_o = math.log(100.0)
                log_o += DOW_EFFECT[rt][d.weekday()]
                log_o += TIER_EFFECT[tier] + city_fx[c] + ct_fx[(c, rt)]
                amp, phase = SEASON[rt]
                log_o += amp * math.sin(2 * math.pi * (doy - phase) / 365.25)
                log_o += math.log1p(GROWTH[rt]) * years
                # Payday cycle
                log_o += 0.055 if d.day <= 5 else (-0.035 if d.day >= 26 else 0.0)

                # ── Events (with build-up and decay around multi-day events) ──
                log_e = 0.0
                for ev in evs:
                    eff = _event_effect(ev["category"], ev.get("subcategory", "*"), rt)
                    span = (date.fromisoformat(ev["end_date"])
                            - date.fromisoformat(ev["start_date"])).days + 1
                    if span > 5:                      # long seasons dilute per-day
                        eff *= 0.45
                    elif span > 1:
                        eff *= 0.8
                    if ev.get("scope") != "pan_india":
                        eff *= 1.12                   # local events hit harder locally
                    log_e += eff
                # Diminishing returns when several events stack
                if log_e > 0:
                    log_e = math.log1p(log_e * 1.6) / 1.6
                log_o += log_e

                # ── Weather (non-linear, with interactions) ───────────────────
                rain_term = RAIN_SENS[rt] * math.log1p(precip) / math.log(21.0)
                if code in (95, 96, 99):
                    rain_term *= 1.45
                log_o += rain_term
                if tmax >= 38:
                    log_o += HEAT_SENS[rt] * (tmax - 38)
                if tmax <= 18:
                    log_o += COLD_SENS[rt] * (18 - tmax)
                # Weekend × good weather interaction
                if d.weekday() >= 5 and precip < 1 and 18 <= tmax <= 32:
                    log_o += 0.05

                orders = math.exp(log_o + rng.normal(0, 0.075))

                # ── AOV ───────────────────────────────────────────────────────
                log_a = math.log(BASELINE_AOV[rt])
                log_a += TIER_AOV_EFFECT[tier] + city_aov_fx[c]
                log_a += 0.055 * years                     # menu inflation
                log_a += AOV_EVENT_DAMPING[rt] * log_e
                log_a += 0.05 if d.weekday() >= 5 else 0.0
                if precip >= 8:                            # delivery cart bundling
                    log_a += 0.06 if rt in ("Cloud Kitchen", "QSR") else -0.03
                aov = math.exp(log_a + rng.normal(0, 0.05))

                records.append({
                    "date": key, "city": c, "restaurant_type": rt,
                    "orders_index": round(orders, 2),
                    "aov": round(aov, 1),
                    "temp_max_c": tmax, "temp_min_c": w["temp_min_c"],
                    "precipitation_mm": precip, "weather_code": code,
                })
            d += timedelta(days=1)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    from ml.config import SYNTHETIC_CSV

    df = generate_panel()
    df.to_csv(SYNTHETIC_CSV, index=False)
    print(f"wrote {len(df):,} rows → {SYNTHETIC_CSV}")
