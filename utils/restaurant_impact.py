"""
restaurant_impact.py
====================
Order-volume / AOV impact engine for 6 restaurant types.

ENGINE SELECTION
----------------
`predict_impact()` and `predict_date_range()` now route to the trained
LightGBM model in `ml/` whenever a model file exists, and fall back to the
legacy hand-written multiplier tables below when it does not. The returned
dict has the same keys either way, so every caller (main.py, dashboard.py,
pages/01_Restaurant_Impact.py) works unchanged.

  * force the ML engine   → DOTPOOL_ENGINE=ml     (errors instead of falling back)
  * force the rule engine → DOTPOOL_ENGINE=rules
  * default               → ML if trained, else rules

The rule-based multipliers below are retained deliberately: they are the
fallback, the A/B comparison baseline, and the documented prior that the ML
model was built to replace.

Restaurant types:
  QSR           — Quick Service (Mad Over Donuts, 99 Pancakes, McDonald's)
  Fine Dining   — Full Service, upscale table dining
  PBCL          — Pub, Bar, Café, Lounge (Barista, brewpubs)
  Casual Dining — Mid-range sit-down (Chili's, Barbeque Nation)
  Cloud Kitchen — Delivery-only (Rebel Foods, Biryani By Kilo, Faasos)
  Cafe          — Coffee shops, bakery-cafes (Starbucks, Blue Tokai, Third Wave)

Each restaurant type responds differently to the same event/weather.
"""

import os
from datetime import date, timedelta
from data.events_db import EVENTS_DB, _event_applies_to_geo

# ── Baseline orders (index, 100 = average weekday) ────────────────────────────
BASELINES = {
    "QSR":           {"weekday": 100, "saturday": 145, "sunday": 130},
    "Fine Dining":   {"weekday": 100, "saturday": 165, "sunday": 155},
    "PBCL":          {"weekday": 100, "saturday": 175, "sunday": 120},
    "Casual Dining": {"weekday": 100, "saturday": 155, "sunday": 145},
    "Cloud Kitchen": {"weekday": 100, "saturday": 130, "sunday": 125},
    "Cafe":          {"weekday": 100, "saturday": 140, "sunday": 135},
}

ALL_RT = list(BASELINES.keys())

# ── Baseline average order value / ticket size (₹, normal weekday) ────────────
BASELINE_AOV = {
    "QSR":           330,
    "Fine Dining":   2100,
    "PBCL":          1350,
    "Casual Dining": 950,
    "Cloud Kitchen": 480,
    "Cafe":          420,
}

# ── Event multipliers per restaurant type ─────────────────────────────────────
EVENT_MULTIPLIERS = {
    "Festival": {
        "Hindu": {
            "QSR":           1.25,
            "Fine Dining":   1.45,
            "PBCL":          1.10,
            "Casual Dining": 1.35,
            "Cloud Kitchen": 1.20,
            "Cafe":          1.15,
        },
        "Muslim": {
            "QSR":           1.15,
            "Fine Dining":   1.40,
            "PBCL":          0.75,
            "Casual Dining": 1.30,
            "Cloud Kitchen": 1.25,
            "Cafe":          1.05,
        },
        "Regional": {
            "QSR":           1.20,
            "Fine Dining":   1.50,
            "PBCL":          1.05,
            "Casual Dining": 1.38,
            "Cloud Kitchen": 1.18,
            "Cafe":          1.12,
        },
        "*": {
            "QSR":           1.20,
            "Fine Dining":   1.42,
            "PBCL":          1.05,
            "Casual Dining": 1.35,
            "Cloud Kitchen": 1.18,
            "Cafe":          1.12,
        },
    },
    "Government Holiday": {
        "*": {
            "QSR":           1.10,
            "Fine Dining":   1.30,
            "PBCL":          1.35,
            "Casual Dining": 1.25,
            "Cloud Kitchen": 1.15,
            "Cafe":          1.20,
        },
    },
    "Sports Event": {
        "Cricket": {
            "QSR":           1.18,
            "Fine Dining":   0.80,
            "PBCL":          1.55,
            "Casual Dining": 0.90,
            "Cloud Kitchen": 1.45,  # big delivery spike during IPL/WC
            "Cafe":          1.10,
        },
        "Football": {
            "QSR":           1.12,
            "Fine Dining":   0.85,
            "PBCL":          1.40,
            "Casual Dining": 0.92,
            "Cloud Kitchen": 1.35,
            "Cafe":          1.08,
        },
        # Multi-month LEAGUE SEASONS (ISL, Pro Kabaddi) are date-ranged as
        # "start of season" -> "end of season" in the events calendar, which
        # is months long — but a real demand spike only happens on actual
        # match nights, a small fraction of those days. Applying the same
        # multiplier as a short, nightly-viewing tournament (Football/Cricket
        # above, both a few weeks long) to every single day of a 6-month
        # season was overstating PBCL/Cloud Kitchen demand badly (this is
        # what caused Navratri/Durga Puja to show an inflated PBCL lift —
        # ISL Season 2026-27 and Pro Kabaddi League 2026-27 both happen to
        # overlap that October window). These use a season-average effect
        # instead of a match-night spike.
        "Football (Season)": {
            "QSR":           1.04, "Fine Dining":   0.99,
            "PBCL":          1.06, "Casual Dining": 0.99,
            "Cloud Kitchen": 1.07, "Cafe":          1.01,
        },
        "Kabaddi (Season)": {
            "QSR":           1.03, "Fine Dining":   1.00,
            "PBCL":          1.03, "Casual Dining": 1.00,
            "Cloud Kitchen": 1.05, "Cafe":          1.01,
        },
        "*": {
            "QSR":           1.15,
            "Fine Dining":   0.85,
            "PBCL":          1.45,
            "Casual Dining": 0.92,
            "Cloud Kitchen": 1.38,
            "Cafe":          1.08,
        },
    },
    "Commercial Event": {
        "*": {
            "QSR":           1.08,
            "Fine Dining":   1.15,
            "PBCL":          1.05,
            "Casual Dining": 1.12,
            "Cloud Kitchen": 1.05,
            "Cafe":          1.10,
        },
    },
    "Public Event": {
        "*": {
            "QSR":           1.15,
            "Fine Dining":   1.05,
            "PBCL":          1.10,
            "Casual Dining": 1.08,
            "Cloud Kitchen": 1.10,
            "Cafe":          1.12,
        },
    },
    "Government Order": {
        # Pure paperwork/licensing compliance — negligible demand effect
        "Food Safety": {
            "QSR": 1.00, "Fine Dining": 1.00, "PBCL": 1.00,
            "Casual Dining": 1.00, "Cloud Kitchen": 1.00, "Cafe": 1.00,
        },
        "Digital Mandate": {
            "QSR": 1.05, "Fine Dining": 1.01, "PBCL": 1.00,
            "Casual Dining": 1.02, "Cloud Kitchen": 1.06, "Cafe": 1.02,
        },
        "Environmental": {
            "QSR": 0.97, "Fine Dining": 0.99, "PBCL": 0.99,
            "Casual Dining": 0.98, "Cloud Kitchen": 0.96, "Cafe": 0.98,
        },
        "Traffic Regulation": {
            "QSR": 0.97, "Fine Dining": 0.85, "PBCL": 0.88,
            "Casual Dining": 0.87, "Cloud Kitchen": 1.02, "Cafe": 0.92,
        },
        "Liquor Ban": {
            "QSR": 0.98, "Fine Dining": 0.95, "PBCL": 0.35,
            "Casual Dining": 0.92, "Cloud Kitchen": 1.00, "Cafe": 1.00,
        },
        "Operating Hours": {
            "QSR": 1.05, "Fine Dining": 1.03, "PBCL": 1.12,
            "Casual Dining": 1.04, "Cloud Kitchen": 1.00, "Cafe": 1.02,
        },
        "Labour Regulation": {
            "QSR": 0.98, "Fine Dining": 1.00, "PBCL": 1.00,
            "Casual Dining": 0.99, "Cloud Kitchen": 0.97, "Cafe": 1.00,
        },
        # Fallback for severe/unlisted restrictive orders (e.g. lockdown-style)
        "*": {
            "QSR":           0.90,
            "Fine Dining":   0.75,
            "PBCL":          0.70,
            "Casual Dining": 0.80,
            "Cloud Kitchen": 0.95,
            "Cafe":          0.85,
        },
    },
    "Emergency Crisis": {
        "Heatwave": {
            "QSR":           0.75,
            "Fine Dining":   0.60,
            "PBCL":          0.65,
            "Casual Dining": 0.65,
            "Cloud Kitchen": 0.85,  # delivery still works
            "Cafe":          0.70,
        },
        "Flood": {
            "QSR":           0.40,
            "Fine Dining":   0.25,
            "PBCL":          0.20,
            "Casual Dining": 0.30,
            "Cloud Kitchen": 0.50,
            "Cafe":          0.35,
        },
        "Cyclone": {
            "QSR":           0.35,
            "Fine Dining":   0.20,
            "PBCL":          0.15,
            "Casual Dining": 0.25,
            "Cloud Kitchen": 0.40,
            "Cafe":          0.30,
        },
        "Air Quality": {
            "QSR":           0.78,
            "Fine Dining":   0.55,
            "PBCL":          0.50,
            "Casual Dining": 0.62,
            "Cloud Kitchen": 0.90,
            "Cafe":          0.65,
        },
        "Severe Weather": {
            "QSR":           0.70,
            "Fine Dining":   0.50,
            "PBCL":          0.45,
            "Casual Dining": 0.58,
            "Cloud Kitchen": 0.80,
            "Cafe":          0.60,
        },
        "*": {
            "QSR":           0.65,
            "Fine Dining":   0.50,
            "PBCL":          0.45,
            "Casual Dining": 0.55,
            "Cloud Kitchen": 0.78,
            "Cafe":          0.58,
        },
    },
}

# ── Weather multipliers ────────────────────────────────────────────────────────
def weather_multiplier(rest_type: str, temp_max: float, precip_mm: float,
                       weather_code: int) -> tuple:
    """
    Returns (multiplier, reason_str).
    Cloud Kitchen benefits from rain (delivery up).
    Fine Dining / Casual Dining / PBCL hurt by rain (dine-in drops).
    QSR and Cafe are partially insulated (delivery offsets footfall loss).
    """
    mult = 1.0; reason = []

    # ── Temperature ──────────────────────────────────────────────────────────
    if temp_max is not None:
        if temp_max >= 44:
            m = {"QSR":0.72,"Fine Dining":0.55,"PBCL":0.60,
                 "Casual Dining":0.62,"Cloud Kitchen":0.82,"Cafe":0.68}[rest_type]
            mult *= m; reason.append(f"Extreme heat {temp_max:.0f}°C — severe footfall suppression")
        elif temp_max >= 40:
            m = {"QSR":0.85,"Fine Dining":0.72,"PBCL":0.78,
                 "Casual Dining":0.76,"Cloud Kitchen":0.90,"Cafe":0.82}[rest_type]
            mult *= m; reason.append(f"High heat {temp_max:.0f}°C — outdoor footfall drops")
        elif temp_max <= 15:
            m = {"QSR":1.05,"Fine Dining":1.15,"PBCL":1.10,
                 "Casual Dining":1.12,"Cloud Kitchen":1.08,"Cafe":1.20}[rest_type]
            mult *= m; reason.append(f"Cool weather {temp_max:.0f}°C — cosy dining boost")

    # ── Precipitation ─────────────────────────────────────────────────────────
    if precip_mm is not None and precip_mm > 0:
        if precip_mm >= 20:
            m = {"QSR":1.18,"Fine Dining":0.45,"PBCL":0.48,
                 "Casual Dining":0.52,"Cloud Kitchen":1.55,"Cafe":0.68}[rest_type]
            mult *= m
            reason.append(f"Heavy rain {precip_mm:.0f}mm — Cloud Kitchen↑↑ delivery↑ dine-in↓↓")
        elif precip_mm >= 8:
            m = {"QSR":1.12,"Fine Dining":0.60,"PBCL":0.62,
                 "Casual Dining":0.65,"Cloud Kitchen":1.40,"Cafe":0.78}[rest_type]
            mult *= m
            reason.append(f"Moderate rain {precip_mm:.0f}mm — Cloud Kitchen↑ delivery↑ dine-in↓")
        elif precip_mm >= 3:
            m = {"QSR":1.06,"Fine Dining":0.80,"PBCL":0.78,
                 "Casual Dining":0.82,"Cloud Kitchen":1.22,"Cafe":0.88}[rest_type]
            mult *= m
            reason.append(f"Light rain/drizzle {precip_mm:.0f}mm — slight delivery lift, dine-in dip")

    # ── Severe weather codes ──────────────────────────────────────────────────
    if weather_code in (95, 96, 99):
        m = {"QSR":1.15,"Fine Dining":0.40,"PBCL":0.38,
             "Casual Dining":0.45,"Cloud Kitchen":1.50,"Cafe":0.55}[rest_type]
        mult *= m
        reason.append("Thunderstorm — delivery spike, dine-in/PBCL severe crash")

    return round(mult, 3), "; ".join(reason) if reason else "Normal weather — no significant weather impact"


def aov_event_multiplier(order_event_mult: float) -> float:
    """
    Events that push order volume also nudge average order value (AOV) —
    but by a smaller margin (group/combo orders, festive premium items).
    Dampened at 35% of the order-volume swing, floored so AOV never
    collapses as hard as raw order count during a crisis.
    """
    delta = (order_event_mult - 1.0) * 0.35
    return round(max(0.55, 1.0 + delta), 3)


def aov_weather_multiplier(weather_mult: float) -> float:
    """
    Weather-driven channel shift (dine-in → delivery) tends to lift AOV
    slightly (delivery minimum-cart-value + combo bundling), dampened
    at 25% of the weather order-volume swing.
    """
    delta = (weather_mult - 1.0) * 0.25
    return round(max(0.6, 1.0 + delta), 3)


def get_base(rest_type: str, weekday: int) -> float:
    b = BASELINES[rest_type]
    if weekday == 5: return b["saturday"]
    if weekday == 6: return b["sunday"]
    return b["weekday"]


def predict_impact_rules(city, pred_date, rest_type,
                   temp_max=None, precip_mm=None, weather_code=None):
    base = get_base(rest_type, pred_date.weekday())
    evts = []
    from datetime import date as dt_date
    d = dt_date.fromisoformat(pred_date.isoformat())
    for ev in EVENTS_DB:
        ev_s = dt_date.fromisoformat(ev["start_date"])
        ev_e = dt_date.fromisoformat(ev["end_date"])
        if not (ev_s <= d <= ev_e): continue
        if not _event_applies_to_geo(ev, city=city): continue
        evts.append(ev)

    event_mult = 1.0; event_factors = []
    for ev in evts:
        cat = ev["category"]; subcat = ev.get("subcategory","*")
        cat_m = EVENT_MULTIPLIERS.get(cat,{})
        sub_m = cat_m.get(subcat, cat_m.get("*",{}))
        m = sub_m.get(rest_type, 1.0)
        event_mult *= m
        direction = "↑" if m>1.0 else ("↓" if m<1.0 else "→")
        event_factors.append(f"{ev['name']} ({cat}) {direction}{round((m-1)*100):+.0f}%")

    w_mult, w_reason = weather_multiplier(rest_type, temp_max, precip_mm, weather_code or 0)
    total_mult  = event_mult * w_mult
    predicted   = round(base * total_mult, 1)
    pct_change  = round((total_mult - 1) * 100, 1)

    # ── Price / AOV impact (separate from order-volume impact) ───────────────
    aov_mult      = round(aov_event_multiplier(event_mult) * aov_weather_multiplier(w_mult), 3)
    aov_base      = BASELINE_AOV[rest_type]
    predicted_aov = round(aov_base * aov_mult)
    revenue_total_mult   = round(total_mult * aov_mult, 3)
    predicted_revenue_index = round(base * revenue_total_mult, 1)
    revenue_pct_change      = round((revenue_total_mult - 1) * 100, 1)

    if   pct_change >= 30:   signal = "very_high_up"
    elif pct_change >= 15:   signal = "high_up"
    elif pct_change >= 5:    signal = "slight_up"
    elif pct_change <= -35:  signal = "very_low"
    elif pct_change <= -20:  signal = "low"
    elif pct_change <= -8:   signal = "slight_down"
    else:                    signal = "neutral"

    if   len(evts)==0 and w_mult==1.0: confidence="High"
    elif abs(pct_change)>40:           confidence="Low"
    elif abs(pct_change)>20:           confidence="Medium"
    else:                              confidence="High"

    all_factors = event_factors + ([w_reason] if w_reason!="Normal weather" else [])
    return {
        "date": pred_date.isoformat(), "city": city,
        "restaurant_type": rest_type,
        "weekday": pred_date.strftime("%A"),
        "base_index": base, "event_multiplier": round(event_mult,3),
        "weather_multiplier": w_mult, "total_multiplier": round(total_mult,3),
        "predicted_index": predicted, "pct_change": pct_change,
        "signal": signal, "confidence": confidence,
        "active_events": [e["name"] for e in evts],
        "factors": all_factors, "weather_note": w_reason,
        "aov_base": aov_base, "aov_multiplier": aov_mult,
        "predicted_aov": predicted_aov,
        "predicted_revenue_index": predicted_revenue_index,
        "revenue_pct_change": revenue_pct_change,
    }


def predict_date_range_rules(city, start_dt, end_dt, temp_data=None):
    results = {rt: [] for rt in ALL_RT}
    cur = start_dt
    while cur <= end_dt:
        d_str = cur.isoformat()
        td    = (temp_data or {}).get(d_str, {})
        for rt in ALL_RT:
            results[rt].append(predict_impact_rules(
                city, cur, rt,
                td.get("temp_max_c"), td.get("precipitation_mm"), td.get("weather_code")
            ))
        cur += timedelta(days=1)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE ROUTER — ML first, rules as fallback
# ══════════════════════════════════════════════════════════════════════════════
def _engine_pref() -> str:
    # Default is "rules", not "auto" — the trained ML model was found to
    # significantly overstate PBCL demand in some slices (e.g. October,
    # independent of any specific festival — Shapley attribution shows the
    # named events explain only ~1% of a ~31% total lift, meaning ~30 points
    # come from an unexplained seasonal/baseline drift in the trained model,
    # not from Navratri or any other labeled factor). The hand-audited rules
    # table doesn't have that problem and matches known real-world behavior
    # (e.g. PBCL barely moving during Navratri, a widely-observed fasting
    # period). Set DOTPOOL_ENGINE=ml explicitly to opt back into the trained
    # model once it's been retrained/validated against this, or =auto to
    # restore the old "prefer ML whenever available" behavior.
    return os.environ.get("DOTPOOL_ENGINE", "rules").lower()


def ml_ready() -> bool:
    """True when a trained model is on disk and loadable."""
    if _engine_pref() == "rules":
        return False
    try:
        from ml.predict import model_available
        return model_available()
    except Exception:
        return False


def engine_status() -> dict:
    """What the app is actually running on right now — handy for the UI."""
    pref = _engine_pref()
    try:
        from ml.predict import model_info
        info = model_info()
    except Exception as exc:                              # noqa: BLE001
        info = {"available": False, "error": str(exc)}
    active = "ml" if (pref != "rules" and info.get("available")) else "rules"
    return {"active_engine": active, "preference": pref, "model": info}


def predict_impact(city, pred_date, rest_type,
                   temp_max=None, precip_mm=None, weather_code=None,
                   temp_min=None, explain=True, drivers=False):
    """Demand forecast for one city × date × restaurant type.

    Uses the trained LightGBM model when available; otherwise the legacy
    multiplier tables. Output keys are identical for both engines, plus a few
    ML-only extras (`prediction_interval_80`, `top_drivers`, …) when ML runs.
    """
    pref = _engine_pref()
    if pref != "rules":
        try:
            from ml.predict import predict_impact_ml
            return predict_impact_ml(city, pred_date, rest_type,
                                     temp_max=temp_max, precip_mm=precip_mm,
                                     weather_code=weather_code, temp_min=temp_min,
                                     explain=explain, drivers=drivers)
        except Exception as exc:                          # noqa: BLE001
            if pref == "ml":
                raise
            if not getattr(predict_impact, "_warned", False):
                print(f"[demandpulse] ML engine unavailable ({exc}); "
                      f"using rule-based fallback. Train it with `python -m ml.train`.")
                predict_impact._warned = True
    out = predict_impact_rules(city, pred_date, rest_type,
                               temp_max, precip_mm, weather_code)
    out["engine"] = "rules"
    return out


def predict_date_range(city, start_dt, end_dt, temp_data=None):
    """Same as predict_impact over a date range, for all restaurant types."""
    pref = _engine_pref()
    if pref != "rules":
        try:
            from ml.predict import predict_date_range_ml
            return predict_date_range_ml(city, start_dt, end_dt, temp_data,
                                         rest_types=ALL_RT)
        except Exception as exc:                          # noqa: BLE001
            if pref == "ml":
                raise
            print(f"[demandpulse] ML engine unavailable ({exc}); using rules.")
    res = predict_date_range_rules(city, start_dt, end_dt, temp_data)
    for rt in res:
        for r in res[rt]:
            r["engine"] = "rules"
    return res


def compare_engines(city, pred_date, rest_type,
                    temp_max=None, precip_mm=None, weather_code=None) -> dict:
    """Side-by-side ML vs rules for the same day — useful for validation."""
    rules = predict_impact_rules(city, pred_date, rest_type,
                                 temp_max, precip_mm, weather_code)
    try:
        from ml.predict import predict_impact_ml
        ml = predict_impact_ml(city, pred_date, rest_type,
                               temp_max=temp_max, precip_mm=precip_mm,
                               weather_code=weather_code)
    except Exception as exc:                              # noqa: BLE001
        return {"rules": rules, "ml": None, "error": str(exc)}
    return {
        "rules": rules,
        "ml": ml,
        "delta_pct_points": round(ml["pct_change"] - rules["pct_change"], 1),
    }
