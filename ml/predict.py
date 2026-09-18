"""
ml/predict.py
=============
Inference layer — the ML replacement for the hand-written multiplier engine.

`predict_impact_ml()` returns exactly the dict shape the API, the dashboard and
pages/01_Restaurant_Impact.py already consume (base_index, event_multiplier,
weather_multiplier, total_multiplier, predicted_index, pct_change, signal,
confidence, factors, AOV / revenue fields …), so it is a drop-in swap.

How the multipliers are produced now
------------------------------------
They are no longer constants. Each one is a **counterfactual** run of the same
trained model:

    D = model(day with no events, benign weather)   → baseline index
    C = model(day with events,    benign weather)   → event effect  = C / D
    B = model(day with no events, actual weather)   → weather effect = B / D
    A = model(day as it actually is)                → total effect  = A / D

so total ≈ event × weather × interaction, and every number traces back to a
learned relationship rather than a typed-in 1.55.

Per-event attribution uses leave-one-out: the model is re-scored with that one
event stripped out, and the difference is that event's contribution — which is
what fills the `factors` list the UI shows.

If the model files are missing (never trained, fresh clone), `is_ml=False` is
returned by `model_available()` and utils/restaurant_impact.py falls back to the
legacy rule engine, so the app never breaks.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta
from functools import lru_cache
from math import factorial

import numpy as np

from ml.config import (
    AOV_MODEL_PATH,
    QUANTILES,
    QUANTILE_PATHS,
    BASELINE_AOV,
    METRICS_PATH,
    META_PATH,
    ORDERS_MODEL_PATH,
)
from ml.features import active_events, build_row, rows_to_frame

# Benign reference weather used for the counterfactual baseline: a dry,
# pleasant, unremarkable day.
NEUTRAL_WEATHER = dict(temp_max=27.0, temp_min=19.0, precip_mm=0.0, weather_code=1)

# Up to this many active events, per-event credit is the exact Shapley value
# (2^n counterfactual rows). Beyond it, Shapley is estimated by sampling
# permutations — still exactly additive, just averaged over fewer orderings.
SHAPLEY_MAX_EVENTS = 5
SHAPLEY_PERMUTATIONS = 12        # single-day calls
SHAPLEY_PERMUTATIONS_BATCH = 3   # date-range calls (cost × days × types)


class _Engine:
    """Lazily-loaded singleton holding the two boosters + metadata."""

    def __init__(self):
        self.orders = None
        self.aov = None
        self.quantiles = {}
        self.meta = {}
        self.metrics = {}
        self.loaded = False
        self.error = None

    def load(self):
        if self.loaded or self.error:
            return
        try:
            import lightgbm as lgb

            if not os.path.exists(ORDERS_MODEL_PATH):
                raise FileNotFoundError(
                    f"{ORDERS_MODEL_PATH} not found — run `python -m ml.train`"
                )
            self.orders = lgb.Booster(model_file=ORDERS_MODEL_PATH)
            if os.path.exists(AOV_MODEL_PATH):
                self.aov = lgb.Booster(model_file=AOV_MODEL_PATH)
            for q, path in QUANTILE_PATHS.items():
                if os.path.exists(path):
                    self.quantiles[q] = lgb.Booster(model_file=path)
            if os.path.exists(META_PATH):
                self.meta = json.load(open(META_PATH))
            if os.path.exists(METRICS_PATH):
                self.metrics = json.load(open(METRICS_PATH))
            self.loaded = True
        except Exception as exc:                      # noqa: BLE001
            self.error = str(exc)

    # ── trend offsets (see ml/train.fit_trend) ────────────────────────────────
    def _offset(self, X, which: str) -> np.ndarray:
        tr = (self.meta or {}).get(f"trend_{which}") or {}
        if not tr:
            return np.zeros(len(X))
        g = tr.get("__global__", [0.0, 0.0])
        rt = X["restaurant_type"].astype(str).to_numpy()
        t = X["t_index"].to_numpy(dtype=float)
        slope = np.array([tr.get(r, g)[0] for r in rt])
        inter = np.array([tr.get(r, g)[1] for r in rt])
        return slope * t + inter

    def predict_orders(self, X) -> np.ndarray:
        return np.exp(self.orders.predict(X) + self._offset(X, "orders"))

    def predict_aov(self, X):
        if self.aov is None:
            return None
        return np.exp(self.aov.predict(X) + self._offset(X, "aov"))

    def predict_quantiles(self, X, rows=None):
        """{quantile: predicted index} for the requested rows, or None if untrained.

        The conformal margin from training is applied here, so the returned p10/p90
        really do bracket ~80% of outcomes rather than the ~63% the raw quantile
        models managed on held-out future dates.
        """
        if len(self.quantiles) < 2:
            return None
        Xr = X if rows is None else X.iloc[rows]
        off = self._offset(Xr, "orders")
        margin = float((self.meta or {}).get("cqr_margin") or 0.0)
        out = {}
        for q, booster in self.quantiles.items():
            adj = -margin if q <= 0.5 else (margin if q >= 0.9 else 0.0)
            if q == 0.5:
                adj = 0.0
            out[q] = np.exp(booster.predict(Xr) + off + adj)
        return out

    def contributions(self, X):
        """LightGBM native SHAP values (last column is the base value)."""
        return self.orders.predict(X, pred_contrib=True)


ENGINE = _Engine()


def model_available() -> bool:
    ENGINE.load()
    return ENGINE.loaded


def model_info() -> dict:
    ENGINE.load()
    m = ENGINE.metrics.get("orders", {})
    return {
        "available": ENGINE.loaded,
        "error": ENGINE.error,
        "trained_at": ENGINE.meta.get("trained_at"),
        "training_source": ENGINE.meta.get("source"),
        "training_rows": ENGINE.meta.get("rows"),
        "n_features": len(ENGINE.meta.get("features", [])),
        "has_aov_model": ENGINE.meta.get("has_aov_model", False),
        "has_quantile_models": bool(ENGINE.quantiles),
        "interval_coverage_pct": (ENGINE.metrics.get("quantiles", {})
                                  .get("interval_80", {}).get("empirical_coverage_pct")),
        "validation": {
            "orders_mape_pct": m.get("mape_pct"),
            "orders_r2": m.get("r2"),
            "naive_baseline_mape_pct":
                ENGINE.metrics.get("orders_naive_baseline", {}).get("mape_pct"),
            "aov_mape_pct": ENGINE.metrics.get("aov", {}).get("mape_pct"),
            "holdout_cutoff": ENGINE.meta.get("cutoff_date"),
        },
        "top_features": ENGINE.meta.get("top_features", [])[:15],
    }


# ── Human-readable helpers ────────────────────────────────────────────────────
def _weather_note(temp_max, precip_mm, weather_code, w_mult) -> str:
    bits = []
    if temp_max is not None:
        if temp_max >= 44:
            bits.append(f"extreme heat {temp_max:.0f}°C")
        elif temp_max >= 40:
            bits.append(f"high heat {temp_max:.0f}°C")
        elif temp_max <= 15:
            bits.append(f"cold snap {temp_max:.0f}°C")
    if precip_mm:
        if precip_mm >= 20:
            bits.append(f"heavy rain {precip_mm:.0f}mm")
        elif precip_mm >= 8:
            bits.append(f"moderate rain {precip_mm:.0f}mm")
        elif precip_mm >= 1:
            bits.append(f"light rain {precip_mm:.0f}mm")
    if weather_code in (95, 96, 99):
        bits.append("thunderstorm")
    if not bits:
        return "Normal weather — no significant weather impact"
    pct = (w_mult - 1) * 100
    return f"{', '.join(bits)} — model puts weather at {pct:+.0f}% vs a clear day"


_DRIVER_GROUPS = [
    ("Restaurant type & city baseline", ("restaurant_type", "city", "state", "zone", "tier", "lat", "lon")),
    ("Day of week", ("dow", "is_weekend", "is_saturday", "is_sunday", "is_friday")),
    ("Season & trend", ("month", "doy_sin", "doy_cos", "month_sin", "month_cos",
                        "week_of_year", "t_index", "year_index")),
    ("Salary cycle", ("is_payday_window", "is_month_end", "day_of_month",
                      "days_from_month_start")),
    ("Weather", ("temp_max", "temp_min", "temp_range", "precip_mm", "log_precip",
                 "weather_code", "is_rain", "is_heavy_rain", "is_thunder", "is_fog",
                 "is_extreme_heat", "is_high_heat", "is_cool", "is_pleasant",
                 "has_weather_data")),
]


def _grouped_drivers(X, top_n: int = 6, row_index: int = 0):
    """Group LightGBM SHAP contributions into readable buckets (log-space)."""
    try:
        contrib = ENGINE.contributions(X)[row_index][:-1]
    except Exception:                                  # noqa: BLE001
        return []
    cols = list(X.columns)
    named = dict(zip(cols, contrib))
    used, groups = set(), []
    for label, keys in _DRIVER_GROUPS:
        v = sum(named.get(k, 0.0) for k in keys)
        used.update(keys)
        groups.append((label, v))
    ev = sum(v for k, v in named.items() if k not in used)
    groups.append(("Events & holidays", ev))
    groups = [
        {"driver": g, "effect_pct": round((math.exp(v) - 1) * 100, 1)}
        for g, v in groups if abs(v) > 1e-4
    ]
    return sorted(groups, key=lambda d: -abs(d["effect_pct"]))[:top_n]


@lru_cache(maxsize=8192)
def _normalizer(city: str, rest_type: str, year: int) -> float:
    """Scale factor putting the output back on the app's 100-index scale.

    100 = a typical quiet weekday for THIS city × restaurant type in THIS year.
    Computed from 12 event-free, benign-weather Wednesdays (one per month), so
    city size, restaurant type and the growth trend are divided out while the
    within-year seasonal shape is preserved.
    """
    ref = []
    for m in range(1, 13):
        d = date(year, m, 15)
        d += timedelta(days=(2 - d.weekday()) % 7)          # nearest Wednesday
        ref.append(build_row(city, d, rest_type, evs=[], proximity=False,
                             **NEUTRAL_WEATHER))
    preds = ENGINE.predict_orders(rows_to_frame(ref))
    mean = float(np.mean(preds))
    return 100.0 / mean if mean > 1e-6 else 1.0


def _signal(pct_change: float) -> str:
    if pct_change >= 30:
        return "very_high_up"
    if pct_change >= 15:
        return "high_up"
    if pct_change >= 5:
        return "slight_up"
    if pct_change <= -35:
        return "very_low"
    if pct_change <= -20:
        return "low"
    if pct_change <= -8:
        return "slight_down"
    return "neutral"


# ── Main entry point ──────────────────────────────────────────────────────────
def _plan(city: str, pred_date: date, rest_type: str, wx: dict, explain: bool,
          perms: int = 0):
    """Build the counterfactual rows needed to score one day × restaurant type.

    Rows 0-3 are always A (the day as it is), B (weather only), C (events only)
    and D (baseline). After that come the rows needed to split credit between
    individual events:

      * ≤ SHAPLEY_MAX_EVENTS active events → every subset of events, so each
        event's credit is its exact Shapley value. This matters when events
        overlap: Diwali's gazetted holiday and the festival itself both fire on
        the same day, and leave-one-out gives each ~0% because removing one
        leaves the other. Shapley splits the joint lift between them instead.
      * more than that → sampled Shapley: a handful of random orderings, each
        contributing the marginal lift of adding that event next. The estimate
        stays exactly additive (the per-event effects still compose to the total
        event effect) — only the averaging is approximate.
    """
    evs = list(active_events(city, pred_date))

    def row(events, weather_actual: bool, proximity: bool):
        w = dict(wx) if weather_actual else dict(NEUTRAL_WEATHER)
        return build_row(city, pred_date, rest_type, evs=events,
                         proximity=proximity, **w)

    rows = [
        row(None, True, True),      # A — the day as it actually is
        row([], True, False),       # B — weather only, no events
        row(None, False, True),     # C — events only, benign weather
        row([], False, False),      # D — baseline
    ]
    attr = {"mode": "none"}
    n = len(evs)
    if explain and n:
        if n <= SHAPLEY_MAX_EVENTS:
            subsets = {}
            full = frozenset(range(n))
            for mask in range(1 << n):
                S = frozenset(i for i in range(n) if mask >> i & 1)
                if S == full:
                    subsets[S] = 0                     # row A already is the full set
                else:
                    subsets[S] = len(rows)
                    rows.append(row([evs[i] for i in sorted(S)], True, True))
            attr = {"mode": "shapley", "subsets": subsets}
        else:
            rng = np.random.default_rng(abs(hash((city, pred_date.toordinal(),
                                                  rest_type))) % (2 ** 32))
            k = perms if perms else SHAPLEY_PERMUTATIONS
            orders = [list(rng.permutation(n)) for _ in range(k)]
            empty_row = len(rows)
            rows.append(row([], True, True))
            prefix_rows = {}                       # (perm_idx, prefix_len) → row
            for pi, order in enumerate(orders):
                for j in range(1, n):
                    prefix_rows[(pi, j)] = len(rows)
                    rows.append(row([evs[i] for i in order[:j]], True, True))
            attr = {"mode": "shapley_sampled", "orders": orders,
                    "empty_row": empty_row, "prefix_rows": prefix_rows}
    return rows, evs, attr


def _shapley_effects(p, subsets, n) -> list:
    """Exact Shapley values in log space — they sum to the total event effect."""
    v = {S: math.log(max(float(p[i]), 1e-9)) for S, i in subsets.items()}
    phi = [0.0] * n
    for i in range(n):
        for S, val in v.items():
            if i in S:
                continue
            s = len(S)
            w = factorial(s) * factorial(n - s - 1) / factorial(n)
            phi[i] += w * (v[frozenset(S | {i})] - val)
    return [math.exp(x) - 1 for x in phi]


def _event_effects(p, evs, attr) -> list:
    """Per-event effect as a fraction (+0.18 = +18%), aligned with `evs`."""
    if attr.get("mode") == "shapley":
        return _shapley_effects(p, attr["subsets"], len(evs))
    if attr.get("mode") == "shapley_sampled":
        n = len(evs)
        lg = lambda i: math.log(max(float(p[i]), 1e-9))          # noqa: E731
        v_empty, v_full = lg(attr["empty_row"]), lg(0)
        phi = [0.0] * n
        for pi, order in enumerate(attr["orders"]):
            prev = v_empty
            for j, ev_i in enumerate(order):
                cur = v_full if j == n - 1 else lg(attr["prefix_rows"][(pi, j + 1)])
                phi[ev_i] += cur - prev
                prev = cur
        k = len(attr["orders"])
        return [math.exp(x / k) - 1 for x in phi]
    return [None] * len(evs)


def _assemble(city, pred_date, rest_type, wx, evs, attr, p, aov_pred,
              explain: bool, X=None, base_row: int = 0,
              q_pred=None, qscale: float = 1.0) -> dict:
    # `X` is passed only when grouped SHAP drivers are wanted — they cost ~150ms
    # per call, so callers that do not display them leave X as None.
    """Turn scored counterfactual rows into the app's prediction dict."""
    A, B, C = float(p[0]), float(p[1]), float(p[2])
    D = max(float(p[3]), 1e-6)

    event_mult = round(C / D, 3)
    weather_mult = round(B / D, 3)
    total_mult = round(A / D, 3)
    interaction = round(total_mult / max(event_mult * weather_mult, 1e-6), 3)
    pct_change = round((total_mult - 1) * 100, 1)

    temp_max = wx.get("temp_max")
    precip_mm = wx.get("precip_mm")
    weather_code = wx.get("weather_code")

    factors, contributions = [], []
    effects = _event_effects(p, evs, attr)
    for ev, eff in zip(evs, effects):
        if eff is None:
            factors.append(f"{ev['name']} ({ev['category']})")
            continue
        arrow = "↑" if eff > 0.005 else ("↓" if eff < -0.005 else "→")
        factors.append(f"{ev['name']} ({ev['category']}) {arrow}{eff * 100:+.0f}%")
        contributions.append({
            "event": ev["name"], "category": ev["category"],
            "subcategory": ev.get("subcategory"), "scope": ev.get("scope"),
            "effect_pct": round(eff * 100, 1),
            "method": attr.get("mode"),
        })

    w_note = _weather_note(temp_max, precip_mm, weather_code, weather_mult)
    if precip_mm or (temp_max is not None and (temp_max >= 40 or temp_max <= 15)) \
            or weather_code in (95, 96, 99):
        factors.append(w_note)

    if aov_pred is not None:
        aov_A, aov_D = float(aov_pred[0]), max(float(aov_pred[3]), 1e-6)
        aov_mult = round(aov_A / aov_D, 3)
        predicted_aov, aov_base = round(aov_A), round(aov_D)
    else:
        aov_mult = 1.0
        aov_base = BASELINE_AOV.get(rest_type, 500)
        predicted_aov = aov_base

    revenue_total_mult = round(total_mult * aov_mult, 3)

    # ── Uncertainty ───────────────────────────────────────────────────────────
    # Preferred: the trained quantile models (conformally calibrated), which widen
    # on genuinely uncertain days — festivals, storms — instead of stretching one
    # global spread over every day. Falls back to the residual-sigma band when the
    # quantile models have not been trained.
    interval_source = "residual_sigma"
    sigma = float(ENGINE.metrics.get("orders", {}).get("log_resid_std") or 0.12)
    lo, hi = round(A * math.exp(-1.28 * sigma), 1), round(A * math.exp(1.28 * sigma), 1)
    median = None
    if q_pred:
        try:
            lo = round(float(q_pred[0.1][0]) * qscale, 1)
            hi = round(float(q_pred[0.9][0]) * qscale, 1)
            if 0.5 in q_pred:
                median = round(float(q_pred[0.5][0]) * qscale, 1)
            interval_source = "quantile_conformal"
        except Exception:                                          # noqa: BLE001
            pass
    if hi < lo:
        lo, hi = hi, lo

    if len(evs) == 0 and abs(pct_change) < 8:
        confidence = "High"
    elif sigma > 0.25 or abs(pct_change) > 45:
        confidence = "Low"
    elif abs(pct_change) > 20 or len(evs) >= 3:
        confidence = "Medium"
    else:
        confidence = "High"

    out = {
        "date": pred_date.isoformat(),
        "city": city,
        "restaurant_type": rest_type,
        "weekday": pred_date.strftime("%A"),
        "base_index": round(D, 1),
        "event_multiplier": event_mult,
        "weather_multiplier": weather_mult,
        "total_multiplier": total_mult,
        "predicted_index": round(A, 1),
        "pct_change": pct_change,
        "signal": _signal(pct_change),
        "confidence": confidence,
        "active_events": [e["name"] for e in evs],
        "factors": factors,
        "weather_note": w_note,
        "aov_base": aov_base,
        "aov_multiplier": aov_mult,
        "predicted_aov": predicted_aov,
        "predicted_revenue_index": round(D * revenue_total_mult, 1),
        "revenue_pct_change": round((revenue_total_mult - 1) * 100, 1),
        # ── ML-only extras ────────────────────────────────────────────────────
        "engine": "ml",
        "model": "lightgbm",
        "interaction_multiplier": interaction,
        "prediction_interval_80": [lo, hi],
        "interval_source": interval_source,
        "predicted_index_p50": median,
        "expected_error_pct": ENGINE.metrics.get("orders", {}).get("mape_pct"),
        "training_source": ENGINE.meta.get("source"),
    }
    if explain:
        out["event_contributions"] = contributions
        if X is not None:
            out["top_drivers"] = _grouped_drivers(X, row_index=base_row)
    return out


def predict_impact_ml(city: str, pred_date: date, rest_type: str,
                      temp_max=None, precip_mm=None, weather_code=None,
                      temp_min=None, explain: bool = True,
                      drivers: bool = False) -> dict:
    """Drop-in ML replacement for the rule-based predict_impact().

    `explain` adds per-event leave-one-out attribution (cheap).
    `drivers` adds grouped SHAP driver breakdown (~150ms — opt in).
    """
    ENGINE.load()
    if not ENGINE.loaded:
        raise RuntimeError(ENGINE.error or "ML model not loaded")

    wx = dict(temp_max=temp_max, temp_min=temp_min, precip_mm=precip_mm,
              weather_code=weather_code)
    rows, evs, attr = _plan(city, pred_date, rest_type, wx, explain)
    X = rows_to_frame(rows)
    # Rescale onto the app's index convention (100 = quiet weekday for this
    # city x restaurant type). Multipliers are ratios and are unaffected.
    scale = _normalizer(city, rest_type, pred_date.year)
    p = ENGINE.predict_orders(X) * scale
    aov = ENGINE.predict_aov(X)
    q_pred = ENGINE.predict_quantiles(X, rows=[0])
    return _assemble(city, pred_date, rest_type, wx, evs, attr, p, aov,
                     explain, X=X if (explain and drivers) else None,
                     q_pred=q_pred, qscale=scale)


def predict_date_range_ml(city, start_dt, end_dt, temp_data=None, rest_types=None,
                          explain: bool = True):
    """Whole date range × all restaurant types, scored in ONE batched pass.

    Building every counterfactual row up front and calling LightGBM once is
    roughly an order of magnitude faster than looping predict_impact_ml(), which
    matters for the 30-day dashboard views.
    """
    from ml.config import RESTAURANT_TYPES

    ENGINE.load()
    if not ENGINE.loaded:
        raise RuntimeError(ENGINE.error or "ML model not loaded")

    rest_types = rest_types or RESTAURANT_TYPES
    all_rows, jobs = [], []
    cur = start_dt
    while cur <= end_dt:
        td = (temp_data or {}).get(cur.isoformat(), {})
        wx = dict(temp_max=td.get("temp_max_c"), temp_min=td.get("temp_min_c"),
                  precip_mm=td.get("precipitation_mm"),
                  weather_code=td.get("weather_code"))
        for rt in rest_types:
            rows, evs, attr = _plan(city, cur, rt, wx, explain,
                                    perms=SHAPLEY_PERMUTATIONS_BATCH)
            jobs.append((cur, rt, wx, evs, attr, len(all_rows), len(rows)))
            all_rows.extend(rows)
        cur += timedelta(days=1)

    X = rows_to_frame(all_rows)
    p = ENGINE.predict_orders(X)
    aov = ENGINE.predict_aov(X)
    a_rows = [off for *_, off, _n in jobs]              # row A of each job
    q_all = ENGINE.predict_quantiles(X, rows=a_rows)

    results = {rt: [] for rt in rest_types}
    for i, (d, rt, wx, evs, attr, off, n) in enumerate(jobs):
        scale = _normalizer(city, rt, d.year)
        seg = p[off:off + n] * scale
        seg_aov = aov[off:off + n] if aov is not None else None
        q_one = ({q: v[i:i + 1] for q, v in q_all.items()} if q_all is not None else None)
        results[rt].append(_assemble(city, d, rt, wx, evs, attr, seg, seg_aov,
                                     explain, X=None, q_pred=q_one, qscale=scale))
    return results


@lru_cache(maxsize=1)
def _warm():
    ENGINE.load()
    return ENGINE.loaded
