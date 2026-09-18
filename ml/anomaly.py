"""
ml/anomaly.py
=============
Anomaly detection on actuals — the layer that tells you when reality and the
model disagree, and which of the two is probably wrong.

Two different jobs, deliberately separated:

1. DAY ANOMALIES — a single outlet-day that the model badly missed. In practice
   these are rarely model failures: a POS outage, an outlet that closed for a
   private event, a delivery-partner outage, a road dug up outside, a competitor
   opening across the street. This is the operational alert.

2. DRIFT — the model degrading *systematically* over a rolling window. Consistent
   one-sided error means the world moved (new outlets, a price change, a demand
   shift the features do not capture) and the model needs retraining on newer
   history. This is the maintenance alert.

Method
------
Residuals are measured in log space (r = log(actual) − log(forecast)), which
makes the error scale-free: a 20% miss on a small cafe and on a large QSR count
the same. The flagging band is CONFORMAL — the empirical quantiles of that
outlet's own past residuals — rather than a Gaussian assumption, because demand
residuals are fat-tailed and skewed. No distributional assumption, and the band
adapts per city × restaurant type.

Usage
-----
    python -m ml.anomaly --csv data/training/actuals.csv
    python -m ml.anomaly --csv actuals.csv --level 0.995 --out flagged.csv

Input schema — same as real_history.csv:
    date, city, restaurant_type, orders   (+ optional aov, weather columns)
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

from ml.config import ACTUALS_CSV, ANOMALY_CONFORMAL_LEVEL, ANOMALY_MIN_HISTORY
from ml.dataset import load_raw, normalize_orders


# ── Scoring ───────────────────────────────────────────────────────────────────
def _forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the model's expectation to every actual row (batched by city)."""
    from ml.predict import predict_date_range_ml

    out = []
    for city, g in df.groupby("city"):
        s, e = min(g["date"]), max(g["date"])
        try:
            preds = predict_date_range_ml(city, s, e, explain=False)
        except Exception as exc:                                   # noqa: BLE001
            raise RuntimeError(f"forecast failed for {city}: {exc}") from exc
        lookup = {}
        for rt, rows in preds.items():
            for p in rows:
                lookup[(p["date"], rt)] = p
        for r in g.itertuples(index=False):
            p = lookup.get((r.date.isoformat(), r.restaurant_type))
            if not p:
                continue
            out.append({
                "date": r.date, "city": city, "restaurant_type": r.restaurant_type,
                "actual_index": float(r.orders_index),
                "forecast_index": float(p["predicted_index"]),
                "pi_low": (p.get("prediction_interval_80") or [None, None])[0],
                "pi_high": (p.get("prediction_interval_80") or [None, None])[1],
                "events": ", ".join(p.get("active_events", [])[:3]),
            })
    return pd.DataFrame(out)


def scan(df: pd.DataFrame = None, path: str = None,
         level: float = ANOMALY_CONFORMAL_LEVEL) -> dict:
    """Score every row, flag the outliers, and report drift per series."""
    if df is None:
        df = normalize_orders(load_raw(path or ACTUALS_CSV))
    df = df.dropna(subset=["orders_index"]).copy()

    scored = _forecast_frame(df)
    if scored.empty:
        return {"error": "no rows could be scored — check city and restaurant_type spellings"}

    scored["log_resid"] = (np.log(np.clip(scored["actual_index"], 1e-6, None))
                           - np.log(np.clip(scored["forecast_index"], 1e-6, None)))
    scored["error_pct"] = (scored["actual_index"] / scored["forecast_index"] - 1) * 100

    flagged, series_stats = [], []
    for (city, rt), g in scored.groupby(["city", "restaurant_type"]):
        g = g.sort_values("date")
        r = g["log_resid"].to_numpy()
        n = len(r)
        # Conformal band from this series' own residual distribution.
        if n >= ANOMALY_MIN_HISTORY:
            lo, hi = np.quantile(r, [1 - level, level])
        else:                                   # too little history — fall back wide
            lo, hi = np.quantile(r, [0.02, 0.98]) if n >= 10 else (-np.inf, np.inf)
        g = g.assign(band_low=lo, band_high=hi)
        g["is_anomaly"] = (g["log_resid"] < lo) | (g["log_resid"] > hi)
        # Severity: how far outside the band, in band-widths
        width = max(hi - lo, 1e-6)
        g["severity"] = np.where(
            g["log_resid"] > hi, (g["log_resid"] - hi) / width,
            np.where(g["log_resid"] < lo, (lo - g["log_resid"]) / width, 0.0))
        g["direction"] = np.where(g["log_resid"] > 0, "above forecast", "below forecast")
        flagged.append(g[g["is_anomaly"]])

        recent = g.tail(28)
        series_stats.append({
            "city": city, "restaurant_type": rt, "days": int(n),
            "mape_pct": round(float(np.mean(np.abs(g["error_pct"]))), 2),
            "bias_pct": round(float(np.mean(g["error_pct"])), 2),
            "recent_bias_pct": round(float(np.mean(recent["error_pct"])), 2),
            "recent_mape_pct": round(float(np.mean(np.abs(recent["error_pct"]))), 2),
            "anomalies": int(g["is_anomaly"].sum()),
        })

    flagged_df = (pd.concat(flagged).sort_values("severity", ascending=False)
                  if flagged else pd.DataFrame())
    stats = pd.DataFrame(series_stats)

    # Drift: a series whose recent window is consistently one-sided.
    drift = stats[(stats["days"] >= ANOMALY_MIN_HISTORY)
                  & (stats["recent_bias_pct"].abs() > 12)].copy()
    drift["diagnosis"] = np.where(
        drift["recent_bias_pct"] > 0,
        "running hot — actuals above forecast, model is under-predicting",
        "running cold — actuals below forecast, model is over-predicting")

    return {
        "scored_rows": int(len(scored)),
        "series": stats.sort_values("mape_pct", ascending=False).to_dict("records"),
        "anomalies": flagged_df.to_dict("records") if len(flagged_df) else [],
        "n_anomalies": int(len(flagged_df)),
        "drift": drift.to_dict("records"),
        "overall_mape_pct": round(float(np.mean(np.abs(scored["error_pct"]))), 2),
        "overall_bias_pct": round(float(np.mean(scored["error_pct"])), 2),
        "level": level,
        "note": ("Positive bias means actuals ran above forecast. Sustained bias on many "
                 "series means retrain; isolated day anomalies usually mean something "
                 "happened at the outlet, not in the model."),
        "_scored_frame": scored,
    }


def explain_anomaly(row: dict) -> str:
    """One-line operator-readable summary of a flagged day."""
    d = row.get("date")
    d = d.isoformat() if hasattr(d, "isoformat") else str(d)
    pct = row.get("error_pct", 0)
    ev = row.get("events") or "no notable events"
    return (f"{d}  {row.get('city')} / {row.get('restaurant_type')}: actual "
            f"{row.get('actual_index'):.0f} vs forecast {row.get('forecast_index'):.0f} "
            f"({pct:+.0f}%, {row.get('direction')}) — active: {ev}")


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=ACTUALS_CSV, help="actuals file to scan")
    ap.add_argument("--level", type=float, default=ANOMALY_CONFORMAL_LEVEL,
                    help="conformal level, e.g. 0.99 flags the worst 1%% of days")
    ap.add_argument("--out", help="write flagged rows to this CSV")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        print(f"No actuals file at {a.csv}.\n"
              f"Export daily orders as: date,city,restaurant_type,orders  "
              f"(optionally aov and weather columns) and rerun.")
        return
    res = scan(path=a.csv, level=a.level)
    if "error" in res:
        print(res["error"]); return

    print(f"scanned {res['scored_rows']:,} outlet-days  |  "
          f"overall MAPE {res['overall_mape_pct']}%  bias {res['overall_bias_pct']:+.2f}%")
    print(f"\n{res['n_anomalies']} anomalous day(s) at level {a.level}:")
    for r in res["anomalies"][:25]:
        print("  " + explain_anomaly(r))
    if res["drift"]:
        print("\nDrift warnings (recent 28 days consistently one-sided):")
        for d in res["drift"]:
            print(f"  {d['city']} / {d['restaurant_type']}: "
                  f"{d['recent_bias_pct']:+.1f}% — {d['diagnosis']}")
        print("\n  → retrain on newer history: python -m ml.train")
    if a.out and res["anomalies"]:
        pd.DataFrame(res["anomalies"]).to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    _main()
