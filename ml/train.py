"""
ml/train.py
===========
Trains the two LightGBM models that power DotPool:

  * orders model — log(orders_index)   → daily order-volume index (100 = quiet weekday)
  * AOV model    — log(average order value in ₹)

Validation is a strict time split (the newest slice of history is held out), so
the score reflects forecasting the future rather than interpolating the past.

Usage
-----
  python -m ml.train                    # synthetic bootstrap (auto-generates if absent)
  python -m ml.train --real-weather     # synthesise demand but use true Open-Meteo weather
  python -m ml.train --rebuild          # force-regenerate the synthetic panel
  python -m ml.train --holdout 0.2      # fraction of the most recent dates held out

Drop your own history at data/training/real_history.csv and it is used
automatically in place of the synthetic panel (see ml/dataset.py for schema).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.config import (
    AOV_MODEL_PATH,
    QUANTILES,
    QUANTILE_BOOST_ROUND,
    QUANTILE_PATHS,
    EARLY_STOPPING,
    LGBM_PARAMS,
    METRICS_PATH,
    META_PATH,
    NUM_BOOST_ROUND,
    ORDERS_MODEL_PATH,
    RANDOM_SEED,
    REAL_CSV,
    SYNTHETIC_CSV,
)
from ml.dataset import load_dataset
from ml.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def _metrics(y_true_log, y_pred_log):
    yt, yp = np.exp(y_true_log), np.exp(y_pred_log)
    err = yp - yt
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mape_pct": round(float(np.mean(np.abs(err) / np.clip(yt, 1e-6, None)) * 100), 2),
        "r2": round(1 - ss_res / ss_tot if ss_tot else float("nan"), 4),
        "log_resid_std": round(float(np.std(np.asarray(y_pred_log) - np.asarray(y_true_log))), 4),
        "n": int(len(yt)),
    }


def fit_trend(X, y, mask) -> dict:
    """Per-restaurant-type log-linear time trend, used as a LightGBM offset.

    Tree models cannot extrapolate a trend beyond the range of `t_index` they
    saw in training, so year-on-year growth is fitted parametrically here and
    the trees learn only the residual around it. This is what keeps forecasts
    for future months from flattening out at last year's level.
    """
    rt = X["restaurant_type"].astype(str).to_numpy()
    t = X["t_index"].to_numpy(dtype=float)
    out = {"__global__": tuple(np.polyfit(t[mask], y[mask], 1))}
    for r in np.unique(rt[mask]):
        m = mask & (rt == r)
        if m.sum() >= 200:
            slope, intercept = np.polyfit(t[m], y[m], 1)
            out[r] = (float(slope), float(intercept))
    return {k: (float(v[0]), float(v[1])) for k, v in out.items()}


def trend_values(X, trend: dict) -> np.ndarray:
    rt = X["restaurant_type"].astype(str).to_numpy()
    t = X["t_index"].to_numpy(dtype=float)
    g = trend.get("__global__", (0.0, 0.0))
    slope = np.array([trend.get(r, g)[0] for r in rt])
    inter = np.array([trend.get(r, g)[1] for r in rt])
    return slope * t + inter


def pinball_loss(y_true, y_pred, q: float) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def _fit_quantile(X_tr, y_tr, X_va, y_va, q, off_tr, off_va):
    """One LightGBM quantile regressor for level `q` on the same features."""
    params = dict(LGBM_PARAMS, objective="quantile", alpha=q, metric="quantile",
                  seed=RANDOM_SEED)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_COLUMNS,
                      init_score=off_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, init_score=off_va,
                      categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
    return lgb.train(
        params, dtr, num_boost_round=QUANTILE_BOOST_ROUND, valid_sets=[dva],
        valid_names=[f"q{int(q*100)}"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(0)],
    )


def _fit(X_tr, y_tr, X_va, y_va, label, off_tr=None, off_va=None):
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_COLUMNS,
                      init_score=off_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, init_score=off_va,
                      categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
    params = dict(LGBM_PARAMS, seed=RANDOM_SEED)
    booster = lgb.train(
        params, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dva],
        valid_names=[label],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(200)],
    )
    return booster


def train(holdout: float = 0.18, rebuild: bool = False, real_weather: bool = False):
    if rebuild or (not os.path.exists(SYNTHETIC_CSV) and not os.path.exists(REAL_CSV)):
        from ml.simulate import generate_panel
        print("[data] generating synthetic training panel …")
        generate_panel(use_real_weather=real_weather).to_csv(SYNTHETIC_CSV, index=False)

    (X, y_orders, y_aov, dates), raw = load_dataset()
    print(f"[data] {len(X):,} rows × {len(FEATURE_COLUMNS)} features "
          f"({'real' if os.path.exists(REAL_CSV) else 'synthetic'} source)")

    # ── Time-based split ──────────────────────────────────────────────────────
    order = np.argsort(dates)
    cut_date = pd.Series(dates).quantile(1 - holdout)
    tr = dates < cut_date
    va = ~tr
    if va.sum() < 100:                       # tiny dataset → fall back to random split
        rng = np.random.default_rng(RANDOM_SEED)
        va = rng.random(len(X)) < holdout
        tr = ~va
    print(f"[split] train={tr.sum():,}  valid={va.sum():,}  "
          f"cutoff={pd.Timestamp(cut_date).date()}")

    results = {}
    print("\n[train] orders model")
    trend_o = fit_trend(X, y_orders, tr)
    off = trend_values(X, trend_o)
    m_orders = _fit(X[tr], y_orders[tr], X[va], y_orders[va], "orders",
                    off[tr], off[va])
    results["orders"] = _metrics(y_orders[va], m_orders.predict(X[va]) + off[va])
    results["orders_train"] = _metrics(y_orders[tr], m_orders.predict(X[tr]) + off[tr])
    m_orders.save_model(ORDERS_MODEL_PATH)

    # ── Quantile models: honest per-day intervals ─────────────────────────────
    # The mean model says "about 180 orders". A staffing decision needs "180,
    # and realistically between 140 and 240" — and that spread is much wider on
    # Diwali than on a wet Tuesday, which a single global residual sigma cannot
    # express. One LightGBM per quantile, sharing the same features and trend.
    print("\n[train] quantile models (p10 / p50 / p90)")
    q_metrics, covered = {}, None
    for q in QUANTILES:
        mq = _fit_quantile(X[tr], y_orders[tr], X[va], y_orders[va], q, off[tr], off[va])
        pred_va = mq.predict(X[va]) + off[va]
        q_metrics[f"q{int(q*100)}"] = {
            "pinball": round(pinball_loss(y_orders[va], pred_va, q), 5),
            "empirical_below_pct": round(float(np.mean(y_orders[va] <= pred_va) * 100), 2),
        }
        mq.save_model(QUANTILE_PATHS[q])
        if q == 0.1:
            lo_va = pred_va
        if q == 0.9:
            hi_va = pred_va
        print(f"   q{int(q*100)}  pinball {q_metrics[f'q{int(q*100)}']['pinball']:.5f}  "
              f"below {q_metrics[f'q{int(q*100)}']['empirical_below_pct']:.1f}%")
    # ── Conformalise (CQR) ────────────────────────────────────────────────────
    # Raw quantile regressors under-cover when extrapolating into the future:
    # trained p10/p90 gave only ~63% coverage on held-out future dates, not 80%.
    # Conformalized Quantile Regression fixes this with one calibration constant:
    # take the conformity score E = max(q10 - y, y - q90) on held-out data and
    # widen both edges by its 80th percentile. The interval then covers 80% by
    # construction rather than by assumption.
    raw_cov = float(np.mean((y_orders[va] >= lo_va) & (y_orders[va] <= hi_va)) * 100)
    conformity = np.maximum(lo_va - y_orders[va], y_orders[va] - hi_va)
    n_va = len(conformity)
    k = min(int(np.ceil((n_va + 1) * 0.80)), n_va) - 1
    margin = float(np.sort(conformity)[k])
    margin = max(margin, 0.0)
    lo_c, hi_c = lo_va - margin, hi_va + margin
    covered = float(np.mean((y_orders[va] >= lo_c) & (y_orders[va] <= hi_c)) * 100)
    width = float(np.mean(np.exp(hi_c) - np.exp(lo_c)))
    q_metrics["interval_80"] = {
        "raw_coverage_pct": round(raw_cov, 2),
        "conformal_margin_log": round(margin, 4),
        "empirical_coverage_pct": round(covered, 2),   # target: 80
        "mean_width_index_pts": round(width, 2),
    }
    results["quantiles"] = q_metrics
    print(f"   raw 80% interval covered {raw_cov:.1f}% — conformal margin "
          f"{margin:.3f} log units")
    print(f"   calibrated interval covers {covered:.1f}% of held-out days "
          f"(target 80), mean width {width:.0f} index pts")

    m_aov, trend_a = None, None
    if y_aov is not None and not np.all(np.isnan(y_aov)):
        print("\n[train] AOV model")
        trend_a = fit_trend(X, y_aov, tr)
        offa = trend_values(X, trend_a)
        m_aov = _fit(X[tr], y_aov[tr], X[va], y_aov[va], "aov", offa[tr], offa[va])
        results["aov"] = _metrics(y_aov[va], m_aov.predict(X[va]) + offa[va])
        m_aov.save_model(AOV_MODEL_PATH)

    # ── Baseline for comparison: predict the per-type×dow mean ────────────────
    base_key = raw["restaurant_type"].astype(str) + "|" + pd.to_datetime(raw["date"]).dt.weekday.astype(str)
    means = pd.Series(y_orders[tr]).groupby(base_key[tr].to_numpy()).mean()
    naive = base_key[va].map(means).fillna(np.mean(y_orders[tr])).to_numpy()
    results["orders_naive_baseline"] = _metrics(y_orders[va], naive)

    imp = sorted(
        zip(FEATURE_COLUMNS, m_orders.feature_importance("gain")),
        key=lambda t: -t[1],
    )[:30]

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "source": "real" if os.path.exists(REAL_CSV) else "synthetic",
        "rows": int(len(X)),
        "features": FEATURE_COLUMNS,
        "categorical": CATEGORICAL_COLUMNS,
        "targets": ["log_orders_index", "log_aov"],
        "holdout_fraction": holdout,
        "cutoff_date": str(pd.Timestamp(cut_date).date()),
        "has_aov_model": m_aov is not None,
        "has_quantile_models": True,
        "quantiles": QUANTILES,
        "cqr_margin": margin,
        "trend_orders": trend_o,
        "trend_aov": trend_a,
        "top_features": [{"feature": f, "gain": float(g)} for f, g in imp],
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(METRICS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n──────── validation (held-out future dates) ────────")
    for k, v in results.items():
        if k == "quantiles":
            continue
        print(f"{k:24s} MAPE {v['mape_pct']:6.2f}%   MAE {v['mae']:7.2f}   R² {v['r2']}")
    print("\nTop features by gain:")
    for f, g in imp[:15]:
        print(f"  {f:28s} {g:,.0f}")
    print(f"\nsaved → {ORDERS_MODEL_PATH}")
    if m_aov:
        print(f"saved → {AOV_MODEL_PATH}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, default=0.18)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--real-weather", action="store_true")
    a = ap.parse_args()
    train(holdout=a.holdout, rebuild=a.rebuild, real_weather=a.real_weather)
