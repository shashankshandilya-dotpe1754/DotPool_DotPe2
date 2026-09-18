"""
ml/dataset.py
=============
Loads the training table and turns it into a model matrix.

Two sources, in priority order:
  1. data/training/real_history.csv   — your actual history (preferred)
  2. data/training/synthetic_history.csv — bootstrap panel from ml/simulate.py

Required columns
----------------
  date              YYYY-MM-DD
  city              must match a key in data/india_geo.CITIES
  restaurant_type   one of ml.config.RESTAURANT_TYPES
  orders            absolute daily orders  (or `orders_index`, already indexed)

Optional columns
----------------
  aov               average order value in ₹ (or `revenue`, from which AOV is derived)
  temp_max_c, temp_min_c, precipitation_mm, weather_code
                    if absent, weather features are left missing and LightGBM
                    handles them as NaN — the model still works, just blind to weather.

`orders` is converted to an index where 100 = that city×type's own quiet-weekday
average, which is what the API and dashboard already speak in and which lets
outlets of very different sizes be pooled into one model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ml.config import REAL_CSV, RESTAURANT_TYPES, SYNTHETIC_CSV
from ml.features import build_row, rows_to_frame

REQUIRED = ["date", "city", "restaurant_type"]


def source_path() -> str:
    if os.path.exists(REAL_CSV):
        return REAL_CSV
    return SYNTHETIC_CSV


def load_raw(path: str = None) -> pd.DataFrame:
    path = path or source_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No training data at {path}. Run `python -m ml.simulate` to bootstrap "
            f"a synthetic panel, or drop your own CSV at {REAL_CSV}."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if "orders_index" not in df.columns and "orders" not in df.columns:
        raise ValueError(f"{path} needs either an `orders` or an `orders_index` column.")
    if "aov" not in df.columns and {"revenue", "orders"} <= set(df.columns):
        df["aov"] = df["revenue"] / df["orders"].replace(0, np.nan)
    df = df[df["restaurant_type"].isin(RESTAURANT_TYPES)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute orders → index (100 = quiet weekday for that city × type)."""
    if "orders_index" in df.columns:
        return df
    df = df.copy()
    d = pd.to_datetime(df["date"])
    quiet = d.dt.weekday < 5
    base = (
        df[quiet].groupby(["city", "restaurant_type"])["orders"].median().rename("base")
    )
    df = df.merge(base, on=["city", "restaurant_type"], how="left")
    df["base"] = df["base"].replace(0, np.nan)
    df["orders_index"] = 100.0 * df["orders"] / df["base"]
    return df.drop(columns=["base"])


def build_matrix(df: pd.DataFrame):
    """DataFrame of raw history → (X, y_orders, y_aov, dates)."""
    # Calendar/event/weather features depend only on (city, date) — compute them
    # once per day and reuse across the 6 restaurant types. ~6x faster.
    from data.india_geo import CITIES
    from ml.features import calendar_features, event_features, weather_features

    ctx_cache = {}
    rows = []
    for r in df.itertuples(index=False):
        key = (r.city, r.date)
        ctx = ctx_cache.get(key)
        if ctx is None:
            geo = CITIES.get(r.city, {})
            ctx = {
                "city": r.city,
                "state": geo.get("state", "Unknown"),
                "zone": geo.get("zone", "Unknown"),
            }
            ctx.update(calendar_features(r.city, r.date))
            ctx.update(event_features(r.city, r.date))
            ctx.update(weather_features(
                getattr(r, "temp_max_c", None), getattr(r, "temp_min_c", None),
                getattr(r, "precipitation_mm", None), getattr(r, "weather_code", None),
            ))
            ctx_cache[key] = ctx
        rows.append(dict(ctx, restaurant_type=r.restaurant_type))
    X = rows_to_frame(rows)
    y_orders = np.log(np.clip(df["orders_index"].to_numpy(dtype=float), 1e-3, None))
    y_aov = (
        np.log(np.clip(df["aov"].to_numpy(dtype=float), 1e-3, None))
        if "aov" in df.columns else None
    )
    return X, y_orders, y_aov, pd.to_datetime(df["date"]).to_numpy()


def load_dataset(path: str = None):
    df = normalize_orders(load_raw(path))
    df = df.dropna(subset=["orders_index"])
    return build_matrix(df), df
