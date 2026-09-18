"""
ml/config.py
============
Central configuration for the DotPool ML forecasting engine.

Everything that the training pipeline, the feature builder and the inference
layer need to agree on lives here — so there is exactly one definition of
restaurant types, model paths, and the synthetic-data assumptions.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(ML_DIR)
MODEL_DIR    = os.path.join(ML_DIR, "models")
TRAINING_DIR = os.path.join(ROOT_DIR, "data", "training")

ORDERS_MODEL_PATH  = os.path.join(MODEL_DIR, "orders_lgbm.txt")
# Quantile models — honest per-day prediction intervals (see ml/train.py)
QUANTILES          = [0.1, 0.5, 0.9]
QUANTILE_PATHS     = {q: os.path.join(MODEL_DIR, f"orders_q{int(q*100)}_lgbm.txt")
                      for q in QUANTILES}
AOV_MODEL_PATH     = os.path.join(MODEL_DIR, "aov_lgbm.txt")
META_PATH          = os.path.join(MODEL_DIR, "model_meta.json")
METRICS_PATH       = os.path.join(MODEL_DIR, "metrics.json")

# Where the training table lives. Drop a real-world CSV here with the same
# column names and `python -m ml.train` will use it instead of synthetic data.
SYNTHETIC_CSV = os.path.join(TRAINING_DIR, "synthetic_history.csv")
REAL_CSV      = os.path.join(TRAINING_DIR, "real_history.csv")

# Actuals for anomaly scanning (same schema as real_history.csv)
ACTUALS_CSV   = os.path.join(TRAINING_DIR, "actuals.csv")
# Campaign log for uplift modelling (see ml/uplift.py for the schema)
CAMPAIGNS_CSV = os.path.join(TRAINING_DIR, "campaigns.csv")

# Review queue written by the LLM event ingester (never merged unreviewed)
PENDING_EVENTS_JSON = os.path.join(ROOT_DIR, "data", "pending_events.json")

for _d in (MODEL_DIR, TRAINING_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Restaurant types ──────────────────────────────────────────────────────────
RESTAURANT_TYPES = [
    "QSR", "Fine Dining", "PBCL", "Casual Dining", "Cloud Kitchen", "Cafe",
]

# Reference AOV (₹) per type — used as the scale the AOV model predicts against.
BASELINE_AOV = {
    "QSR":           330,
    "Fine Dining":   2100,
    "PBCL":          1350,
    "Casual Dining": 950,
    "Cloud Kitchen": 480,
    "Cafe":          420,
}

# ── Event taxonomy the feature builder encodes ────────────────────────────────
EVENT_CATEGORIES = [
    "Festival", "Government Holiday", "Sports Event", "Commercial Event",
    "Public Event", "Government Order", "Emergency Crisis", "Weather Event",
]

# Subcategories that carry a distinct demand signature and therefore get their
# own binary feature. Everything else falls back to its parent category.
EVENT_SUBCATEGORIES = [
    # Festival
    "Hindu", "Muslim", "Christian", "Sikh", "Jain", "Buddhist", "Regional",
    # Holiday / public
    "National Holiday", "Civic", "Political",
    # Sports
    "Cricket", "Football", "Kabaddi", "Multi-Sport",
    # Commercial
    "E-commerce Sale", "F&B Promotion", "Commercial",
    # Government orders / regulation
    "Liquor Ban", "Traffic Regulation", "Operating Hours", "Environmental",
    "Digital Mandate", "Food Safety", "Labour Regulation",
    # Crisis / weather
    "Heatwave", "Summer Heatwave", "Flood", "Cyclone", "Air Quality",
    "Natural Disaster", "Monsoon", "Winter",
]

# Ordinal encoding of the events DB `impact_on_demand` label. This is *not* a
# hand-tuned multiplier — it is a coarse prior the model is free to override,
# ignore, or invert per restaurant type.
IMPACT_ORDINAL = {
    "very_high_up":            3,
    "high_up":                 2,
    "slight_up":               1,
    "neutral_to_slight_up":    0.5,
    "neutral":                 0,
    "neutral_to_slight_down": -0.5,
    "slight_down":            -1,
    "low":                    -2,
    "very_low":               -3,
}

# ── Geography ─────────────────────────────────────────────────────────────────
# Metro / tier-1 / tier-2 classification drives baseline order volume and AOV.
TIER_1 = {
    "New Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad",
}
TIER_2 = {
    "Gurugram", "Noida", "Jaipur", "Lucknow", "Chandigarh", "Kochi", "Indore",
    "Bhopal", "Nagpur", "Surat", "Coimbatore", "Visakhapatnam", "Vadodara",
    "Ludhiana", "Thane", "Navi Mumbai", "Mysuru", "Bhubaneswar", "Guwahati",
    "Goa", "Panaji", "Dehradun", "Ranchi", "Raipur", "Varanasi", "Agra",
    "Madurai", "Trivandrum", "Thiruvananthapuram", "Amritsar", "Nashik",
}


def city_tier(city: str) -> int:
    if city in TIER_1:
        return 1
    if city in TIER_2:
        return 2
    return 3


# ── Training config ───────────────────────────────────────────────────────────
TRAIN_START = "2023-01-01"
TRAIN_END   = "2026-12-31"

# Cities sampled for the synthetic training panel. Keeping this to a
# representative national spread keeps training fast; geography enters the model
# through zone / state / tier / lat / lon features, so unseen cities still score.
SYNTH_CITY_SAMPLE = 46

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 96,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}
NUM_BOOST_ROUND = 1200
EARLY_STOPPING  = 60

RANDOM_SEED = 42


# ── Quantile training (lighter than the mean model — 3 extra boosters) ────────
QUANTILE_BOOST_ROUND = 700

# ── Anomaly detection ─────────────────────────────────────────────────────────
# A day is flagged when its log-residual falls outside the conformal band built
# from the trailing history. 0.99 ≈ flag the worst 1% of days.
ANOMALY_CONFORMAL_LEVEL = 0.99
ANOMALY_MIN_HISTORY     = 30

# ── LLM layer ─────────────────────────────────────────────────────────────────
# Key is read from the environment or Streamlit secrets — never hard-coded.
LLM_PROVIDER   = os.environ.get("DOTPOOL_LLM_PROVIDER", "anthropic")
LLM_MODEL      = os.environ.get("DOTPOOL_LLM_MODEL", "")
LLM_MAX_TOKENS = 2048
