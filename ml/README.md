# DemandPulse ML — learned demand forecasting

Replaces the hand-written multiplier tables in `utils/restaurant_impact.py` with
two trained **LightGBM** models:

| model | target | file |
|---|---|---|
| orders | `log(orders_index)` — 100 = quiet weekday for that city × type | `ml/models/orders_lgbm.txt` |
| AOV | `log(average order value ₹)` | `ml/models/aov_lgbm.txt` |

Everything the old engine hard-coded (`1.55` for cricket + cloud kitchen, `0.75`
for heat, `0.35` for a liquor ban) is now a **counterfactual run of the model**,
so the numbers come from data instead of judgement.

---

## Quick start

```bash
pip install -r requirements.txt          # adds lightgbm
python -m ml.simulate                    # bootstrap training panel (~8s, 403k rows)
python -m ml.train                       # train both models (~3 min CPU)
uvicorn main:app --reload                # API now serves /forecast/*
streamlit run dashboard.py               # dashboard uses the ML engine automatically
```

If no model file exists, the app silently falls back to the old rule engine —
nothing breaks on a fresh clone.

| env var | effect |
|---|---|
| *(unset)* | ML if a trained model exists, else rules |
| `DEMANDPULSE_ENGINE=ml` | force ML, raise if unavailable |
| `DEMANDPULSE_ENGINE=rules` | force the legacy multiplier engine |

---

## What the model sees (93 features)

All Pan-India factors the events DB already tracks are fed in as *descriptive*
features — the model learns their effect per restaurant type, city and season:

1. **Festivals** — Hindu / Muslim / Christian / Sikh / Jain / Buddhist / regional,
   with multi-day span, eve / day-after flags and days-to-next-festival.
2. **Events** — national and city-level public events, commercial events,
   e-commerce sale windows, F&B promotions.
3. **Local events** — geo-scoped (`city` / `state` / `zone`) events, counted
   separately from pan-India ones, since a local event hits its own city harder.
4. **Sports** — cricket, football, kabaddi, multi-sport, as separate features.
5. **Weather** — max/min temperature, range, precipitation (raw + log), WMO code,
   rain / heavy rain / thunderstorm / fog / heat / cold / pleasant-day flags.
6. **Government rules & regulations** — liquor bans, traffic restrictions,
   operating-hour orders, environmental rules, digital mandates, food-safety and
   labour regulation, each as its own feature.
7. **Emergency crises** — heatwave, flood, cyclone, air quality, natural disaster.
8. **Calendar** — day of week, month, week of year, cyclic day-of-year terms,
   Indian salary cycle (1st–5th payday burst, month-end dip), long-weekend and
   holiday-adjacency flags.
9. **Geography** — city, state, zone, tier-1/2/3, latitude, longitude.
10. **Trend** — a per-restaurant-type log-linear time trend fitted parametrically
    and used as a LightGBM offset, because trees cannot extrapolate growth.

No lag features are used: the app must score arbitrary future dates where
yesterday's orders are unknown.

---

## How the output stays drop-in compatible

`predict_impact()` returns the same keys as before. The multipliers are produced
by scoring the same day four ways:

```
D = no events, benign weather   → base_index
C = events,    benign weather   → event_multiplier   = C / D
B = no events, actual weather   → weather_multiplier = B / D
A = the day as it is            → total_multiplier   = A / D
```

`interaction_multiplier = total / (event × weather)` exposes what the model
learned that the old multiplicative rule could not — e.g. a festival during heavy
rain is not simply festival × rain.

Per-event `factors` come from **Shapley attribution**: the model is re-scored
across subsets of the day's active events (exact for up to 5 events, sampled
over random orderings beyond that), so each event gets its fair share of the
joint lift. This matters because events overlap — Diwali fires as both a
gazetted holiday and a festival, plus whatever sports seasons and standing
government orders are running — and naive leave-one-out would credit each of
them ~0% since removing one leaves the others in place. The effects are exactly
additive: they compose to the day's total event effect.

The joint effect is also where the old engine went most wrong: it multiplied
every active event together and produced +261% on Dussehra in Delhi, where the
model — which learned that stacked festivals saturate — says +34%.

ML-only extras added to the response: `prediction_interval_80`, `top_drivers`
(grouped SHAP contributions), `event_contributions`, `expected_error_pct`,
`engine`, `training_source`.

---

## Accuracy (held-out future dates, synthetic bootstrap)

Validation is a **time split** — trained on everything before 2026-04-13, scored
on everything after, so it measures forecasting, not interpolation.

| model | MAPE | R² |
|---|---|---|
| orders (ML) | **8.68%** | 0.935 |
| orders — naive day-of-week × type baseline | 28.05% | 0.125 |
| AOV (ML) | 4.66% | 0.986 |

Read this as *"the pipeline works and beats the obvious baseline by 3×"*, not as
a claim about real Indian restaurants — see the caveat below.

---

## ⚠️ Train it on real data

`ml/simulate.py` generates the training panel from a latent process (city
heterogeneity, seasonality, payday cycle, growth, non-linear weather response,
event build-up/decay, noise). A model trained on it reproduces those assumptions.
It is a working pipeline and a reasonable prior — **not evidence** about real
demand.

To make it empirical, drop your history at `data/training/real_history.csv`:

| column | required | notes |
|---|---|---|
| `date` | ✅ | `YYYY-MM-DD` |
| `city` | ✅ | must match a key in `data/india_geo.CITIES` |
| `restaurant_type` | ✅ | one of the six types |
| `orders` | ✅ | absolute daily orders (or pre-indexed `orders_index`) |
| `aov` | optional | ₹ per order (or supply `revenue` and it is derived) |
| `temp_max_c`, `temp_min_c`, `precipitation_mm`, `weather_code` | optional | if absent, weather features are left missing and LightGBM treats them as NaN |

Then:

```bash
python -m ml.train        # real_history.csv wins over the synthetic panel automatically
```

`orders` is converted internally to the 100-index using each city × type's own
quiet-weekday median, so outlets of very different sizes pool into one model.
A few hundred days per city × type is enough to start; the more festival and
crisis days the history covers, the better those coefficients get.

---

## Files

```
ml/
├── config.py     paths, restaurant types, event taxonomy, LightGBM params
├── features.py   the single source of truth for feature engineering
├── simulate.py   synthetic bootstrap panel + offline climatology
├── dataset.py    CSV loading, order→index normalisation, matrix building
├── train.py      time-split training, trend offset, metrics, artefacts
├── predict.py    inference, counterfactual multipliers, SHAP drivers
└── models/       trained boosters + metrics.json + model_meta.json
```

## API

| endpoint | what it does |
|---|---|
| `GET /forecast/day?city=Bengaluru` | one day, all six restaurant types, live weather attached |
| `GET /forecast/range?city=Mumbai&start_date=…&end_date=…` | up to 120 days |
| `GET /forecast/compare?city=New Delhi&forecast_date=…&restaurant_type=PBCL` | ML vs legacy rules, side by side |
| `GET /ml/model-info` | active engine, training date/source, holdout scores, top features |
