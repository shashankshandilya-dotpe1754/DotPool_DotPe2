# DotPool

**Pan-India demand forecasting for restaurants** — an ML model that learns how
festivals, events, weather, sports and government regulations move order volume
and ticket size, city by city.

DotPool forecasts daily order volume and average order value for six
restaurant formats — **QSR, Fine Dining, PBCL, Casual Dining, Cloud Kitchen and
Cafe** — across **130 Indian cities**.

It combines a curated events database (260 entries — 163 festivals plus gazetted
holidays, sports fixtures, commercial events, government orders and emergency
crises, geo-scoped to city, state, zone or pan-India) with live Open-Meteo weather, and feeds both
into a LightGBM model that has learned how each factor moves demand for each
format in each city. A Diwali weekend lifts fine dining and suppresses pubs;
heavy monsoon rain crashes dine-in while cloud kitchens spike; a liquor ban guts
PBCL and leaves QSR untouched — the model learns these relationships from
history instead of relying on hand-written multipliers.

Ships as a **Streamlit dashboard** and a **FastAPI service**, with every forecast
explained: which events drove it, by how much, and how confident the model is.

Around the forecaster sit four more layers: **calibrated prediction intervals**,
**LLM-assisted event ingestion**, a **chat assistant** that answers by calling the
model, and **anomaly, drift and campaign-uplift** analysis.

---

## Run it locally

```bash
pip install -r requirements.txt

streamlit run About.py        # dashboard  → http://localhost:8501
uvicorn main:app --reload         # API + docs → http://localhost:8000/docs
```

The trained models ship in `ml/models/`, so both run out of the box. To retrain:

```bash
python -m ml.simulate             # rebuild the bootstrap training panel (~8s)
python -m ml.train                # train orders + AOV models (~3 min CPU)
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → **New app**, pick the
   repo and set the main file to **`About.py`**.
3. Deploy. `requirements.txt` and `packages.txt` (which installs `libgomp1` for
   LightGBM) are picked up automatically.

The FastAPI service in `main.py` is not part of the Streamlit deployment — host
it separately (Render, Railway, Fly.io, or any container host) if you need the
JSON API.

---

## What's inside

```
├── About.py                Streamlit app entry point (About page)
├── pages/
│   ├── 01_Restaurant_Impact.py   per-format demand view
│   ├── 03_Ask_DotPool.py     chat assistant over the model
│   ├── 03_Event_Review.py        approve LLM-proposed calendar events
│   └── 04_Model_Health.py        anomalies, drift, campaign uplift
├── main.py                 FastAPI service: events, weather, forecasts
├── data/
│   ├── events_db.py        260 Pan-India events, 2023–2026
│   ├── india_geo.py        130 cities with zone / state / coordinates
│   └── training/           training CSVs (generated, git-ignored)
├── utils/
│   ├── restaurant_impact.py   engine router: ML first, rules as fallback
│   ├── weather_service.py     Open-Meteo historical + forecast
│   └── auto_updater.py        daily 6 AM IST events refresh
└── ml/                     the forecasting model — see ml/README.md
    ├── features.py         93 features (events, weather, calendar, geo)
    ├── simulate.py         bootstrap training-data generator
    ├── train.py            time-split training: mean, AOV and quantile models
    ├── predict.py          inference + counterfactual explanations
    ├── llm.py              provider-agnostic LLM client (Anthropic / OpenAI)
    ├── event_ingest.py     LLM → validation → human review → events DB
    ├── chat_agent.py       tool-calling assistant over the forecast model
    ├── anomaly.py          conformal anomaly detection + drift monitoring
    ├── uplift.py           cross-fitted causal uplift for campaigns
    └── models/             trained boosters + validation metrics
```

## The model

Two LightGBM models — one on `log(orders_index)`, one on `log(AOV)` — trained on
93 features covering festivals by religion and region, geo-scoped local events,
sports fixtures, weather (temperature, rainfall, WMO codes, storm/heat/cold
flags), government orders (liquor bans, traffic restrictions, operating hours,
environmental and digital mandates), emergency crises, the Indian salary cycle,
festival proximity and long weekends, plus city, state, zone and tier.

Demand multipliers are no longer constants. Each is a **counterfactual run of
the model** — the same day scored with and without its events, with and without
its actual weather — so every number traces back to a learned relationship.
Individual events get their share of the day's lift by Shapley attribution,
which stays fair when a holiday, a festival and a cricket season all land
together.

| model | MAPE | R² |
|---|---|---|
| orders | **8.68%** | 0.935 |
| naive day-of-week × type baseline | 28.05% | 0.125 |
| AOV | 4.66% | 0.986 |

Validated on a strict time split (trained before 2026-04-13, scored after), so
the score measures forecasting rather than interpolation.

### Prediction intervals

Separate LightGBM quantile models (p10 / p50 / p90) give each day its own
uncertainty instead of stretching one global spread over the year — a wet
Tuesday gets a ±14% band, a monsoon thunderstorm ±21%.

Raw quantile models under-cover when extrapolating forward (they managed 62.8%
where 80% was claimed), so they are **conformalised**: a calibration constant
from held-out data widens both edges until coverage is exactly what it says.

| | coverage of held-out days |
|---|---|
| raw p10–p90 | 62.8% |
| conformalised (CQR) | **80.0%** ← target |

## Sales layer: product recommendations

`data/dotpe_products.py` holds DotPe's real price book (Rista, SOVA, Amello,
WhatsApp/WABA, Horizon growth retainers). `ml/product_advisor.py` is a
deterministic lookup — restaurant format + demand signal → which product to
pitch and at what real price — exposed to the chat assistant as the
`get_product_recommendation` tool. It never invents a product or a price; it
only maps an already-computed forecast onto the CSV price list. Ask
"Ask DotPool" something like *"What should the Bengaluru sales team push
for cloud kitchens around Diwali?"* and it will call `get_forecast` then
`get_product_recommendation` and answer with a business view, a pricing
opportunity line, and named SKUs — in the same format `get_forecast` already
uses for demand.

**Caveat carried over from the section above: the underlying orders/AOV model
is trained on the synthetic bootstrap panel, not DotPe's real transaction
history.** The product recommendation is deterministic and safe to trust; the
*size* of the opportunity (which signal fires, what index/AOV it shows) is
only as good as that training data. See "Making PAN-India numbers real" below.

### Making PAN-India numbers real

To get PAN-India order counts and AOV your sales team can rely on:
1. Export real daily orders + AOV by city × restaurant format from DotPe's
   warehouse, in the same columns as `data/training/synthetic_history.csv`.
2. Drop it in as `data/training/real_history.csv` (path is already wired in
   `ml/config.py` — `python -m ml.train` prefers it over synthetic data
   automatically).
3. Re-run `python -m ml.train` and check `ml/models/metrics.json` — MAPE/R²
   on real held-out data will replace the current synthetic-panel scores.
No code change is needed for this step; it's a data drop.

## Beyond forecasting

| layer | what it answers | needs |
|---|---|---|
| **Chat assistant** (`ml/chat_agent.py`) | "Should I add staff in Bengaluru this weekend?" | LLM key |
| **Event ingestion** (`ml/event_ingest.py`) | keeps the calendar correct without hand-maintenance | LLM key |
| **Anomaly & drift** (`ml/anomaly.py`) | "which days did we badly miss, and is the model degrading?" | your actuals |
| **Campaign uplift** (`ml/uplift.py`) | "did the promo cause orders, or discount people who were coming anyway?" | campaign log |

The chat assistant is **not allowed to produce a number**. It has four tools, all
of which call the trained model or the events database, and its job is to route
and narrate. Every figure in an answer traces to a `predict_impact()` call, shown
under "sources" in the UI.

Event ingestion has the same discipline in reverse: the LLM proposes structured
events, **hard Python validation** rejects unknown cities, impossible date spans,
wrong categories and duplicates, and a human approves what survives. Nothing an
LLM writes reaches the database unreviewed. This exists because the calendar was
hand-maintained, and nine 2026 festival dates were wrong — including Janmashtami,
which sat in the wrong month entirely.

Uplift uses a cross-fitted T-learner plus an IPW cross-check on out-of-fold
propensities, and reports overlap and confounding strength alongside the estimate.
On a test where the true campaign effect was +15%, the naive difference reported
+35% while the causal estimate landed at +19.7%. It says plainly when the data
cannot support a causal claim; a geo/switchback holdout remains the trustworthy
version.

```bash
python -m ml.anomaly     --csv data/training/actuals.csv
python -m ml.uplift      --csv data/training/campaigns.csv --segment city
python -m ml.event_ingest --year 2027          # then approve in the dashboard
```

## Language model configuration

Chat and event ingestion need a language model; everything else works without one.
Three options, no code changes to switch between them.

### Ollama — free, local, no API key

```bash
# install from ollama.com/download
ollama pull qwen2.5:7b        # tool-calling capable
ollama serve

export DEMANDPULSE_LLM_PROVIDER=ollama
export DEMANDPULSE_LLM_MODEL=qwen2.5:7b
```

Nothing leaves your machine. Pick a model that supports **tool calling** — qwen2.5,
llama3.1/3.2/3.3, mistral-nemo, command-r. Models without it can still do event
ingestion (plain JSON), but the chat assistant needs tools to reach the forecaster;
the sidebar warns you when the selected model probably can't.

> A **deployed** Streamlit app cannot reach Ollama running on your laptop — different
> machines. Ollama is for local runs, unless you expose it at a URL the app can reach
> and set `DEMANDPULSE_OLLAMA_URL`.

### OpenAI — GPT-4o / GPT-4 Turbo

```bash
export OPENAI_API_KEY=sk-...
export DEMANDPULSE_LLM_PROVIDER=openai
export DEMANDPULSE_LLM_MODEL=gpt-4o          # or gpt-4-turbo, gpt-4.1, gpt-4o-mini
```

Any OpenAI-compatible gateway (LM Studio, vLLM, Groq, Together, OpenRouter) works too —
also set `OPENAI_BASE_URL`.

### Anthropic — Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### On Streamlit Cloud

App → Settings → Secrets:

```toml
OPENAI_API_KEY = "sk-..."
DEMANDPULSE_LLM_PROVIDER = "openai"
DEMANDPULSE_LLM_MODEL = "gpt-4o"
```

With nothing configured, the two LLM pages explain how to set one up and the
forecasting app carries on unaffected. Provider auto-detection order: Anthropic key →
OpenAI key → a reachable Ollama server.

## API endpoints

| endpoint | what it does |
|---|---|
| `GET /forecast/day?city=Bengaluru` | one day, all six formats, live weather attached |
| `GET /forecast/range?city=Mumbai&start_date=…&end_date=…` | up to 120 days |
| `GET /forecast/compare?city=New Delhi&forecast_date=…&restaurant_type=PBCL` | ML vs rules, side by side |
| `GET /ml/model-info` | active engine, training date, holdout scores, interval coverage |
| `POST /chat` | ask a demand question in plain language (needs an LLM key) |
| `GET /llm/status` | whether the LLM layer is configured |
| `GET /ingest/pending` · `POST /ingest/propose` · `POST /ingest/approve` | event review queue |
| `GET /events/today`, `/events/range`, `/events/category/{c}` | events database |
| `GET /weather/city/{city}` | historical + 7-day forecast |
| `GET /calendar/{date}?city=…` | full day context card |
