"""
ml/chat_agent.py
================
"Ask DotPool" — a merchant-facing chat layer over the forecasting model.

THE ONE RULE
------------
The LLM does not forecast. It cannot do arithmetic on demand, cannot estimate a
percentage, cannot guess what Diwali does to a pub. It has exactly four tools —
all of which call the trained LightGBM model or the events database — and its
job is to pick the right one, then explain what came back in plain language.

Everything numeric in an answer traces to a `predict_impact()` call. If the
model has no answer, the correct response is to say so, not to improvise one.

Usage
-----
    from ml.chat_agent import answer
    result = answer("What should we pitch to PBCL outlets in Mumbai next weekend?")
    print(result["text"])          # narrated answer
    print(result["tool_trace"])    # exactly which forecasts backed it

The Streamlit page (pages/02_Ask_DotPool.py) wraps this with chat history.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from data.india_geo import CITIES, CITY_LIST
from ml.config import RESTAURANT_TYPES
from ml.llm import LLMUnavailable, available, chat

# Was 5 — a slow local (Ollama) model pays for every round trip, and a normal
# question here only ever needs get_forecast [+ get_forecast_range] then
# get_product_recommendation, so 3 covers real usage without the worst-case
# latency of a model that wanders through extra rounds.
MAX_TOOL_ROUNDS = 3

SYSTEM = """You are the language model running inside DotPool — DotPe's internal
Sales & Business Strategy Intelligence tool. You are not a generic chatbot bolted
onto the app; you are answering as DotPool itself, so speak with the app's context
already in hand, not as an outside assistant guessing at what it might contain.

WHAT DOTPOOL IS, so you can talk about the app itself accurately if asked:
- Home: today's demand snapshot for a chosen city, a 7-day weather-impact strip,
  and every active/upcoming festival, holiday, sports fixture and government order
  with a per-restaurant-format impact and a Sales Playbook (which DotPe product to
  pitch, and why) for each one.
- Restaurant Impact: the same forecast broken out per restaurant format over a
  chosen date range, with its own Sales Playbook per format.
- Ask DotPool (this conversation): plain-language Q&A over the same forecasting
  model and product/price book, for the Sales team.
- Two things behind those pages: (1) a demand model — either a trained LightGBM
  model or, if that isn't loaded, a hand-typed rules engine — driven by a real,
  hand-compiled events calendar and live weather from Open-Meteo; and (2) a fixed
  DotPe product/price catalogue (SOVA, Amello, Rista POS/CRM/Reservation, QR
  Ordering, WhatsApp SME/Enterprise, Horizon, and a bundle) it recommends from.
- Be straight about what's real and what isn't if asked: the events calendar and
  the weather are real. The baseline order-index and AOV numbers the forecast
  scales from are hand-set reference constants (or, for the ML path, a model
  trained on a synthetic simulated panel), not derived from DotPe's actual
  transaction history yet. Restaurant formats are applied uniformly to every
  city — there is no merchant-level data behind which formats actually exist
  where. Don't oversell this as validated historical accuracy; say plainly that
  the forecast shape is a reasonable working model, not measured fact, if pressed.

DIVISION OF LABOUR — this is the most important rule:
- Every number — a demand index, a percentage change, an AOV, a product price —
  must come from a tool call in this conversation. Never estimate, interpolate,
  or reason your way to a figure. If a tool didn't give you a number, say you
  don't have it. Trivial arithmetic on numbers you already have (a sum, a
  difference) is fine; inventing a new figure is not.
- Everything else — sales strategy, objection handling, how to sequence a pitch,
  how to read a competitor's move, general judgement — is yours to reason through
  using your own knowledge, exactly as you would for anyone. The only requirement
  is that it stays grounded in DotPe's actual context: these six restaurant
  formats, these specific DotPe products and prices, the Indian F&B/restaurant-
  tech market this is operating in — not generic, could-apply-to-any-SaaS advice.
  A good answer reads like it was written by someone who actually knows this
  product line and this market, not a template with DotPe's name dropped in.

You have the trained model behind these tools. Use them freely — call several if the
question spans days, cities or restaurant formats.

How to answer:
- Lead with the sales call to action, not the raw data. "Pitch SOVA to QSR this
  week — delivery demand is up 25% on Navratri" beats "the index is 125".
- ALWAYS call get_product_recommendation after get_forecast (or get_forecast_range)
  when a restaurant format is in scope — which is by default, since this tool exists
  to drive pitches. Fold its output in as a short "Business view", a "Pricing
  opportunity" line, and 1-3 named products with their real price.
- Quote the forecast as a percentage change versus a normal day, and give the 80%
  range when the pitch timing depends on the downside.
- Name the drivers the tool returned (which festival, which weather) rather than
  speaking generally about seasonality.
- Write like a sharp, direct colleague, not a form letter — full sentences, plain
  language, no boilerplate throat-clearing ("I'd be happy to help..."). Get to the
  point in the first line, then give the reasoning underneath. Use short paragraphs
  or a few bullets when that's clearer than prose; don't force structure where a
  couple of sentences would do. Match the length of the answer to the question —
  a quick lookup gets a quick answer, a genuine strategy question gets the room to
  actually reason through trade-offs.
- Default shape for a plain "what's demand like / what should we pitch" question
  (one city, one format, one date or short range) — don't compress this into a
  single line even if you're capable of it. Use it as a floor, not a ceiling:
    <City> — <Restaurant format>
    Demand line: the % change vs. a normal day and the Demand Index.
    Drivers: the named events/weather behind it, in a sentence, not a keyword list.
    Business view: 1-2 sentences of what this means for the sales rep to do.
    Pricing opportunity: the AOV band (High/Medium/Low) and the ₹ figure.
    Recommended products: each named product, its real price, and one sentence
    on why it fits this specific driver — not the generic catalogue blurb.
  Only compress below this shape for a trivial yes/no or a pure lookup with no
  sales angle ("what's the AOV baseline for Cafe" needs one line, not a template).
- Broader sales-strategy questions are in scope too, not just single forecast
  lookups — objection handling, how to sequence a pitch across a multi-outlet
  account, how to prioritise a city list, how to frame a renewal conversation,
  general product positioning against competitors in Indian restaurant-tech.
  Reason through these like a knowledgeable colleague would, staying inside the
  DIVISION OF LABOUR above: any demand number still has to come from a tool, but
  the judgement, structure and recommendation around it is yours to build — and
  should sound like it, not like it was copy-pasted from a forecast tool.
- If a file or image has been attached, use it as real context — read what it
  says or shows and let it shape the answer (e.g. a competitor's menu, a
  screenshot of a merchant's dashboard, a call transcript) — but never let it be
  the source of a demand figure; pull those from the tools as always.
- If a question is outside what the tools cover — a named merchant's own sales,
  competitor data, menu pricing — say so plainly and reframe it at the restaurant-
  format level you can actually answer.

Context you can rely on:
- The forecast index is scaled so 100 = a typical quiet weekday for that city and
  restaurant format. 150 means about 50% more orders than such a day.
- Restaurant formats: QSR, Fine Dining, PBCL (pub/bar/cafe/lounge), Casual Dining,
  Cloud Kitchen, Cafe.
- The models are currently trained on a synthetic bootstrap panel, not real sales
  history. If someone asks how accurate this is, say so honestly and mention the
  holdout MAPE from get_model_info.
- Today's date is {today}.
"""

TOOLS = [
    {
        "name": "get_forecast",
        "description": ("Forecast for ONE city on ONE date. Returns predicted index "
                        "(100 = typical quiet weekday), percent change, 80% prediction "
                        "interval, active events with each one's contribution, weather "
                        "effect, predicted average order value, and confidence. Call it "
                        "once per restaurant format you need."),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": f"One of the supported cities, e.g. {CITY_LIST[:5]}"},
                "forecast_date": {"type": "string", "description": "YYYY-MM-DD"},
                "restaurant_type": {"type": "string", "enum": RESTAURANT_TYPES,
                                    "description": "Omit to get all six formats"},
            },
            "required": ["city", "forecast_date"],
        },
    },
    {
        "name": "get_forecast_range",
        "description": ("Forecast every day between two dates for one city. Use this for "
                        "'next week', 'this month', 'when is my busiest day' questions. "
                        "Max 60 days."),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "restaurant_type": {"type": "string", "enum": RESTAURANT_TYPES},
            },
            "required": ["city", "start_date", "end_date"],
        },
    },
    {
        "name": "list_events",
        "description": ("Events active in a date range for a city — festivals, holidays, "
                        "sports, government orders, crises. Use when asked WHAT is "
                        "happening rather than how much demand changes."),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "city": {"type": "string"},
                "category": {"type": "string",
                             "description": "Optional filter, e.g. Festival, Government Order"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_model_info",
        "description": ("How the live model was trained and how accurate it is on held-out "
                        "data. Use for questions about trust, accuracy or methodology."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_product_recommendation",
        "description": ("Maps an already-computed forecast onto DotPe's real product/price "
                        "list (Rista, SOVA, Amello, WhatsApp/WABA, Horizon growth retainers) "
                        "and returns a 'business view', a 'pricing opportunity' line, and named "
                        "products with real prices to pitch. Call this AFTER get_forecast when the "
                        "question is about sales strategy or what to sell, not just the raw forecast."),
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_type": {"type": "string", "enum": RESTAURANT_TYPES},
                "signal": {"type": "string",
                           "description": "the 'signal' field from a get_forecast result, e.g. high_up"},
                "pct_change": {"type": "number", "description": "the pct_change_vs_normal_day from get_forecast"},
                "predicted_aov": {"type": "number", "description": "predicted_aov_rupees from get_forecast, if present"},
            },
            "required": ["restaurant_type", "signal", "pct_change"],
        },
    },
]


# ── Tool implementations (all call the real model) ────────────────────────────
def _slim(p: dict) -> dict:
    """Trim a prediction to what the LLM needs — keeps context small and focused."""
    return {
        "date": p["date"], "weekday": p["weekday"], "city": p["city"],
        "restaurant_type": p["restaurant_type"],
        "predicted_index": p["predicted_index"],
        "pct_change_vs_normal_day": p["pct_change"],
        "interval_80": p.get("prediction_interval_80"),
        "signal": p["signal"], "confidence": p["confidence"],
        "predicted_aov_rupees": p.get("predicted_aov"),
        "event_effect_pct": round((p["event_multiplier"] - 1) * 100, 1),
        "weather_effect_pct": round((p["weather_multiplier"] - 1) * 100, 1),
        "weather_note": p.get("weather_note"),
        "events": p.get("factors", [])[:6],
        "engine": p.get("engine"),
    }


def _resolve_city(city: str) -> str:
    if city in CITIES:
        return city
    lower = {c.lower(): c for c in CITY_LIST}
    if city.lower() in lower:
        return lower[city.lower()]
    hits = [c for c in CITY_LIST if city.lower() in c.lower()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"Unknown city {city!r}. Supported cities include: {CITY_LIST[:12]} …")


def tool_get_forecast(city: str, forecast_date: str, restaurant_type: str = None) -> dict:
    from utils.restaurant_impact import predict_impact
    from utils.weather_service import get_forecast_weather, get_historical_weather

    city = _resolve_city(city)
    d = date.fromisoformat(forecast_date)
    geo = CITIES[city]
    wx = {}
    try:
        rows = (get_forecast_weather(geo["lat"], geo["lon"], days=8)
                if d >= date.today() else
                get_historical_weather(geo["lat"], geo["lon"], forecast_date, forecast_date))
        for r in rows:
            if r.get("date") == forecast_date and "error" not in r:
                wx = r
    except Exception:
        pass
    types = [restaurant_type] if restaurant_type else RESTAURANT_TYPES
    out = [
        _slim(predict_impact(city, d, t,
                             temp_max=wx.get("temp_max_c"),
                             precip_mm=wx.get("precipitation_mm"),
                             weather_code=wx.get("weather_code"),
                             temp_min=wx.get("temp_min_c")))
        for t in types
    ]
    return {"city": city, "date": forecast_date,
            "weather_used": bool(wx), "forecasts": out}


def tool_get_forecast_range(city: str, start_date: str, end_date: str,
                            restaurant_type: str = None) -> dict:
    from utils.restaurant_impact import predict_date_range

    city = _resolve_city(city)
    s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if (e - s).days > 60:
        e = s + timedelta(days=60)
    res = predict_date_range(city, s, e)
    types = [restaurant_type] if restaurant_type else RESTAURANT_TYPES
    trimmed = {
        t: [{"date": p["date"], "weekday": p["weekday"][:3],
             "predicted_index": p["predicted_index"],
             "pct_change_vs_normal_day": p["pct_change"],
             "signal": p["signal"],
             "top_event": (p.get("factors") or [None])[0]}
            for p in res.get(t, [])]
        for t in types if t in res
    }
    return {"city": city, "start_date": start_date, "end_date": e.isoformat(),
            "series": trimmed}


def tool_list_events(start_date: str, end_date: str, city: str = None,
                     category: str = None) -> dict:
    from data.events_db import get_events_by_range

    city = _resolve_city(city) if city else None
    evs = get_events_by_range(start_date, end_date, category=category, city=city)
    return {"count": len(evs), "events": [
        {"name": e["name"], "category": e["category"],
         "subcategory": e.get("subcategory"), "start_date": e["start_date"],
         "end_date": e["end_date"], "scope": e.get("scope"),
         "expected_direction": e.get("impact_on_demand"),
         "why": e.get("description")}
        for e in evs[:40]
    ]}


def tool_get_model_info() -> dict:
    from utils.restaurant_impact import engine_status

    return engine_status()


def tool_get_product_recommendation(restaurant_type: str, signal: str,
                                    pct_change: float, predicted_aov: float = None) -> dict:
    from ml.product_advisor import recommend

    return recommend(restaurant_type, signal, pct_change, predicted_aov)


DISPATCH = {
    "get_forecast": tool_get_forecast,
    "get_forecast_range": tool_get_forecast_range,
    "list_events": tool_list_events,
    "get_model_info": lambda **kw: tool_get_model_info(),
    "get_product_recommendation": tool_get_product_recommendation,
}


# ── Agent loop ────────────────────────────────────────────────────────────────
def answer(question: str, history: list = None, attachments: list = None) -> dict:
    """Answer one question, running tool calls until the model is done.

    attachments: optional list of dicts —
      {"kind": "image", "name": str, "media_type": str, "data": base64-str}
      {"kind": "text",  "name": str, "text": str}
    Images are only actually shown to the model when the Anthropic provider is
    active (Claude has native vision here; the other providers aren't wired
    for it). Text-extractable attachments (csv/txt/md/json) are folded in as
    plain context for every provider. Attachments inform judgement — what a
    menu, a sales deck, or a screenshot shows — but never supply a number;
    every figure in the reply still has to come from a tool call.
    """
    if not available():
        from ml.llm import status
        raise LLMUnavailable(status()["hint"])

    from ml.llm import provider
    user_content = question
    if attachments:
        text_notes = [f"--- {a['name']} ---\n{a['text']}" for a in attachments
                       if a.get("kind") == "text" and a.get("text")]
        images = [a for a in attachments if a.get("kind") == "image"]
        text_block = question
        if text_notes:
            text_block += "\n\n[Attached file content]\n" + "\n\n".join(text_notes)
        if images and provider() == "anthropic":
            blocks = [{"type": "image",
                      "source": {"type": "base64", "media_type": a["media_type"], "data": a["data"]}}
                     for a in images]
            blocks.append({"type": "text", "text": text_block})
            user_content = blocks
        else:
            if images:
                names = ", ".join(a["name"] for a in images)
                text_block += (f"\n\n[{len(images)} image(s) attached: {names} — this "
                              f"provider doesn't have vision wired in here, so describe "
                              f"what's relevant in words if you need me to act on them.]")
            user_content = text_block

    messages = list(history or []) + [{"role": "user", "content": user_content}]
    system = SYSTEM.format(today=date.today().isoformat())
    trace = []

    for _ in range(MAX_TOOL_ROUNDS):
        resp = chat(messages, system=system, tools=TOOLS)
        if not resp["tool_calls"]:
            return {"text": resp["text"], "tool_trace": trace, "messages": messages}

        # Record the assistant turn, then execute every requested tool.
        messages.append({"role": "assistant", "content": _assistant_block(resp)})
        results = []
        for call in resp["tool_calls"]:
            fn = DISPATCH.get(call["name"])
            try:
                out = fn(**call["input"]) if fn else {"error": f"no such tool {call['name']}"}
                err = None
            except Exception as exc:                                   # noqa: BLE001
                out, err = {"error": str(exc)}, str(exc)
            trace.append({"tool": call["name"], "input": call["input"], "error": err})
            results.append((call, out))
        messages.append({"role": "user", "content": _result_block(results)})

    return {"text": "I could not finish that lookup — try narrowing the question to "
                    "one city and a shorter date range.",
            "tool_trace": trace, "messages": messages}


def _assistant_block(resp):
    """Provider-shaped assistant turn carrying the tool calls."""
    from ml.llm import provider

    p = provider()
    if p == "ollama":
        # Ollama takes plain strings — describe the calls so the next turn has context.
        calls = "; ".join(f"{c['name']}({json.dumps(c['input'])})"
                          for c in resp["tool_calls"])
        return (resp["text"] or "") + (f"\n[calling {calls}]" if calls else "")
    if p == "anthropic":
        blocks = ([{"type": "text", "text": resp["text"]}] if resp["text"] else []) + [
            {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
            for c in resp["tool_calls"]
        ]
        return blocks
    return resp["text"] or ""


def _result_block(results):
    from ml.llm import provider

    if provider() == "anthropic":
        return [{"type": "tool_result", "tool_use_id": c["id"],
                 "content": json.dumps(out, default=str)[:12000]}
                for c, out in results]
    # OpenAI-compatible and Ollama both accept the results as a plain user turn.
    return ("TOOL RESULTS — use these figures, do not invent others:\n" + "\n".join(
        f"{c['name']}({json.dumps(c['input'])}) -> {json.dumps(out, default=str)[:6000]}"
        for c, out in results))


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What should we pitch to cafes in Pune next weekend?"
    try:
        r = answer(q)
        print(r["text"])
        print("\n--- tools used ---")
        for t in r["tool_trace"]:
            print(" ", t["tool"], t["input"], "ERROR: " + t["error"] if t["error"] else "")
    except LLMUnavailable as exc:
        print(exc)
