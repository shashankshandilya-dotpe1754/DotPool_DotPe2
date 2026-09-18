"""
ml/event_ingest.py
==================
LLM-assisted event ingestion — the fix for the weakest part of DotPool.

The events database is what the whole forecast rests on, and it was maintained
by hand. That is exactly how Janmashtami 2026 ended up dated 16 August (2025's
date, copy-pasted) and how nine other 2026 festival dates drifted. A model
cannot see a festival that the calendar puts in the wrong month.

This module reads unstructured sources — panchang pages, state gazette
notifications, municipal traffic orders, sports schedules, local news — and asks
an LLM to emit *structured candidate events* in the exact schema of
`data/events_db.py`.

THE LLM NEVER WRITES TO THE DATABASE.
Every candidate goes through:

    fetch → LLM extraction → hard validation → data/pending_events.json → human approval → merge

`validate_candidates()` is deliberately strict and runs entirely in Python, so a
hallucinated city, an impossible date span, a wrong category or a duplicate of
an event already in the DB is rejected before a human ever sees it. Approval is
a person clicking Approve in the Event Review page (or calling `approve_ids()`).

Usage
-----
    python -m ml.event_ingest --year 2027                  # propose next year's calendar
    python -m ml.event_ingest --url https://... --url ...  # extract from specific pages
    python -m ml.event_ingest --text "$(cat notice.txt)"   # extract from pasted text
    python -m ml.event_ingest --list                       # show the review queue
    python -m ml.event_ingest --approve FE200,FE201        # merge approved candidates
    python -m ml.event_ingest --reject FE202               # drop candidates
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime

from data.events_db import EVENTS_DB
from data.india_geo import CITY_LIST, STATE_LIST, ZONE_LIST
from ml.config import EVENT_CATEGORIES, EVENT_SUBCATEGORIES, IMPACT_ORDINAL, PENDING_EVENTS_JSON
from ml.llm import LLMUnavailable, available, complete_json

MAX_SPAN_DAYS = 400
VALID_SCOPES = {"pan_india", "zone", "state", "city"}

SYSTEM = """You extract structured calendar events for an Indian restaurant demand-forecasting system.

You return ONLY a JSON array. No prose, no markdown fence, no commentary.

Each element must be an object with exactly these keys:
  category        one of: {categories}
  subcategory     one of: {subcategories}
  name            short human name INCLUDING the year, e.g. "Janmashtami 2027"
  start_date      "YYYY-MM-DD"
  end_date        "YYYY-MM-DD"  (same as start_date for single-day events)
  scope           one of: pan_india | zone | state | city
  zones           list of zone names, [] unless scope is "zone"
  states          list of state names, [] unless scope is "state"
  cities          list of city names, [] unless scope is "city"
  description     one sentence on WHY it moves restaurant demand (which formats, which direction)
  impact_on_demand  one of: {impacts}
  source          where the date came from
  tags            short lowercase keyword list

Rules that matter:
- Only include events you are confident about. Omitting a doubtful event is far
  better than guessing its date. Accuracy of `start_date` is the single most
  important field.
- Use the correct year's date. Hindu, Islamic and regional festivals move every
  year — never copy a previous year's date forward.
- Islamic dates depend on moon sighting; mark them in `description` as approximate.
- scope: a festival observed nationally is pan_india. One observed mainly in
  certain states is scope "state" with those states listed. Zones are:
  {zones}
- Prefer separate entries for a festival and its gazetted public holiday when
  both exist, since they affect demand differently.
- Multi-day festivals get a real span (Navratri is nine days, not one).

Valid state names: {states}
Valid zone names: {zones}
"""

USER_YEAR = """Produce the complete Indian festival and public-holiday calendar for {year} that is
relevant to restaurant demand: major Hindu, Muslim, Christian, Sikh, Jain and Buddhist
festivals, gazetted national holidays, and the significant regional New Years and
harvest festivals (Pongal, Bihu, Poila Boishakh, Onam, Ugadi, Vishu, Puthandu,
Baisakhi, Chhath, Bathukamma).

Include demand-suppressing periods too — Pitru Paksha, Navratri and Ramadan fasting
windows, Muharram — not only celebration days.

Today is {today}. Return the JSON array only."""

USER_TEXT = """Extract every event relevant to restaurant demand from the source material below.
Ignore anything with no plausible effect on restaurant footfall, delivery volume or
operating hours.

Today is {today}.

--- SOURCE MATERIAL ---
{material}
--- END ---

Return the JSON array only."""


def _system() -> str:
    return SYSTEM.format(
        categories=" | ".join(EVENT_CATEGORIES),
        subcategories=" | ".join(EVENT_SUBCATEGORIES),
        impacts=" | ".join(IMPACT_ORDINAL.keys()),
        zones=", ".join(ZONE_LIST),
        states=", ".join(STATE_LIST),
    )


# ── Source fetching ───────────────────────────────────────────────────────────
def fetch_url(url: str, max_chars: int = 20000) -> str:
    """Plain-text of a page. Kept dependency-light on purpose."""
    import re

    import requests

    r = requests.get(url, timeout=20, headers={"User-Agent": "DotPool/1.0"})
    r.raise_for_status()
    html = r.text
    html = re.sub(r"(?is)<(script|style|nav|footer|header).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


# ── Validation (pure Python — the LLM's output is never trusted) ──────────────
def validate_candidates(cands: list, existing: list = None) -> tuple:
    """Return (accepted, rejected) where rejected carries a human-readable reason."""
    existing = EVENTS_DB if existing is None else existing
    known_names = {e["name"].strip().lower() for e in existing}
    known_spans = {(e["name"].strip().lower(), e["start_date"]) for e in existing}
    ok, bad = [], []

    for c in cands if isinstance(cands, list) else []:
        def reject(reason):
            bad.append({"candidate": c, "reason": reason})

        if not isinstance(c, dict):
            reject("not an object"); continue
        missing = [k for k in ("category", "name", "start_date", "end_date") if not c.get(k)]
        if missing:
            reject(f"missing field(s): {missing}"); continue
        if c["category"] not in EVENT_CATEGORIES:
            reject(f"unknown category {c['category']!r}"); continue
        sub = c.get("subcategory") or "*"
        if sub != "*" and sub not in EVENT_SUBCATEGORIES:
            reject(f"unknown subcategory {sub!r}"); continue
        try:
            s = date.fromisoformat(c["start_date"])
            e = date.fromisoformat(c["end_date"])
        except Exception:
            reject("unparseable date"); continue
        if e < s:
            reject("end_date before start_date"); continue
        if (e - s).days > MAX_SPAN_DAYS:
            reject(f"span longer than {MAX_SPAN_DAYS} days"); continue
        if not (2023 <= s.year <= date.today().year + 3):
            reject(f"start year {s.year} outside supported range"); continue
        scope = c.get("scope", "pan_india")
        if scope not in VALID_SCOPES:
            reject(f"invalid scope {scope!r}"); continue
        for field, valid, key in (("zones", ZONE_LIST, "zone"),
                                  ("states", STATE_LIST, "state"),
                                  ("cities", CITY_LIST, "city")):
            vals = c.get(field) or []
            unknown = [v for v in vals if v not in valid]
            if unknown:
                reject(f"unknown {field}: {unknown}"); break
            if scope == key and not vals:
                reject(f"scope {scope!r} but {field} is empty"); break
        else:
            impact = c.get("impact_on_demand", "neutral")
            if impact not in IMPACT_ORDINAL:
                reject(f"invalid impact_on_demand {impact!r}"); continue
            nm = c["name"].strip().lower()
            if nm in known_names or (nm, c["start_date"]) in known_spans:
                reject("already in the events database"); continue
            if not (c.get("description") or "").strip():
                reject("empty description"); continue
            ok.append(c)
    return ok, bad


# ── Review queue ──────────────────────────────────────────────────────────────
def _next_ids(n: int, prefix: str = "FE") -> list:
    used = {int(e["id"][2:]) for e in EVENTS_DB if e["id"].startswith(prefix)
            and e["id"][2:].isdigit()}
    used |= {int(e["id"][2:]) for e in load_pending() if e["id"].startswith(prefix)
             and e["id"][2:].isdigit()}
    start = (max(used) if used else 0) + 1
    return [f"{prefix}{start + i:03d}" for i in range(n)]


PREFIX = {"Festival": "FE", "Government Holiday": "GH", "Public Event": "PE",
          "Commercial Event": "CE", "Sports Event": "SP", "Weather Event": "WS",
          "Government Order": "GO", "Emergency Crisis": "EC"}


def load_pending() -> list:
    if not os.path.exists(PENDING_EVENTS_JSON):
        return []
    try:
        return json.load(open(PENDING_EVENTS_JSON, encoding="utf-8"))
    except Exception:
        return []


def save_pending(items: list):
    json.dump(items, open(PENDING_EVENTS_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def queue_candidates(accepted: list, origin: str) -> list:
    """Assign ids, stamp provenance, append to the review queue."""
    pending = load_pending()
    queued = []
    for c in accepted:
        pre = PREFIX.get(c["category"], "FE")
        cid = _next_ids(1, pre)[0]
        ev = {
            "id": cid, "category": c["category"],
            "subcategory": c.get("subcategory") or "*", "name": c["name"].strip(),
            "start_date": c["start_date"], "end_date": c["end_date"],
            "scope": c.get("scope", "pan_india"), "zones": c.get("zones") or [],
            "states": c.get("states") or [], "cities": c.get("cities") or [],
            "description": c["description"].strip(),
            "impact_on_demand": c.get("impact_on_demand", "neutral"),
            "source": c.get("source") or origin, "tags": c.get("tags") or [],
            "_review": {"origin": origin,
                        "proposed_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "pending"},
        }
        pending.append(ev)
        queued.append(ev)
    save_pending(pending)
    return queued


def approve_ids(ids: list) -> dict:
    """Merge approved candidates into data/events_db.py and drop them from the queue."""
    from utils.auto_updater import save_events_to_disk

    pending = load_pending()
    keep, approved = [], []
    for ev in pending:
        if ev["id"] in ids:
            clean = {k: v for k, v in ev.items() if not k.startswith("_")}
            approved.append(clean)
        else:
            keep.append(ev)
    if approved:
        merged = list(EVENTS_DB) + approved
        merged.sort(key=lambda e: (e["category"], e["start_date"]))
        save_events_to_disk(merged)
        EVENTS_DB.extend(approved)          # live list, so the app sees them now
    save_pending(keep)
    return {"approved": [e["id"] for e in approved], "remaining": len(keep),
            "note": "retrain with `python -m ml.train` so the model sees the new events"}


def reject_ids(ids: list) -> dict:
    pending = load_pending()
    keep = [e for e in pending if e["id"] not in ids]
    save_pending(keep)
    return {"rejected": [i for i in ids], "remaining": len(keep)}


# ── Extraction entry points ───────────────────────────────────────────────────
def propose_for_year(year: int) -> dict:
    prompt = USER_YEAR.format(year=year, today=date.today().isoformat())
    return _extract(prompt, origin=f"llm:calendar-{year}")


def propose_from_text(material: str, origin: str = "llm:text") -> dict:
    prompt = USER_TEXT.format(material=material[:40000], today=date.today().isoformat())
    return _extract(prompt, origin=origin)


def propose_from_urls(urls: list) -> dict:
    chunks = []
    for u in urls:
        try:
            chunks.append(f"\n### SOURCE: {u}\n{fetch_url(u)}")
        except Exception as exc:                                  # noqa: BLE001
            chunks.append(f"\n### SOURCE: {u}\n(could not fetch: {exc})")
    return propose_from_text("\n".join(chunks), origin="llm:url:" + ",".join(urls)[:200])


def _extract(prompt: str, origin: str) -> dict:
    if not available():
        raise LLMUnavailable(
            "No LLM key configured — set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(environment or Streamlit secrets)."
        )
    raw = complete_json(prompt, system=_system(), max_tokens=8000)
    accepted, rejected = validate_candidates(raw)
    queued = queue_candidates(accepted, origin)
    return {
        "proposed": len(raw) if isinstance(raw, list) else 0,
        "queued_for_review": len(queued),
        "rejected_by_validation": rejected,
        "candidates": queued,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, help="propose a full calendar for this year")
    ap.add_argument("--url", action="append", default=[], help="source page (repeatable)")
    ap.add_argument("--text", help="raw text to extract events from")
    ap.add_argument("--list", action="store_true", help="show the review queue")
    ap.add_argument("--approve", help="comma-separated ids to merge into the DB")
    ap.add_argument("--reject", help="comma-separated ids to drop")
    a = ap.parse_args()

    if a.list:
        pending = load_pending()
        if not pending:
            print("review queue is empty")
            return
        print(f"{len(pending)} candidate(s) awaiting review:\n")
        for e in pending:
            print(f"  {e['id']}  {e['start_date']} → {e['end_date']}  "
                  f"[{e['category']}/{e['subcategory']}]  {e['name']}")
            print(f"        {e['description'][:100]}")
        return
    if a.approve:
        print(json.dumps(approve_ids([i.strip() for i in a.approve.split(",")]), indent=2))
        return
    if a.reject:
        print(json.dumps(reject_ids([i.strip() for i in a.reject.split(",")]), indent=2))
        return

    if a.year:
        res = propose_for_year(a.year)
    elif a.url:
        res = propose_from_urls(a.url)
    elif a.text:
        res = propose_from_text(a.text)
    else:
        ap.print_help()
        return

    print(f"proposed {res['proposed']}, queued {res['queued_for_review']} for review")
    for r in res["rejected_by_validation"]:
        print(f"  rejected: {r['reason']}  ({r['candidate'].get('name', '?')})")
    for c in res["candidates"]:
        print(f"  {c['id']}  {c['start_date']} → {c['end_date']}  {c['name']}")
    print("\nReview them in the dashboard (Event Review page) or with "
          "`python -m ml.event_ingest --list`, then --approve the ones you trust.")


if __name__ == "__main__":
    _main()
