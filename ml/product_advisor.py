"""
ml/product_advisor.py
======================
Turns a DotPool forecast into a sales recommendation for DotPe's own
sales team — which product to pitch, to which restaurant format, and why.

Same discipline as the rest of ml/: this is a deterministic lookup over
data/dotpe_products.py (real SKUs, real prices), not something an LLM
free-associates. `ml/chat_agent.py` exposes it as a tool so "Business view"
and "Pricing opportunity" in a chat answer trace back to this function,
exactly like every demand figure traces back to predict_impact().

It does NOT forecast anything itself — it takes a forecast that has already
been produced (predicted_index, pct_change, signal, predicted_aov) and maps
it onto the price book.
"""

from __future__ import annotations

from data.dotpe_products import PRODUCTS, TAGS, FAMILY_REASONS, horizon_tier_for
from ml.config import BASELINE_AOV

# Which product tags fit which restaurant format when demand is *up*
_SURGE_TAG_BY_TYPE = {
    "QSR":           "surge_delivery",
    "Cloud Kitchen":  "surge_delivery",
    "Cafe":           "surge_delivery",
    "Casual Dining":  "surge_dine_in",
    "Fine Dining":    "surge_fine_dining",
    "PBCL":           "surge_bar",
}


def _aov_band(restaurant_type: str, predicted_aov: float | None) -> str:
    """High / Medium / Low pricing opportunity, benchmarked against this
    format's own baseline AOV (not a fixed rupee cutoff — a ₹500 cloud
    kitchen order and a ₹500 QSR order mean different things)."""
    baseline = BASELINE_AOV.get(restaurant_type)
    if not predicted_aov or not baseline:
        return "Unknown"
    ratio = predicted_aov / baseline
    if ratio >= 1.10:
        return "High"
    if ratio <= 0.90:
        return "Low"
    return "Medium"


def recommend(restaurant_type: str, signal: str, pct_change: float,
              predicted_aov: float | None = None,
              event_categories: list[str] | None = None) -> dict:
    """
    signal: one of the values utils/restaurant_impact.py already produces
            (very_high_up, high_up, slight_up, neutral, slight_down, low, very_low)
    Returns: business_view (str), pricing_opportunity (str),
             aov_band (High/Medium/Low), recommended_products (list of dicts).
    """
    event_categories = event_categories or []
    is_up = signal in ("very_high_up", "high_up", "slight_up")
    is_soft = signal in ("low", "very_low", "slight_down")

    if is_up:
        posture = ("This is a strong sales opportunity. Consider increasing "
                   "promotional visibility during the high-demand window and "
                   "using the stronger demand environment to test higher-value bundles.")
        tag = _SURGE_TAG_BY_TYPE.get(restaurant_type, "always_on_growth")
    elif is_soft:
        posture = ("Demand is soft here — this is a retention play, not a push "
                   "play. Lean on win-back and loyalty tooling to protect the "
                   "customer base until the window passes, rather than pitching "
                   "growth spend.")
        tag = "defend_retention"
    else:
        posture = ("Demand is close to normal. A steady always-on growth "
                   "motion (segmentation, automated campaigns) fits better "
                   "than a one-off festival push.")
        tag = "always_on_growth"

    product_keys = TAGS.get(tag, TAGS["always_on_growth"])
    products = [PRODUCTS[k] for k in product_keys if k in PRODUCTS]

    aov_band = _aov_band(restaurant_type, predicted_aov)
    if predicted_aov:
        pricing_line = (f"{aov_band} — estimated AOV is \u20b9{predicted_aov:,.0f}.")
    else:
        pricing_line = "Unknown — no AOV forecast was returned for this call."

    tier_name, tier = horizon_tier_for(signal)
    horizon_line = (f"Horizon {tier_name} (\u20b9{tier['retainer']:,}/mo, "
                    f"{tier['outlets_included']} outlet(s) included) — {tier['pitch']}"
                    if tier["retainer"] else
                    f"Horizon {tier_name} (custom pricing) — {tier['pitch']}")

    return {
        "restaurant_type": restaurant_type,
        "signal": signal,
        "business_view": posture,
        "pricing_opportunity": pricing_line,
        "aov_band": aov_band,
        "recommended_products": [
            {"name": p["name"], "platform": p["platform"], "family": p.get("family", p["platform"]),
             "price": p["price"], "unit": p["unit"], "why": p["what_it_is"],
             "family_pitch": FAMILY_REASONS.get(p.get("family"), p["what_it_is"])}
            for p in products
        ],
        "recommended_horizon_tier": {"tier": tier_name, **tier},
        "horizon_line": horizon_line,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(recommend("Cloud Kitchen", "high_up", 18.4, 512), indent=2, ensure_ascii=False))
