"""
ml/uplift.py
============
Campaign uplift — did the promotion CAUSE incremental orders, or did it discount
people who were going to order anyway?

This is a different question from forecasting, and the forecasting model cannot
answer it. A correlational model sees "campaign days have more orders" and
credits the campaign — but campaigns are run on weekends, before festivals, and
when a competitor opens. Those days would have been busy regardless. Paying 30%
margin to people who were already coming is the most common way food-delivery
promotion budgets are wasted.

What this does
--------------
T-learner with two LightGBM models on the SAME features the forecaster uses,
plus campaign attributes:

    μ₁(x) = expected orders GIVEN a campaign ran
    μ₀(x) = expected orders GIVEN no campaign ran
    uplift(x) = μ₁(x) − μ₀(x)

and, because campaign assignment is observational rather than randomised, an
IPW (inverse-propensity-weighted) estimate as a cross-check: a third LightGBM
predicts the probability of being treated, and days are reweighted so treated
and untreated look alike on their features. When the T-learner and IPW numbers
disagree badly, the confounding is too strong to trust either — the honest
answer, and the module says so.

HONEST LIMITS
-------------
Observational uplift is not a randomised experiment. It adjusts for the features
it can see, and nothing else. If campaigns are triggered by something not in the
data (a manager's hunch about a slow week), no estimator here recovers the true
effect. The trustworthy version is a switchback or geo holdout test — hold out
20% of outlets at random, run the campaign on the rest, compare. Use this module
to size and prioritise experiments, not to replace them.

Input schema — data/training/campaigns.csv
------------------------------------------
    date              YYYY-MM-DD
    city              must match data/india_geo.CITIES
    restaurant_type   one of the six formats
    orders            actual daily orders
    treated           1 if a campaign ran that day, 0 if not
    discount_pct      optional, e.g. 20 for 20% off (0 for control rows)
    channel           optional, e.g. swiggy / zomato / own-app
    aov               optional — enables revenue-level uplift

Usage
-----
    python -m ml.uplift                         # uses data/training/campaigns.csv
    python -m ml.uplift --csv my_campaigns.csv --segment city
"""

from __future__ import annotations

import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.config import CAMPAIGNS_CSV, LGBM_PARAMS, RANDOM_SEED
from ml.dataset import build_matrix, load_raw, normalize_orders
from ml.features import CATEGORICAL_COLUMNS

REQUIRED = ["date", "city", "restaurant_type", "orders", "treated"]

UPLIFT_PARAMS = dict(LGBM_PARAMS, learning_rate=0.05, num_leaves=48,
                     min_data_in_leaf=40, seed=RANDOM_SEED)
ROUNDS = 400


def _load(path: str) -> pd.DataFrame:
    df = load_raw(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    df["treated"] = df["treated"].astype(int)
    if set(df["treated"].unique()) - {0, 1}:
        raise ValueError("`treated` must be 0 or 1")
    return normalize_orders(df)


def _design(df: pd.DataFrame):
    """PRE-TREATMENT features only, plus log-orders as the outcome.

    `discount_pct` and `channel` are deliberately excluded. They are consequences
    of the treatment, not covariates that existed before it: a control day always
    has discount 0, so including them lets the propensity model identify treatment
    perfectly (AUC 1.0), which destroys overlap and makes every causal estimate
    meaningless. Conditioning on post-treatment variables is a classic way to get
    a confident wrong answer. They are used only for the by-discount breakdown.
    """
    X, y, _, _ = build_matrix(df)
    return X.copy(), y


def _fit(X, y, cats, params=None, rounds=ROUNDS):
    d = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
    return lgb.train(params or UPLIFT_PARAMS, d, num_boost_round=rounds,
                     callbacks=[lgb.log_evaluation(0)])


def _crossfit(X, y, t, cats, n_folds: int = 5, seed: int = RANDOM_SEED):
    """Out-of-fold mu1, mu0 and propensity — the debiased-ML construction.

    Fitting and scoring on the same rows is what made the first version of this
    report nonsense: LightGBM memorises the treatment assignment, propensities
    saturate at 0/1, overlap reads 0% and the IPW estimate collapses back onto
    the naive difference. Every prediction here comes from a model that never saw
    that row.
    """
    from sklearn.model_selection import StratifiedKFold

    mu1 = np.zeros(len(X)); mu0 = np.zeros(len(X)); ps = np.zeros(len(X))
    pparams = dict(UPLIFT_PARAMS, objective="binary", metric="binary_logloss",
                   num_leaves=24, min_data_in_leaf=80)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(X, t):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, ttr = y[tr_idx], t[tr_idx]
        m1 = _fit(Xtr[ttr == 1], ytr[ttr == 1], cats)
        m0 = _fit(Xtr[ttr == 0], ytr[ttr == 0], cats)
        pm = _fit(Xtr, ttr.astype(float), cats, params=pparams, rounds=250)
        mu1[te_idx] = m1.predict(Xte)
        mu0[te_idx] = m0.predict(Xte)
        ps[te_idx] = pm.predict(Xte)
    return mu1, mu0, np.clip(ps, 0.02, 0.98)


def estimate(path: str = None, segment: str = None) -> dict:
    path = path or CAMPAIGNS_CSV
    df = _load(path)
    X, y = _design(df)
    t = df["treated"].to_numpy()
    cats = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    n_t, n_c = int(t.sum()), int((1 - t).sum())
    if n_t < 30 or n_c < 30:
        return {"error": f"need at least 30 treated and 30 control rows "
                         f"(have {n_t} treated, {n_c} control)"}

    # ── Naive difference: what a dashboard would tell you ─────────────────────
    naive_pct = (np.exp(y[t == 1].mean() - y[t == 0].mean()) - 1) * 100

    # ── Cross-fitted T-learner + propensity (debiased ML) ─────────────────────
    mu1, mu0, ps = _crossfit(X, y, t, cats)
    uplift_ratio = np.exp(mu1 - mu0)                 # multiplicative, per row
    treated_rows = t == 1
    att_pct = (uplift_ratio[treated_rows].mean() - 1) * 100      # effect where it ran
    ate_pct = (uplift_ratio.mean() - 1) * 100                    # effect if run everywhere

    # ── IPW cross-check on the out-of-fold propensities ───────────────────────
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    ipw_treated = np.average(y[t == 1], weights=w[t == 1])
    ipw_control = np.average(y[t == 0], weights=w[t == 0])
    ipw_pct = (np.exp(ipw_treated - ipw_control) - 1) * 100

    overlap = float(np.mean((ps > 0.1) & (ps < 0.9)) * 100)
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = round(float(roc_auc_score(t, ps)), 3)      # 0.5 = assignment looks random
    except Exception:                                     # noqa: BLE001
        pass
    disagreement = abs(att_pct - ipw_pct)
    if overlap < 60:
        verdict = ("WEAK — campaign days look very different from non-campaign days, "
                   "so the comparison is not like-for-like. Run a holdout test.")
    elif disagreement > max(6.0, 0.5 * abs(att_pct)):
        verdict = ("SHAKY — the two estimators disagree materially, which means "
                   "confounding is doing real work here. Treat the range, not a point.")
    else:
        verdict = "REASONABLE — estimators agree and treated/control overlap is decent."

    # ── Incrementality economics ──────────────────────────────────────────────
    obs_treated_orders = float(df.loc[treated_rows, "orders"].sum())
    incr_share = 1 - 1 / max(uplift_ratio[treated_rows].mean(), 1e-6)
    incremental_orders = obs_treated_orders * incr_share
    discounted_share = (1 - incr_share) * 100

    out = {
        "rows": int(len(df)), "treated_rows": n_t, "control_rows": n_c,
        "naive_difference_pct": round(float(naive_pct), 2),
        "uplift_att_pct": round(float(att_pct), 2),
        "uplift_ate_pct": round(float(ate_pct), 2),
        "uplift_ipw_pct": round(float(ipw_pct), 2),
        "propensity_overlap_pct": round(overlap, 1),
        "propensity_auc": auc,
        "confounding_strength": (
            "none detected" if auc is None or auc < 0.6 else
            "mild" if auc < 0.7 else "moderate" if auc < 0.8 else "strong"),
        "verdict": verdict,
        "naive_overstatement_pct_points": round(float(naive_pct - att_pct), 2),
        "incremental_orders_on_campaign_days": round(incremental_orders),
        "observed_orders_on_campaign_days": round(obs_treated_orders),
        "cannibalised_share_pct": round(float(discounted_share), 1),
        "interpretation": (
            f"Of the orders on campaign days, roughly {discounted_share:.0f}% would have "
            f"arrived anyway — those are margin given away, not demand created."),
        "caveat": ("Observational estimate: adjusts only for observed features. "
                   "A geo/switchback holdout is the trustworthy version."),
    }

    if "discount_pct" in df.columns and df["discount_pct"].nunique() > 2:
        bands = pd.cut(df.loc[treated_rows, "discount_pct"],
                       [0, 10, 20, 30, 100], include_lowest=True)
        by_disc = (pd.Series(uplift_ratio[treated_rows]).groupby(bands.to_numpy())
                   .agg(["mean", "size"]))
        out["uplift_by_discount_band"] = [
            {"discount_band": str(k), "uplift_pct": round((v["mean"] - 1) * 100, 1),
             "n_days": int(v["size"])}
            for k, v in by_disc.iterrows() if v["size"] > 0
        ]

    if segment and segment in df.columns:
        seg = (pd.Series(uplift_ratio[treated_rows])
               .groupby(df.loc[treated_rows, segment].to_numpy()).agg(["mean", "size"]))
        out[f"uplift_by_{segment}"] = sorted(
            [{segment: str(k), "uplift_pct": round((v["mean"] - 1) * 100, 1),
              "n_days": int(v["size"])} for k, v in seg.iterrows()],
            key=lambda d: -d["uplift_pct"])
    return out


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CAMPAIGNS_CSV)
    ap.add_argument("--segment", help="extra breakdown column, e.g. city or channel")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        print(f"No campaign file at {a.csv}.\n\nExpected columns:\n"
              "  date, city, restaurant_type, orders, treated"
              " [, discount_pct, channel, aov]\n\n"
              "`treated` = 1 on days a campaign ran for that city+format, 0 otherwise.\n"
              "Control rows are what make this work — export non-campaign days too.")
        return
    res = estimate(a.csv, a.segment)
    if "error" in res:
        print(res["error"]); return

    print(f"rows {res['rows']:,}  (treated {res['treated_rows']:,} / "
          f"control {res['control_rows']:,})\n")
    print(f"  naive difference        {res['naive_difference_pct']:+7.2f}%   "
          f"<- what a dashboard reports")
    print(f"  uplift (T-learner, ATT) {res['uplift_att_pct']:+7.2f}%   <- causal estimate")
    print(f"  uplift (IPW check)      {res['uplift_ipw_pct']:+7.2f}%")
    print(f"  overlap                 {res['propensity_overlap_pct']:7.1f}%   "
          f"(propensity AUC {res['propensity_auc']}, "
          f"confounding {res['confounding_strength']})")
    print(f"\n  {res['verdict']}")
    print(f"\n  naive overstates by {res['naive_overstatement_pct_points']:.1f} pct points")
    print(f"  incremental orders on campaign days: "
          f"{res['incremental_orders_on_campaign_days']:,} of "
          f"{res['observed_orders_on_campaign_days']:,}")
    print(f"  {res['interpretation']}")
    for key in list(res):
        if key.startswith("uplift_by_"):
            print(f"\n  {key}:")
            for row in res[key]:
                print("   ", row)
    print(f"\n  {res['caveat']}")


if __name__ == "__main__":
    _main()
