"""
Restaurant Impact Page — QSR, Fine Dining, PBCL, Casual Dining, Cloud Kitchen, Cafe
pages/01_Restaurant_Impact.py
"""

import streamlit as st
import sys, os
from datetime import date, timedelta, datetime
import pytz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.india_geo import CITIES
from utils.restaurant_impact import predict_date_range, BASELINES, ALL_RT
from components.branding import page_icon, inject_theme, sidebar_header, page_logo, INK, PANEL, LINE

try:
    from ml.product_advisor import recommend as _pa_recommend
    from data.dotpe_products import contextual_pitch as _pa_contextual_pitch
    _PA_AVAIL = True
except Exception:
    _PA_AVAIL = False

def _signal_for_pct(pct: float) -> str:
    if pct >= 30:  return "very_high_up"
    if pct >= 15:  return "high_up"
    if pct >= 5:   return "slight_up"
    if pct <= -35: return "very_low"
    if pct <= -20: return "low"
    if pct <= -8:  return "slight_down"
    return "neutral"

st.set_page_config(page_title="DotPool — Restaurant Impact", page_icon=page_icon(), layout="wide")
inject_theme()
page_logo()

IST   = pytz.timezone("Asia/Kolkata")
now_i = datetime.now(IST)
today = now_i.date()

st.markdown(f"""
<style>
.rt-card{{border-radius:16px;padding:16px 20px;color:{INK};margin-bottom:12px;border-left:4px solid}}
.sig-badge{{display:inline-block;padding:4px 12px;border-radius:16px;font-size:15px;font-weight:600;margin-top:6px}}
.kpi-big{{text-align:center;background:{PANEL};border-radius:14px;
  padding:14px 8px;border:1px solid {LINE};color:{INK}}}
.kpi-val{{font-size:27px;font-weight:600}}
.kpi-lbl{{font-size:13px;opacity:.6;margin-top:3px}}
</style>
""", unsafe_allow_html=True)

RT_COLORS = {
    "QSR":           "#FF6F00",
    "Fine Dining":   "#1565C0",
    "PBCL":          "#4A148C",
    "Casual Dining": "#2E7D32",
    "Cloud Kitchen": "#00838F",
    "Cafe":          "#6A1B9A",
}
RT_ICONS = {
    "QSR":"","Fine Dining":"","PBCL":"",
    "Casual Dining":"","Cloud Kitchen":"","Cafe":"",
}
RT_DESC = {
    "QSR":           "Quick Service — counter orders, delivery, takeaway (Mad Over Donuts, 99 Pancakes, McDonald's)",
    "Fine Dining":   "Upscale table dining — à la carte, reservations, white-tablecloth (ITC, Taj restaurants)",
    "PBCL":          "Pub, Bar, Café, Lounge — beverages + light bites (Social, Barista, brewpubs)",
    "Casual Dining": "Mid-range sit-down — Chili's, Barbeque Nation, Mainland China, Olive",
    "Cloud Kitchen": "Delivery-only brands — Rebel Foods, Biryani By Kilo, Faasos, Freshmenu",
    "Cafe":          "Coffee shops & bakery-cafes — Starbucks, Blue Tokai, Third Wave Coffee, Chaayos",
}
SIGNAL_COLOR = {
    "very_high_up":"#00C853","high_up":"#43A047","slight_up":"#1B5E20",
    "neutral":"#78909C","slight_down":"#D84315","low":"#C62828","very_low":"#B71C1C",
}
SIGNAL_LABEL = {
    "very_high_up":"Very High Demand","high_up":"High Uplift",
    "slight_up":"↑ Slight Uplift","neutral":"-> Neutral",
    "slight_down":"↓ Slight Drop","low":"Low Demand","very_low":"Very Low",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
sidebar_header("Restaurant Impact")
with st.sidebar:
    st.markdown("Predicted order volume & price impact vs normal baseline")
    st.markdown(f"<div style='font-size:14px;opacity:.5'>IST: {now_i.strftime('%d %b %Y, %I:%M %p')}</div>", unsafe_allow_html=True)
    st.markdown("---")
    city = st.selectbox("City", sorted(CITIES.keys()),
                        index=sorted(CITIES.keys()).index("New Delhi"))
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1: start_date = st.date_input("From", today)
    with c2: end_date   = st.date_input("To",   today+timedelta(days=14))
    st.markdown("---")
    st.markdown("###  Weather Override")
    override  = st.toggle("Apply custom weather")
    temp_max  = st.slider("Max Temp (°C)", 10, 50, 32) if override else None
    precip_mm = st.slider("Rainfall (mm)",  0, 80,  0) if override else None
    st.markdown("---")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

# ── Weather fetch ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(city, start_str, end_str):
    """
    Fetch weather for the full date range.
    - Historical dates  → Open-Meteo archive API (actual observed data)
    - Today + future    → Open-Meteo forecast API
    For forecast dates, precipitation_sum can be None — we use
    precipitation_probability_max to estimate mm when sum is missing.
    """
    import requests; coords=CITIES[city]; temp_map={}
    try:
        # ── Historical (up to yesterday) ──────────────────────────────────────
        hist_end = min(date.fromisoformat(end_str), today - timedelta(days=1))
        if date.fromisoformat(start_str) <= hist_end:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "daily": ["temperature_2m_max","temperature_2m_min",
                          "precipitation_sum","weathercode"],
                "start_date": start_str, "end_date": hist_end.isoformat(),
                "timezone": "Asia/Kolkata"}, timeout=12)
            if r.status_code == 200:
                d = r.json()["daily"]
                for i in range(len(d["time"])):
                    temp_map[d["time"][i]] = {
                        "temp_max_c":      d["temperature_2m_max"][i],
                        "temp_min_c":      d["temperature_2m_min"][i],
                        "precipitation_mm":d["precipitation_sum"][i] or 0,
                        "weather_code":    d["weathercode"][i],
                    }
        # ── Forecast (today + future) ─────────────────────────────────────────
        fcst_start = max(date.fromisoformat(start_str), today)
        if fcst_start <= date.fromisoformat(end_str):
            days = (date.fromisoformat(end_str) - today).days + 3
            r2 = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "daily": ["temperature_2m_max","temperature_2m_min",
                          "precipitation_sum","weathercode",
                          "precipitation_probability_max"],
                "forecast_days": min(days, 16),
                "timezone": "Asia/Kolkata"}, timeout=10)
            if r2.status_code == 200:
                d2 = r2.json()["daily"]
                for i in range(len(d2["time"])):
                    precip = d2["precipitation_sum"][i]
                    rain_prob = d2["precipitation_probability_max"][i] or 0
                    # If precipitation_sum is None (common for forecast),
                    # estimate from probability: >60% → ~8mm, >30% → ~3mm, else 0
                    if precip is None:
                        if rain_prob >= 60:   precip = 10.0
                        elif rain_prob >= 40: precip = 5.0
                        elif rain_prob >= 25: precip = 2.0
                        else:                precip = 0.0
                    temp_map[d2["time"][i]] = {
                        "temp_max_c":           d2["temperature_2m_max"][i],
                        "temp_min_c":           d2["temperature_2m_min"][i],
                        "precipitation_mm":     precip,
                        "weather_code":         d2["weathercode"][i],
                        "rain_probability_pct": rain_prob,
                    }
    except Exception as e:
        pass
    return temp_map

with st.spinner("Calculating impact for all 6 restaurant types..."):
    weather_data = get_weather(city, start_date.isoformat(), end_date.isoformat())
    if override:
        for k in weather_data:
            weather_data[k]["temp_max_c"]=temp_max; weather_data[k]["precipitation_mm"]=precip_mm
    predictions = predict_date_range(city, start_date, end_date, weather_data)

# ── Context for the Sales Playbook pitch lines: which events are actually
# active in this window, and any weather worth naming — computed once here
# so every tab's contextual_pitch() call names the real driver. ────────────
# Built per-day (not just "anything overlapping the whole range") because a
# flat range-average % diluted by mostly-normal days, cited against every
# event that merely overlaps some part of the range, is exactly the kind of
# "number doesn't match the reason" mismatch this is meant to avoid — the
# Sales Playbook below keys off each restaurant type's own peak day and only
# names events actually active on THAT day.
_EVENT_CATS = ("Festival","Sports Event","Commercial Event",
               "Public Event","Government Holiday","Emergency Crisis")
_events_by_day = {}
if _PA_AVAIL:
    from data.events_db import EVENTS_DB as _EVDB, _event_applies_to_geo as _ev_geo
    from datetime import date as _dt2, timedelta as _td2
    _s, _e = _dt2.fromisoformat(start_date.isoformat()), _dt2.fromisoformat(end_date.isoformat())
    _relevant_evs = [ev for ev in _EVDB if ev["category"] in _EVENT_CATS and _ev_geo(ev, city=city)
                      and not (_dt2.fromisoformat(ev["end_date"]) < _s or _dt2.fromisoformat(ev["start_date"]) > _e)]
    _cur = _s
    while _cur <= _e:
        _events_by_day[_cur.isoformat()] = [
            ev["name"] for ev in _relevant_evs
            if _dt2.fromisoformat(ev["start_date"]) <= _cur <= _dt2.fromisoformat(ev["end_date"])
        ]
        _cur += _td2(days=1)

# Still used by the top KPI row's context and as a whole-range fallback.
_active_event_names = sorted({n for names in _events_by_day.values() for n in names})

_wx_vals = list(weather_data.values())
_page_weather_note = None
_all_tmax = [w.get("temp_max_c") for w in _wx_vals if w.get("temp_max_c") is not None]
_all_precip = [w.get("precipitation_mm") for w in _wx_vals if w.get("precipitation_mm") is not None]
if _all_tmax and max(_all_tmax) >= 40:
    _page_weather_note = f"forecast heat up to {max(_all_tmax):.0f}°C will also cut outdoor/dine-in footfall"
if _all_precip and sum(_all_precip)/max(len(_all_precip),1) >= 5:
    _page_weather_note = "rain in the forecast will push delivery up and dine-in down further"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"##  Restaurant Demand Impact — {city}")
st.markdown(f"*{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')} · IST: {now_i.strftime('%d %b %Y, %I:%M %p')} · Index: 100 = typical weekday ·  Price shown as avg. order value (AOV)*")

# ── KPI row — all 6 types ──────────────────────────────────────────────────────
cols = st.columns(6)
for col,rt in zip(cols, ALL_RT):
    preds=predictions[rt]
    avg_i=sum(p["predicted_index"] for p in preds)/len(preds) if preds else 100
    avg_p=sum(p["pct_change"] for p in preds)/len(preds) if preds else 0
    avg_aov=sum(p["predicted_aov"] for p in preds)/len(preds) if preds else 0
    avg_rev_p=sum(p["revenue_pct_change"] for p in preds)/len(preds) if preds else 0
    best =max(preds,key=lambda x:x["pct_change"]) if preds else {}
    worst=min(preds,key=lambda x:x["pct_change"]) if preds else {}
    c=RT_COLORS[rt]
    with col:
        st.markdown(f"""
        <div class="kpi-big" style="border-top:3px solid {c}">
          <div style="font-size:19px;margin-bottom:4px">{RT_ICONS[rt]} {rt}</div>
          <div class="kpi-val" style="color:{c}">{avg_i:.0f}</div>
          <div class="kpi-lbl">avg order index</div>
          <div style="font-size:15px;margin-top:5px;color:{'#43A047' if avg_p>=0 else '#C62828'};font-weight:500">{avg_p:+.1f}% orders</div>
          <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(0,0,0,.1)">
            <div style="font-size:18px;font-weight:600">₹{avg_aov:,.0f}</div>
            <div class="kpi-lbl">avg price / order</div>
            <div style="font-size:14px;margin-top:2px;color:{'#43A047' if avg_rev_p>=0 else '#C62828'};font-weight:500">{avg_rev_p:+.1f}% revenue</div>
          </div>
          <div style="font-size:12px;opacity:.35;margin-top:8px">
            ↑ {best.get('pct_change',0):+.0f}% {best.get('date','')[:10]}<br>
            ↓ {worst.get('pct_change',0):+.0f}% {worst.get('date','')[:10]}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "QSR","Fine Dining","PBCL",
    "Casual Dining","Cloud Kitchen","Cafe","Compare All"
])

def render_tab(rt):
    preds=predictions[rt]; c=RT_COLORS[rt]
    st.markdown(f"<div style='color:rgba(0,0,0,.5);font-size:15px;margin-bottom:12px'>{RT_DESC[rt]}</div>",unsafe_allow_html=True)

    # ── Sales Playbook — what DotPe should pitch for this format, right now ──
    # Keyed to this restaurant type's own PEAK day in the range, not a flat
    # range-average — a number diluted across mostly-normal days, cited
    # against every event that merely overlaps some part of the range, is
    # exactly the "number doesn't match the reason" gap this fixes. The
    # recommendation, the % figure, and the named event(s)/weather are all
    # now for the same specific day.
    if _PA_AVAIL and preds:
        peak = max(preds, key=lambda p: abs(p["pct_change"]))
        peak_pct = peak["pct_change"]
        peak_aov = peak["predicted_aov"]
        peak_day_events = _events_by_day.get(peak["date"][:10], [])
        sig = _signal_for_pct(peak_pct)
        try:
            rec = _pa_recommend(rt, sig, peak_pct, peak_aov)
        except Exception:
            rec = None
        if rec:
            st.markdown('<div class="section-hdr">- Sales Playbook — what to pitch</div>', unsafe_allow_html=True)
            from datetime import date as _dt3
            peak_date_txt = _dt3.fromisoformat(peak["date"][:10]).strftime("%d %b")
            st.markdown(f"<div style='font-size:12.5px;opacity:.5;margin-bottom:4px'>"
                        f"Keyed to {rt}'s single biggest swing in this range — {peak_date_txt} "
                        f"({peak_pct:+.0f}% vs. a normal day)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:15px;opacity:.75;margin-bottom:10px'>{rec['business_view']} "
                        f"<b>Pricing opportunity:</b> {rec['pricing_opportunity']}</div>", unsafe_allow_html=True)
            prod_cols = st.columns(min(len(rec["recommended_products"]), 3) or 1)
            for i, prod in enumerate(rec["recommended_products"]):
                is_up = sig in ("very_high_up","high_up","slight_up")
                bg, fc, bd = ("rgba(67,160,71,.12)","#1B5E20","#43A047") if is_up else ("rgba(255,143,0,.12)","#E65100","#FF8F00")
                tag = "Top pick" if i == 0 else "Also fits"
                fam = prod.get('family', prod['name'])
                pitch = (_pa_contextual_pitch(fam, peak_pct, peak_day_events or _active_event_names, _page_weather_note)
                        if _PA_AVAIL else prod.get('family_pitch', prod['why']))
                with prod_cols[i % len(prod_cols)]:
                    st.markdown(f"""
                    <div style="background:{bg};border-left:3px solid {bd};border-radius:8px;
                      padding:10px 12px;margin-bottom:8px;min-height:118px">
                      <div style="font-size:12.5px;font-weight:700;opacity:.5;letter-spacing:.05em;text-transform:uppercase">{tag}</div>
                      <div style="font-size:16px;font-weight:700;color:{fc};margin-top:2px">- {fam}</div>
                      <div style="font-size:13.5px;opacity:.6;margin-top:2px">{prod['name']} · ₹{prod['price']:,} {prod['unit']}</div>
                      <div style="font-size:13.5px;opacity:.75;margin-top:5px;line-height:1.4">{pitch}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:13.5px;opacity:.5;margin:-2px 0 10px'> {rec['horizon_line']}</div>",
                        unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        dates=[p["date"] for p in preds]; idx=[p["predicted_index"] for p in preds]
        base=[p["base_index"] for p in preds]; pct=[p["pct_change"] for p in preds]
        rev_pct=[p["revenue_pct_change"] for p in preds]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dates,y=base,name="Baseline",line=dict(color="rgba(0,0,0,.2)",width=1.5,dash="dot")))
        fig.add_trace(go.Scatter(x=dates,y=idx,name="Predicted Orders",
            line=dict(color=c,width=2.5),fill="tonexty",
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},.12)",
            mode="lines+markers",marker=dict(size=7,color=c)))
        fig.add_trace(go.Bar(x=dates,y=pct,name="Order % Change",
            marker_color=[SIGNAL_COLOR.get(p["signal"],"#78909C") for p in preds],
            opacity=0.55,yaxis="y2",
            text=[f"{v:+.0f}%" for v in pct],textposition="outside",
            textfont=dict(color="rgba(0,0,0,.7)",size=10)))
        fig.add_trace(go.Scatter(x=dates,y=rev_pct,name="Revenue % Change (price-adjusted)",
            line=dict(color="#F57F17",width=2,dash="dash"),mode="lines+markers",
            marker=dict(size=6,color="#F57F17"),yaxis="y2"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,.03)",
            font=dict(color="white",family="Inter"),height=320,margin=dict(l=0,r=0,t=10,b=0),
            hovermode="x unified",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(0,0,0,.08)",tickfont=dict(color="rgba(0,0,0,.6)")),
            yaxis=dict(title="Order Index",gridcolor="rgba(0,0,0,.08)",
                       tickfont=dict(color="rgba(0,0,0,.6)"),range=[0,280]),
            yaxis2=dict(title="% Change",overlaying="y",side="right",
                        tickfont=dict(color="rgba(0,0,0,.4)"),showgrid=False,
                        zeroline=True,zerolinecolor="rgba(0,0,0,.2)"))
        st.plotly_chart(fig,use_container_width=True)
    except Exception as e: st.info(f"Chart: {e}")

    st.markdown('<div class="section-hdr">Notable Impact Days (±8% change)</div>',unsafe_allow_html=True)
    notable=[p for p in preds if abs(p["pct_change"])>=8]
    if notable:
        c1,c2=st.columns(2)
        for i,p in enumerate(notable):
            col=c1 if i%2==0 else c2
            sc=SIGNAL_COLOR.get(p["signal"],"#78909C"); sl=SIGNAL_LABEL.get(p["signal"],p["signal"])
            facts="<br>".join([f"• {f}" for f in p["factors"]]) if p["factors"] else "No major factors"
            with col:
                st.markdown(f"""
                <div class="rt-card" style="background:{c}18;border-left-color:{c}">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <div style="font-size:17px;font-weight:600">{p['weekday']}, {p['date']}</div>
                      <div class="sig-badge" style="background:{sc}22;border:1px solid {sc};color:{sc}">{sl}</div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:31px;font-weight:300;color:{sc}">{p['pct_change']:+.0f}%</div>
                      <div style="font-size:15px;opacity:.5">Index: {p['predicted_index']:.0f}</div>
                    </div>
                  </div>
                  <div style="display:flex;gap:8px;margin-top:8px">
                    <span style="background:rgba(255,213,79,.15);color:#F57F17;border:1px solid rgba(255,213,79,.3);
                      border-radius:8px;padding:3px 9px;font-size:14px;font-weight:500">
                       ₹{p['predicted_aov']:,.0f}/order
                    </span>
                    <span style="background:rgba(255,213,79,.15);color:#F57F17;border:1px solid rgba(255,213,79,.3);
                      border-radius:8px;padding:3px 9px;font-size:14px;font-weight:500">
                      Revenue {p['revenue_pct_change']:+.0f}%
                    </span>
                  </div>
                  <div style="margin-top:8px;font-size:14px;opacity:.65;line-height:1.7">{facts}</div>
                  <div style="font-size:13px;opacity:.35;margin-top:5px">
                    Confidence: {p['confidence']} · {len(p['active_events'])} event(s)
                  </div>
                </div>""",unsafe_allow_html=True)
    else:
        st.info("No significant impact days in this range. Try extending the date range.")

with tab1: render_tab("QSR")
with tab2: render_tab("Fine Dining")
with tab3: render_tab("PBCL")
with tab4: render_tab("Casual Dining")
with tab5: render_tab("Cloud Kitchen")
with tab6: render_tab("Cafe")

with tab7:
    st.markdown("**All 6 restaurant types — order index + signal heatmap + event table**")
    try:
        import plotly.graph_objects as go
        dates=[p["date"] for p in predictions["QSR"]]
        fig3=go.Figure()
        for rt in ALL_RT:
            idx=[p["predicted_index"] for p in predictions[rt]]
            fig3.add_trace(go.Scatter(x=dates,y=idx,name=f"{RT_ICONS[rt]} {rt}",
                line=dict(color=RT_COLORS[rt],width=2),mode="lines+markers",marker=dict(size=5)))
        fig3.add_hline(y=100,line_dash="dot",line_color="rgba(0,0,0,.2)",
                       annotation_text="Weekday baseline",annotation_font_color="rgba(0,0,0,.35)")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,.03)",
            font=dict(color="white",family="Inter"),height=340,margin=dict(l=0,r=0,t=10,b=0),
            hovermode="x unified",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(0,0,0,.08)",tickfont=dict(color="rgba(0,0,0,.6)")),
            yaxis=dict(title="Order Index (100=baseline)",gridcolor="rgba(0,0,0,.08)",
                       tickfont=dict(color="rgba(0,0,0,.6)")))
        st.plotly_chart(fig3,use_container_width=True)

        st.markdown('<div class="section-hdr"> Signal Heatmap</div>',unsafe_allow_html=True)
        sig_val={"very_high_up":3,"high_up":2,"slight_up":1,"neutral":0,"slight_down":-1,"low":-2,"very_low":-3}
        z_data=[]; y_labels=[]; hover_text=[]
        for rt in ALL_RT:
            z_data.append([sig_val.get(p["signal"],0) for p in predictions[rt]])
            y_labels.append(f"{RT_ICONS[rt]} {rt}")
            hover_text.append([
                f"<b>{RT_ICONS[rt]} {rt}</b><br>{p['date']} ({p['weekday']})<br>"
                f"Signal: {SIGNAL_LABEL.get(p['signal'],p['signal'])}<br>"
                f"Change: {p['pct_change']:+.1f}%<br>Index: {p['predicted_index']:.0f}"
                for p in predictions[rt]])
        fig4=go.Figure(go.Heatmap(
            z=z_data,x=dates,y=y_labels,text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0,"#B71C1C"],[0.17,"#C62828"],[0.33,"#D84315"],
                        [0.5,"#455A64"],[0.67,"#1B5E20"],[0.83,"#43A047"],[1,"#00C853"]],
            zmin=-3,zmax=3,showscale=True,
            colorbar=dict(
                title=dict(text="Impact",font=dict(color="white",size=12)),
                tickvals=[-3,-2,-1,0,1,2,3],
                ticktext=["Very Low","Low","Slight↓","Neutral","Slight↑","High","Very High"],
                tickfont=dict(color="white",size=10))))
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white",family="Inter"),height=260,margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(tickfont=dict(color="rgba(0,0,0,.7)"),gridcolor="rgba(0,0,0,.05)"),
            yaxis=dict(tickfont=dict(color="rgba(0,0,0,.9)",size=11)))
        st.plotly_chart(fig4,use_container_width=True)
    except Exception as e: st.warning(f"Chart: {e}")

    st.markdown('<div class="section-hdr"> Event Impact Reference Table</div>',unsafe_allow_html=True)
    from data.events_db import EVENTS_DB, _event_applies_to_geo
    from datetime import date as dt_date
    from utils.restaurant_impact import EVENT_MULTIPLIERS, aov_event_multiplier
    active_evts=[]
    s=dt_date.fromisoformat(start_date.isoformat()); e_d=dt_date.fromisoformat(end_date.isoformat())
    for ev in EVENTS_DB:
        ev_s=dt_date.fromisoformat(ev["start_date"]); ev_e=dt_date.fromisoformat(ev["end_date"])
        if ev_e<s or ev_s>e_d: continue
        if not _event_applies_to_geo(ev,city=city): continue
        active_evts.append(ev)
    if active_evts:
        rows=[]
        for ev in active_evts:
            cat=ev["category"]; sub=ev.get("subcategory","*")
            cm=EVENT_MULTIPLIERS.get(cat,{}); sm=cm.get(sub,cm.get("*",{}))
            def fmt(rt): v=(sm.get(rt,1.0)-1)*100; return f"{'↑' if v>0 else '↓' if v<0 else '→'} {abs(v):.0f}%"
            avg_order_mult = sum(sm.get(rt,1.0) for rt in ALL_RT)/len(ALL_RT)
            avg_aov_mult   = aov_event_multiplier(avg_order_mult)
            avg_rev_pct    = (avg_order_mult*avg_aov_mult-1)*100
            rows.append({"Event":ev["name"],"Category":cat,
                "Dates":f"{ev['start_date']} → {ev['end_date']}",
                "QSR":fmt("QSR"),"Fine Dining":fmt("Fine Dining"),
                "PBCL":fmt("PBCL"),"Casual":fmt("Casual Dining"),
                "Cloud":fmt("Cloud Kitchen"),"Cafe":fmt("Cafe"),
                "Avg Revenue Impact":f"{'↑' if avg_rev_pct>0 else '↓' if avg_rev_pct<0 else '→'} {abs(avg_rev_pct):.0f}%"})
        import pandas as pd
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else: st.info("No events for this city and date range.")

st.markdown("---")
st.markdown(f"<div style='font-size:13px;color:rgba(0,0,0,.3)'>Order baselines: QSR Sat 145, Fine Dining Sat 165, PBCL Sat 175, Casual Sat 155, Cloud Kitchen Sat 130, Cafe Sat 140 · AOV baselines: QSR ₹330, Fine Dining ₹2,100, PBCL ₹1,350, Casual ₹950, Cloud Kitchen ₹480, Cafe ₹420. IST: {now_i.strftime('%d %b %Y %H:%M')}. Predictive model — weather is used internally, not displayed.</div>",unsafe_allow_html=True)
