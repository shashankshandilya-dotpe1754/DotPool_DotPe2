"""
About.py
========
Entry point for the app. Kept as the entry script (rather than a file inside
pages/) specifically so "About" is the first item in the sidebar navigation,
ahead of Home — Streamlit always lists the entry script first, then every
file under pages/ in filename order.
"""

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from components.branding import (
    APP_NAME, TAGLINE, page_icon, inject_theme, sidebar_header, page_logo,
    RED, PANEL, LINE, MUTE, INK,
)

st.set_page_config(page_title=f"{APP_NAME} — About", page_icon=page_icon(), layout="wide")
inject_theme()
page_logo()
sidebar_header("About")

st.markdown(f"""
<style>
.about-hero{{background:linear-gradient(135deg,#151516 0%,#0B0B0C 100%);
  border:1px solid {LINE};border-left:4px solid {RED};border-radius:14px;
  padding:36px 40px;color:white;margin-bottom:26px}}
.about-hero p{{margin:14px 0 0;opacity:.85;font-size:1.15rem;max-width:820px;line-height:1.65}}
.feat-card{{background:{PANEL};border:1px solid {LINE};border-radius:12px;
  padding:24px 26px;height:100%}}
.feat-card h4{{margin:0 0 10px;font-size:1.15rem;font-weight:700;color:{RED}}}
.feat-card p{{margin:0;font-size:1.02rem;color:{MUTE};line-height:1.65}}
.step-num{{display:inline-block;width:32px;height:32px;border-radius:7px;
  background:{RED};color:white;text-align:center;line-height:32px;
  font-weight:700;font-size:1.05rem;margin-right:14px;flex-shrink:0}}
.side-panel{{background:{PANEL};border:1px solid {LINE};border-radius:12px;
  padding:22px 26px;height:100%}}
.side-panel h4{{margin:0 0 12px;font-size:1.05rem;font-weight:700;color:{INK}}}
.side-panel p, .side-panel li{{font-size:1rem;color:{MUTE};line-height:1.6}}
.body-copy{{color:{MUTE};font-size:1.08rem;line-height:1.75;max-width:100%}}
.step-title{{font-weight:600;font-size:1.08rem;color:{INK}}}
.step-desc{{font-size:1rem;color:{MUTE};margin-top:3px;line-height:1.6}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="about-hero">
  <p>DotPool is a simple tool for the Sales team: pick a city, and it tells
  you where demand is heading, why, and exactly which DotPe product to pitch
  because of it — backed by real numbers, not guesswork.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-hdr">What this is for</div>', unsafe_allow_html=True)
st.markdown("""
<p class="body-copy">
Every day, restaurants in different parts of India see more or fewer orders
than usual — because of festivals, weather, holidays, or big sporting events.
A rep on a sales call needs a fast, believable answer to two questions: "what's
demand doing right now for this kind of restaurant?" and "what should I offer
them because of it?" DotPool answers both questions at once, using the same
underlying numbers every time — an actual calendar of festivals and holidays,
actual weather, and DotPe's real product prices — so nobody on the team is
ever guessing or relying on a gut feeling.
</p>
""", unsafe_allow_html=True)

st.markdown('<div class="section-hdr">What it produces</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="feat-card">
      <h4>A demand snapshot</h4>
      <p>For any city and any date range: is demand up or down right now,
      for each type of restaurant — quick-service, fine dining, bars,
      casual dining, cloud kitchens, and cafes?</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="feat-card">
      <h4>A ready-made pitch</h4>
      <p>For every festival, holiday, or big event: which DotPe product to
      offer, to which type of restaurant, and a plain-English reason —
      naming the event and how big the effect is, not a vague description.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="feat-card">
      <h4>A place to ask questions</h4>
      <p>Ask DotPool anything in your own words — "what should I offer this
      client" or "why is demand soft this month" — and get an answer built
      from the same real numbers, not a guess.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-hdr">How it works, in short</div>', unsafe_allow_html=True)

col_steps, col_example = st.columns([1.3, 1], gap="large")
with col_steps:
    steps = [
        ("Pick a city and a date range",
         "That's it to start — Home and Restaurant Impact both use the same simple pickers."),
        ("DotPool checks what's happening on those dates",
         "Festivals, holidays, big sports fixtures, and the weather forecast — all checked automatically, for every restaurant type."),
        ("It turns that into a specific pitch",
         "DotPool matches what it finds against DotPe's real product list and prices, and tells you exactly what to offer and why — not just a number on a chart."),
        ("Ask DotPool for anything else",
         "Type a question in plain English — about a city, a restaurant type, or a broader strategy call — and get an answer from the same data, with the working shown."),
    ]
    for i, (title, desc) in enumerate(steps, start=1):
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;margin-bottom:18px">
          <span class="step-num">{i}</span>
          <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

with col_example:
    st.markdown("""
    <div class="side-panel">
      <h4>A quick example</h4>
      <p style="margin-bottom:10px">Say Diwali is coming up and you're calling
      on cloud kitchens in Bengaluru. Open Home, pick Bengaluru — DotPool
      shows Diwali week is expected to lift cloud-kitchen orders well above
      normal, and recommends leading with <b>SOVA</b>, DotPe's AI order-taking
      product, because delivery volume is about to spike.</p>
      <p style="margin-bottom:0">You now have a specific, current, defensible
      reason to bring up SOVA on that call — instead of a generic pitch, or
      spending time checking a festival calendar and a weather app yourself
      first.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-hdr">Why it is easy to work with</div>', unsafe_allow_html=True)

col_easy, col_solves = st.columns([1.3, 1], gap="large")
with col_easy:
    st.markdown("""
    <p class="body-copy">
    There's nothing to set up and nothing to learn beforehand. Pick a city,
    and everything you need — what demand looks like, what's driving it, and
    what to offer because of it — is already there, in plain language, ready
    to say out loud on a call. Ask DotPool is there for the questions a
    dashboard can't answer on its own: a specific objection, a multi-outlet
    account, or how to frame a renewal — answered from the same data, with
    the option to attach a file or a screenshot for extra context.
    </p>""", unsafe_allow_html=True)
with col_solves:
    st.markdown("""
    <div class="side-panel">
      <h4>What this replaces</h4>
      <ul style="padding-left:20px;margin:0">
        <li style="margin-bottom:8px">Checking a festival calendar and a weather site separately, by hand</li>
        <li style="margin-bottom:8px">Guessing which product fits which type of restaurant</li>
        <li style="margin-bottom:8px">Waiting on a data or analytics team for a number before a call</li>
        <li style="margin-bottom:0">Pitching the same product to everyone regardless of what's actually happening that week</li>
      </ul>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<div style='font-size:1rem;color:{MUTE}'>Continue to Home for today's demand snapshot, "
           f"or Ask DotPool for a direct question.</div>", unsafe_allow_html=True)
