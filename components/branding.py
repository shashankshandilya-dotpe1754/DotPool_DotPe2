"""
components/branding.py
=======================
Single source of truth for the app's identity — name, tagline, favicon,
sidebar logo and the theme — so every page looks and reads the same way
instead of each page carrying its own copy of this.

Layout: black sidebar (dark text-on-black), white main canvas (dark
text-on-white), orange buttons. Two separate text-color scales are kept
deliberately — INK/MUTE for the white canvas, SIDEBAR_INK/SIDEBAR_MUTE for
the black sidebar — because the same grey doesn't read on both backgrounds.
"""

import os
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

APP_NAME = "DotPool"
TAGLINE = "Business Strategy Intelligence"
FAVICON_PATH = os.path.join(ASSETS, "favicon.png")
LOGO_PATH = os.path.join(ASSETS, "dotpool_logo.png")

# Brand red — used for the logo, section-header underlines, tab-active
# underline, and other brand accents. NOT the button color (that's ORANGE).
RED = "#C7333A"
RED_DIM = "rgba(199,51,58,.14)"

# Buttons / interactive-control accent.
ORANGE = "#F0631D"
ORANGE_DIM = "rgba(240,99,29,.14)"

# Sidebar — black background, light text.
SIDEBAR_BG = "#000000"
SIDEBAR_INK = "#F2F2F2"
SIDEBAR_MUTE = "rgba(255,255,255,.58)"
SIDEBAR_LINE = "rgba(255,255,255,.14)"

# Main canvas — white background, dark text.
CANVAS_BG = "#FFFFFF"
INK = "#14161A"
MUTE = "rgba(20,22,26,.60)"
LINE = "rgba(20,22,26,.12)"
PANEL = "#F6F6F7"

# Legacy alias some pages may still reference for a dark chip/hero background
# (e.g. the Ask DotPool hero card, which stays dark-on-purpose for contrast).
BLACK = "#000000"


def page_icon():
    try:
        from PIL import Image
        return Image.open(FAVICON_PATH)
    except Exception:
        return "●"


def inject_theme():
    """Black sidebar / white canvas / orange buttons theme."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{
      font-family:'Inter',sans-serif;
      font-size: clamp(16px, 1vw + 9px, 21px);
    }}
    .main .block-container {{
      max-width: min(1450px, 96vw) !important;
    }}
    h1 {{ font-size: clamp(1.75rem, 1.6vw + 1.2rem, 2.5rem) !important; }}
    h2 {{ font-size: clamp(1.45rem, 1.2vw + 1rem, 2rem) !important; }}
    h3 {{ font-size: clamp(1.2rem, 1vw + 0.85rem, 1.6rem) !important; }}
    h4 {{ font-size: clamp(1.05rem, 0.6vw + 0.8rem, 1.3rem) !important; }}

    /* ── Canvas: white ────────────────────────────────────────────────── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .main .block-container {{
      background:{CANVAS_BG} !important;
    }}
    [data-testid="stHeader"] {{ background:{CANVAS_BG} !important; }}

    .main, .main p, .main span, .main label, .main div {{ color:{INK}; }}
    /* Headings are dark-on-white EXCEPT inside a deliberately-dark card
       (.hero / .about-hero) — those need a more specific override below,
       since a bare "h1{{color}}" rule would otherwise win everywhere. */
    h1, h2, h3, h4 {{ color:{INK} !important; }}
    .hero h1, .hero h2, .hero h3, .hero p, .hero span, .hero div,
    .about-hero h1, .about-hero h2, .about-hero h3, .about-hero p, .about-hero span, .about-hero div {{
      color:#ffffff !important;
    }}

    /* ── Sidebar: black ───────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
      background:{SIDEBAR_BG} !important;
      border-right:1px solid {SIDEBAR_LINE};
    }}
    section[data-testid="stSidebar"] * {{ color:{SIDEBAR_INK} !important; }}

    /* Anything with its OWN light-themed chrome (alerts, code blocks, text/
       date inputs, select dropdowns) already renders correctly light-bg +
       dark-text on its own — exactly as it does everywhere in the white
       main canvas. The blanket rule above was reaching *inside* these and
       force-coloring their text light too, making it invisible against
       their own light background. These rules are placed after the blanket
       rule on purpose (equal-specificity ties resolve by source order) to
       win that fight and restore normal, readable contrast inside them. */
    section[data-testid="stSidebar"] [data-testid="stAlert"],
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"],
    section[data-testid="stSidebar"] pre,
    section[data-testid="stSidebar"] .stTextInput, section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextArea, section[data-testid="stSidebar"] .stTextArea textarea,
    section[data-testid="stSidebar"] .stDateInput, section[data-testid="stSidebar"] .stDateInput input,
    section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] [data-baseweb="select"] {{
      color:{INK} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stAlert"] *,
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"] *,
    section[data-testid="stSidebar"] pre *,
    section[data-testid="stSidebar"] .stTextInput *, section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextArea *, section[data-testid="stSidebar"] .stTextArea textarea,
    section[data-testid="stSidebar"] .stDateInput *, section[data-testid="stSidebar"] .stDateInput input,
    section[data-testid="stSidebar"] .stSelectbox *, section[data-testid="stSidebar"] [data-baseweb="select"] * {{
      color:inherit !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
      background:{ORANGE_DIM} !important;
      border-left:3px solid {ORANGE};
    }}
    /* Logo flush to the extreme top-left of the sidebar, no default padding */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] > div:first-child {{
      padding-top:0 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child {{
      margin-top:0 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stImage"] {{
      margin:0 !important; padding:0 !important; text-align:left !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stImage"] img {{
      margin:0 !important; display:block !important;
    }}

    .section-hdr {{
      color:{INK}; font-size:20px; font-weight:700; margin:22px 0 14px;
      padding-bottom:7px; border-bottom:2px solid {RED};
      text-transform:uppercase; letter-spacing:.04em;
    }}

    .card {{
      background:{PANEL}; border:1px solid {LINE}; border-radius:14px;
      padding:18px 20px; color:{INK};
    }}

    /* ── Buttons: orange ──────────────────────────────────────────────── */
    .stButton>button, .stDownloadButton>button {{
      background:{ORANGE} !important; color:#fff !important;
      border:none !important; border-radius:8px !important;
      font-weight:600 !important; font-size:1rem !important;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
      background:#C74F14 !important;
    }}

    button[data-baseweb="tab"] {{ color:{MUTE} !important; font-size:1.05rem !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{
      color:{INK} !important; border-bottom-color:{RED} !important;
    }}
    div[data-baseweb="tab-highlight"] {{ background-color:{RED} !important; }}

    /* Inputs — light field, dark text, readable on either background */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"]>div,
    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] input {{
      background:{PANEL} !important; color:{INK} !important; border-color:{LINE} !important;
      font-size:1rem !important;
    }}
    /* Date inputs sit in narrow sidebar columns (From/To side by side) — the
       general 1rem size (up to 21px on a wide screen) overflows a box that
       narrow and clips the date. Fixed, smaller size so it always fits. */
    .stDateInput input {{
      background:{PANEL} !important; color:{INK} !important; border-color:{LINE} !important;
      font-size:13px !important; padding:8px 6px !important;
    }}
    [data-testid="stChatInput"] {{ background:{PANEL} !important; border-color:{LINE} !important; }}

    div[data-testid="stToggle"] [role="switch"][aria-checked="true"] {{ background:{ORANGE} !important; }}

    .sidebar-tagline {{
      font-size:14px; letter-spacing:.05em; text-transform:uppercase;
      color:{SIDEBAR_MUTE}; margin-top:8px; margin-bottom:14px;
    }}
    </style>
    """, unsafe_allow_html=True)


def sidebar_header(current_page: str = ""):
    """Logo, flush to the top-left of the sidebar, + the active section label."""
    with st.sidebar:
        try:
            st.image(LOGO_PATH, width=180)
        except Exception:
            st.markdown(f"### {APP_NAME}")
        st.markdown(f"<div class='sidebar-tagline'>{TAGLINE}</div>", unsafe_allow_html=True)
        if current_page:
            st.markdown(f"<div style='font-size:15px;opacity:.5;margin-bottom:10px'>{current_page}</div>",
                       unsafe_allow_html=True)
        st.markdown(f"<hr style='border-color:{SIDEBAR_LINE};margin:4px 0 14px'>", unsafe_allow_html=True)


def page_logo(width: int = 280):
    """A large, clear logo at the very top of the main content area — on
    every page, above anything else — separate from the small sidebar
    logo. Call this first, right after inject_theme(), before any hero,
    title, or other content."""
    try:
        st.image(LOGO_PATH, width=width)
    except Exception:
        st.markdown(f"## {APP_NAME}")
