"""Saarthi home page. Run from the repo root: streamlit run app/Home.py"""

import html
import json
import os
import random
import sys

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.components.background import start_warm_up  # noqa: E402
from app.components.decor import hero, lotus_divider  # noqa: E402
from app.components.style import apply_style, footer, speaker_name  # noqa: E402
from app.components.verse_card import translation_for  # noqa: E402

VERSES_PATH = os.path.join(ROOT_DIR, "data", "verses.json")

MOODS = [
    ("😟 Anxious", "Mujhe future ki bahut tension rehti hai, kya karun?"),
    ("😠 Angry", "मुझे बहुत गुस्सा आता है, मैं इसे कैसे नियंत्रित करूँ?"),
    ("📉 Failed", "I studied hard but still failed my exam. How do I cope?"),
    ("🌫️ Lost", "I don't know what the purpose of my life is."),
    ("💔 Grieving", "I lost someone I love and I can't stop feeling sad."),
    ("⚖️ Torn", "Career ko lekar bahut confused hoon, kya karun?"),
]

# Well-known verses for "A verse for you"
FAVOURITES = ["2.47", "2.14", "2.20", "2.48", "2.62", "3.21", "3.35", "4.7", "5.18", "6.5", "6.26",
              "6.35", "9.22", "12.13", "18.66"]

STEPS = [
    ("🗣️", "Understands you", "English, Hindi or Hinglish"),
    ("🪷", "Thinks in Gita terms", "turns your worry into ideas like duty or detachment"),
    ("🔍", "Searches two ways", "exact words and meaning, across 9 translations"),
    ("📜", "Follows the commentary", "verses the commentators keep returning to rank higher"),
    ("✓", "Checks every claim", "each sentence is tested against the verse it cites"),
]

STATS = [
    ("94.6%", "of claims about a verse are supported by it", "plain RAG: 62.7%"),
    ("0", "unsupported claims shown per answer", "plain RAG: 2.1"),
    ("3×", "better search for Hinglish questions", "by translating to English first"),
]


@st.cache_data(show_spinner=False)
def load_verses() -> dict:
    with open(VERSES_PATH, encoding="utf-8") as f:
        return {v["id"]: v for v in json.load(f)}


def go_ask(question: str) -> None:
    st.session_state["ask_prefill"] = question


def new_verse() -> None:
    choices = [v for v in FAVOURITES if v != st.session_state.get("home_verse")]
    st.session_state["home_verse"] = random.choice(choices)


st.set_page_config(page_title="Saarthi", page_icon="🪔", layout="centered")
apply_style()
start_warm_up()  # load the models now, so the first question is quick
hero("Guidance from the Bhagavad Gita, in your own language")

st.markdown(
    "<p style='text-align:center; font-size:1.12rem;'>Ask about the things that weigh on you, such as exams, "
    "anger, loss or a hard decision, and Saarthi answers with the Bhagavad Gita's own verses, "
    "in the language you asked in.</p>",
    unsafe_allow_html=True,
)

# --- moods -----------------------------------------------------------------
st.markdown("### How are you feeling?")
st.caption("Tap one and Saarthi will answer a question like it.")
for row in (MOODS[:3], MOODS[3:]):
    for column, (label, question) in zip(st.columns(3), row):
        if column.button(label, key=f"mood_{label}", use_container_width=True, on_click=go_ask,
                         args=(question,), help=question):
            st.switch_page("pages/1_Ask.py")
if st.button("Ask your own question", type="primary", use_container_width=True):
    st.switch_page("pages/1_Ask.py")

lotus_divider()

# --- a verse for you -------------------------------------------------------
if "home_verse" not in st.session_state:
    new_verse()
verse = load_verses()[st.session_state["home_verse"]]
st.markdown("### A verse for you")
with st.container(border=True):
    sanskrit = "<br>".join(html.escape(line) for line in verse.get("sanskrit", "").splitlines())
    st.markdown(
        f"<div class='saarthi-sanskrit saarthi-reveal' style='animation-delay:.1s'>{sanskrit}</div>"
        f"<div class='saarthi-reveal' style='animation-delay:.5s; text-align:center; font-size:1.1rem'>"
        f"{html.escape(translation_for(verse, 'en'))}</div>"
        f"<div style='text-align:center; color:#B5462C; margin-top:.5rem'>— {speaker_name(verse)}, "
        f"Bhagavad Gita {verse['id']}</div>",
        unsafe_allow_html=True,
    )
st.button("✨ Another verse", on_click=new_verse)

lotus_divider()

# --- how it works and results ---------------------------------------------
st.markdown("### How Saarthi answers")
for column, (icon, title, text) in zip(st.columns(len(STEPS)), STEPS):
    with column:
        st.markdown(
            f"<div style='text-align:center'><div style='font-size:1.6rem'>{icon}</div>"
            f"<div style='font-weight:600; color:#6E2A1A'>{title}</div>"
            f"<div style='font-size:.88rem; color:#6B4A36'>{text}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("### Why you can trust it")
st.caption("Measured on 80 test questions that were never used to tune the system.")
for column, (number, text, compare) in zip(st.columns(len(STATS)), STATS):
    with column, st.container(border=True):
        st.markdown(
            f"<div style='text-align:center'><div style='font-family:\"Cormorant Garamond\",serif; "
            f"font-size:2.4rem; font-weight:700; color:#B5462C'>{number}</div>"
            f"<div style='font-size:.95rem'>{text}</div>"
            f"<div style='font-size:.8rem; color:#8A6A54'>{compare}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("### Explore more")
left, middle, right = st.columns(3)
left.page_link("pages/2_Explore.py", label="Read any verse", icon="📖")
middle.page_link("pages/3_Advanced_Search.py", label="Keyword search", icon="🔎")
right.page_link("pages/4_Compare.py", label="See each search step", icon="⚖️")

footer()
