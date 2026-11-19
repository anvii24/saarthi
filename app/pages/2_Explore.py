"""Explore page: browse any verse by chapter."""

import os
import random
import sys

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline import verses_by_id
from app.components.decor import page_header
from app.components.style import apply_style, footer
from app.components.verse_card import render_verse_card

st.set_page_config(page_title="Explore · Saarthi", page_icon="📖", layout="centered")
apply_style()

page_header("Explore the Bhagavad Gita", "peacock")
st.write(
    "Browse the 18 chapters of the Bhagavad Gita, explore individual verses, "
    "and discover their meanings and interpretations."
)

verses = list(verses_by_id().values())

chapters = sorted(set(v["chapter"] for v in verses))


def surprise_me() -> None:
    """Jump to a random verse."""
    verse = random.choice(verses)
    st.session_state["explore_chapter"] = verse["chapter"]
    st.session_state["explore_verse"] = verse["id"]


st.button("🎲 Surprise me", on_click=surprise_me)

chapter = st.selectbox(
    "Select a chapter",
    chapters,
    format_func=lambda x: f"Chapter {x}",
    key="explore_chapter"
)

chapter_verses = sorted(
    [v for v in verses if v["chapter"] == chapter],
    key=lambda v: v["verse"]
)

verse_options = [v["id"] for v in chapter_verses]

if st.session_state.get("explore_verse") not in verse_options:
    st.session_state["explore_verse"] = verse_options[0]

selected_id = st.selectbox(
    "Select a verse",
    verse_options,
    format_func=lambda x: f"Verse {x.split('.')[1]}",
    key="explore_verse"
)

lang = st.radio(
    "Language",
    ["English", "हिन्दी"],
    horizontal=True
)

selected_lang = "en" if lang == "English" else "hi"

selected_verse = next(
    v for v in chapter_verses if v["id"] == selected_id
)

st.divider()

render_verse_card(selected_verse, selected_lang)

if selected_lang == "en":
    explanation = selected_verse.get("explanation_en")
    commentary = selected_verse.get("commentary")
else:
    explanation = None
    commentary = selected_verse.get("commentary_hi")

if explanation:
    with st.expander("Simple explanation"):
        st.write(explanation)

if commentary:
    with st.expander("Detailed commentary"):
        st.write(commentary)

footer()
