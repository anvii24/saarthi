"""
Advanced Search page (Person 2: Kashvi).

Keyword search over the inverted index, for people who want exact control:
  - Boolean queries with AND / OR / NOT and brackets   (retrieval/boolean.py)
  - exact phrase search                                 (InvertedIndex.phrase_search)
  - a zone: translation, commentary or Sanskrit         (the index keeps them apart)
  - English or Hindi index, and an optional chapter filter
  - a "Did you mean ...?" line from the spell checker   (query/spelling.py)

The spell checker lives here and not in the main pipeline: it corrects against the
Gita's own vocabulary, so it would turn everyday words into Gita words
("exam" -> "enam"). Here people type Gita terms such as "dharmma", where it helps.
"""

import json
import os
import re
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from query.spelling import correct  # noqa: E402
from retrieval.boolean import load_index, search_boolean  # noqa: E402

VERSES_PATH = os.path.join("data", "verses.json")
OPERATORS = {"AND", "OR", "NOT"}
MAX_SHOWN = 50

ZONES = {"Translation": "translation", "Commentary": "commentary", "Sanskrit": "sanskrit"}
LANGUAGES = {"English": "en", "Hindi": "hi"}

st.set_page_config(page_title="Advanced Search - Saarthi", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_verses() -> dict:
    with open(VERSES_PATH, encoding="utf-8") as f:
        return {v["id"]: v for v in json.load(f)}


def show_text(verse: dict, lang: str, limit: int = 280) -> str:
    """A short translation of the verse in the chosen language."""
    translations = verse.get("translations", {})
    preferred = "hi_tejomayananda" if lang == "hi" else "en_gambirananda"
    text = translations.get(preferred) or next(
        (t for key, t in translations.items() if key.startswith(lang) and t), ""
    )
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def suggest(query: str, lang: str, zone: str) -> str | None:
    """Spell-corrected version of the query, or None if nothing changed.

    AND / OR / NOT and brackets are kept as typed; every other word goes through
    query.spelling.correct() against the index vocabulary.
    """
    tokens = re.findall(r"\(|\)|[^\s()]+", query)

    def tidy(words: list[str]) -> str:
        return " ".join(words).replace("( ", "(").replace(" )", ")")

    try:
        fixed = [
            token if token in ("(", ")") or token.upper() in OPERATORS else correct(token, lang, zone)
            for token in tokens
        ]
    except Exception:  # a failed suggestion must never break the search
        return None

    suggestion, original = tidy(fixed), tidy(tokens)
    return suggestion if suggestion.lower() != original.lower() else None


def run_search(query: str, mode: str, lang: str, zone: str) -> list[str]:
    if mode == "Boolean (AND / OR / NOT)":
        return search_boolean(query, lang=lang, zone=zone)
    return load_index(lang).phrase_search(query, zone=zone, lang=lang)


def use_suggestion(text: str) -> None:
    st.session_state["adv_query"] = text


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Advanced Search")
st.write("Exact keyword search over the verses. For questions in everyday words, use the Ask page instead.")

left, middle, right, far_right = st.columns([2, 1.2, 1.2, 1.2])
with left:
    mode = st.radio("Search type", ["Boolean (AND / OR / NOT)", "Phrase"], horizontal=True)
with middle:
    language = st.selectbox("Language", list(LANGUAGES))
with right:
    zone_options = list(ZONES) if mode.startswith("Boolean") else ["Translation", "Commentary"]
    zone_label = st.selectbox("Search in", zone_options)
with far_right:
    chapter = st.selectbox("Chapter", ["All chapters", *range(1, 19)])

lang, zone = LANGUAGES[language], ZONES[zone_label]

if mode.startswith("Boolean"):
    placeholder = "karma AND duty NOT war" if lang == "en" else "कर्म AND धर्म"
    st.caption("Use AND, OR, NOT and brackets, for example: (karma OR action) AND duty NOT war")
else:
    placeholder = "fruits of action" if lang == "en" else "कर्म का फल"
    st.caption("Finds verses containing these words next to each other, in this order. "
               "Phrase search covers translations and commentary only.")

query = st.text_input("Search", key="adv_query", placeholder=placeholder)

if query.strip():
    suggestion = suggest(query, lang, zone)
    if suggestion:
        note, button = st.columns([4, 1])
        note.info(f"Did you mean: **{suggestion}**?")
        button.button("Search for this", on_click=use_suggestion, args=(suggestion,))

    try:
        found = run_search(query, mode, lang, zone)
    except FileNotFoundError:
        st.error("The search index is missing. Run  python build_indexes.py  and reload this page.")
        st.stop()
    except Exception as err:  # bad brackets, a word that cleans to nothing, and so on
        st.error(f"Could not run this search: {err}")
        st.stop()

    if chapter != "All chapters":
        found = [vid for vid in found if int(vid.split(".")[0]) == chapter]

    st.subheader(f"{len(found)} verse(s) found")
    if not found:
        st.write("No verse matches. Try fewer words, OR instead of AND, or another zone.")

    verses = load_verses()
    for verse_id in found[:MAX_SHOWN]:
        verse = verses.get(verse_id)
        if not verse:
            continue
        with st.container(border=True):
            st.markdown(f"**Bhagavad Gita {verse_id}**")
            st.write(show_text(verse, lang))
            with st.expander("Sanskrit and transliteration"):
                st.markdown(verse.get("sanskrit", ""))
                st.caption(verse.get("transliteration", ""))
    if len(found) > MAX_SHOWN:
        st.caption(f"Showing the first {MAX_SHOWN} of {len(found)}. Add a word or a chapter filter to narrow it down.")
