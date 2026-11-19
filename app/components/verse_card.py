"""One verse as a card: who speaks it, the Sanskrit, transliteration and a translation."""

import html

import streamlit as st

from app.components.style import speaker_name


def translation_for(verse: dict, lang: str) -> str:
    """A Hindi translation for lang 'hi', otherwise English.

    English prefers Adidevananda: the source copies of Gambhirananda and Sivananda lost
    every "qu" (e.g. "alities" for "qualities").
    """
    translations = verse.get("translations", {})
    keys = ["hi_ramsukhdas", "hi_tejomayananda"] if lang == "hi" else ["en_adidevananda", "en_gambirananda"]
    return next((translations[k] for k in keys if translations.get(k)), "Translation unavailable.")


def render_verse_card(verse: dict, lang: str, border: bool = True, title: bool = True) -> None:
    """Show one verse; lang 'hi' shows a Hindi translation, anything else English."""
    with st.container(border=border):
        if title:
            st.markdown(
                f"<div class='saarthi-verse-title'>Chapter {verse['chapter']} · Verse {verse['verse']}"
                f"<span class='saarthi-speaker'>{html.escape(speaker_name(verse))}</span></div>",
                unsafe_allow_html=True,
            )
        if verse.get("sanskrit"):
            lines = "<br>".join(html.escape(line) for line in verse["sanskrit"].splitlines())
            st.markdown(f"<div class='saarthi-sanskrit'>{lines}</div>", unsafe_allow_html=True)
        if verse.get("transliteration"):
            lines = "<br>".join(html.escape(line) for line in verse["transliteration"].splitlines())
            st.markdown(f"<div class='saarthi-translit'>{lines}</div>", unsafe_allow_html=True)
        st.write(translation_for(verse, lang))
