"""One verse as a card: Sanskrit, transliteration and a translation."""

import streamlit as st


def render_verse_card(verse: dict, lang: str) -> None:
    """Show one verse; lang 'hi' shows a Hindi translation, anything else English."""
    translations = verse.get("translations", {})

    if lang == "hi":
        translation_keys = ["hi_ramsukhdas", "hi_tejomayananda"]
    else:
        translation_keys = ["en_gambirananda", "en_sivananda"]

    translation = next(
        (translations[key] for key in translation_keys if translations.get(key)),
        "Translation unavailable."
    )

    with st.container(border=True):
        st.markdown(
            f"**Chapter {verse['chapter']}, Verse {verse['verse']}**"
        )

        if verse.get("speaker"):
            st.caption(f"Speaker: {verse['speaker']}")

        if verse.get("sanskrit"):
            st.markdown("**Sanskrit**")
            st.text(verse["sanskrit"])

        if verse.get("transliteration"):
            st.markdown("**Transliteration**")
            st.write(verse["transliteration"])

        st.markdown("**Translation**")
        st.write(translation)
