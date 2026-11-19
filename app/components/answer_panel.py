"""Answer panel: ✓ sentences checked by the fact-checker, italic suggestions, the cited verses,
what the checker removed, and an English / हिन्दी / Hinglish toggle.

Try it alone: streamlit run app/components/answer_panel.py
"""

import html
import os
import re
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generation.fact_check import CITATION, split_sentences  # noqa: E402
from generation.translate import translate_answer  # noqa: E402
from app.components.style import speaker_name  # noqa: E402
from app.components.verse_card import render_verse_card  # noqa: E402

LANGS = {"English": "en", "हिन्दी": "hi", "Hinglish": "hinglish"}

CSS = """
<style>
.saarthi-verse {margin: 0 0 .6rem 0; line-height: 1.7; font-size: 1.08rem;}
.saarthi-tick {color: #3B7A3B; font-weight: 700; margin-right: .4rem;}
.saarthi-app {margin: 0 0 .6rem 0; line-height: 1.7; font-size: 1.05rem; font-style: italic; color: #6B4A36;}
.saarthi-pill {display: inline-block; font-size: .78rem; font-weight: 600; padding: .05rem .5rem;
               margin-left: .35rem; border-radius: .7rem; background: rgba(200,144,42,.18); color: #7A4A0E;}
</style>
"""


def sentence_html(sentence: str, delay: float = 0.0) -> str:
    """One sentence as HTML: ✓ + text + verse tags if it cites a verse, else italic.
    delay (seconds) makes the sentences fade in one after another."""
    ids = CITATION.findall(sentence)
    text = " ".join(CITATION.sub("", sentence).split())
    body = html.escape(re.sub(r"\s+([.,;:!?।])", r"\1", text))  # "alone ." -> "alone."
    reveal = f' saarthi-reveal" style="animation-delay:{delay:.2f}s' if delay else ""
    if not ids:
        return f'<p class="saarthi-app{reveal}">{body}</p>'
    tags = "".join(f'<span class="saarthi-pill">{html.escape(v)}</span>' for v in dict.fromkeys(ids))
    return f'<p class="saarthi-verse{reveal}"><span class="saarthi-tick">✓</span>{body}{tags}</p>'


def render_answer_panel(answer: dict, verses_by_id: dict, default_lang: str = "en", key: str = "answer") -> None:
    """Draw pipeline.ask(...)["answer"]; default_lang is the language the question was asked in."""
    if not answer:
        st.warning("No verses were found for this question, so there is no answer to show.")
        return

    st.markdown(CSS, unsafe_allow_html=True)

    # Start each new answer in the language it was asked in (Streamlit would
    # otherwise keep the toggle's last choice from the previous question).
    labels = list(LANGS)
    default_label = next((label for label, code in LANGS.items() if code == default_lang), labels[0])
    if st.session_state.get(f"{key}_answer") != answer["text_en"]:
        st.session_state[f"{key}_answer"] = answer["text_en"]
        st.session_state[f"{key}_lang"] = default_label
    lang = LANGS[st.radio("Answer language", labels, horizontal=True, key=f"{key}_lang")]

    def show(text: str) -> str:
        if lang == "en" or not text:
            return text
        with st.spinner("Translating..."):
            return translate_answer(text, lang)

    if answer.get("safety_note"):
        st.info(show(answer["safety_note"]))

    text = show(answer["text_en"])
    if not text.strip():
        st.warning("None of the generated sentences passed the fact-checker, so there is no answer to show.")
    for i, sentence in enumerate(split_sentences(text)):
        st.markdown(sentence_html(sentence, delay=0.15 + 0.45 * i), unsafe_allow_html=True)
    st.caption("✓ = checked against the cited verse by our fact-checker · "
               "italic lines are gentle suggestions, not claims about the text")

    if answer["citations"]:
        st.markdown("**Verses cited**")
        for vid in answer["citations"]:
            verse = verses_by_id.get(vid)
            if not verse:
                continue
            with st.expander(f"Chapter {verse['chapter']}, verse {verse['verse']}  ·  {speaker_name(verse)}"):
                render_verse_card(verse, lang, border=False, title=False)
                if lang == "en" and verse.get("explanation_en"):
                    st.caption("In simple words: " + verse["explanation_en"])

    if answer["removed"]:
        with st.expander(f"Removed by the fact-checker ({len(answer['removed'])})"):
            st.caption("These sentences claimed more than their cited verse says, so they were not shown. "
                       "(Shown in English, as generated.)")
            for c in answer["removed"]:
                st.markdown(f"- ~~{html.escape(c['sentence'])}~~  ·  support score {c['score']}")


# ---------------------------------------------------------------------------
# Stand-alone test app:  streamlit run app/components/answer_panel.py
# ---------------------------------------------------------------------------

def _demo() -> None:
    from pipeline import ask, verses_by_id

    st.set_page_config(page_title="Saarthi answer panel", layout="centered")
    st.title("Saarthi")
    st.caption("Answer panel test, using the full pipeline")
    question = st.text_input("Ask anything, in English, Hindi or Hinglish")
    if not question.strip():
        return

    if st.session_state.get("demo_question") != question:
        with st.spinner("Finding verses and writing an answer..."):
            st.session_state.demo_result = ask(question)
            st.session_state.demo_question = question

    result = st.session_state.demo_result
    render_answer_panel(result["answer"], verses_by_id(), default_lang=result["lang"], key="demo")


if __name__ == "__main__":
    _demo()