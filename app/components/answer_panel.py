"""
Answer panel for the Streamlit app (Person 3: Anvi).

Shows one answer from pipeline.ask() the way the demo needs it:
  - an English / हिन्दी / Hinglish toggle (defaults to the language the user asked in)
  - each sentence that states what a verse says, with a ✓ (it passed the fact-checker)
    and the verse ids it cites as small tags
  - the gentler "application" sentences in italics, with no ✓ (they make no claim
    about the text, so there is nothing to check)
  - the safety note, when the question suggests serious distress
  - the cited verses (Sanskrit, transliteration, translation, explanation)
  - what the fact-checker removed, so the audience can see it working

Use on the Ask page:
    from pipeline import ask, verses_by_id
    result = ask(question)
    render_answer_panel(result["answer"], verses_by_id(), default_lang=result["lang"])

Why the ✓ marks survive translation: every sentence left after the fact-checker
that cites a verse was supported, and translate_answer() keeps the [2.47]
citations in place, so in any language "has a citation" means "checked and supported".

Try it on its own (from the repo root):
    streamlit run app/components/answer_panel.py
"""

import html
import os
import re
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generation.fact_check import CITATION, split_sentences  # noqa: E402
from generation.translate import translate_answer  # noqa: E402

LANGS = {"English": "en", "हिन्दी": "hi", "Hinglish": "hinglish"}

CSS = """
<style>
.saarthi-verse {margin: 0 0 .55rem 0; line-height: 1.6;}
.saarthi-tick {color: #2e7d32; font-weight: 700; margin-right: .35rem;}
.saarthi-app {margin: 0 0 .55rem 0; line-height: 1.6; font-style: italic; opacity: .8;}
.saarthi-pill {display: inline-block; font-size: .78rem; font-weight: 600; padding: .05rem .45rem;
               margin-left: .3rem; border-radius: .6rem; background: rgba(46,125,50,.12); color: #2e7d32;}
</style>
"""


def sentence_html(sentence: str) -> str:
    """One sentence as HTML: ✓ + text + verse tags if it cites a verse, else italic."""
    ids = CITATION.findall(sentence)
    text = " ".join(CITATION.sub("", sentence).split())
    body = html.escape(re.sub(r"\s+([.,;:!?।])", r"\1", text))  # "alone ." -> "alone."
    if not ids:
        return f'<p class="saarthi-app">{body}</p>'
    tags = "".join(f'<span class="saarthi-pill">{html.escape(v)}</span>' for v in dict.fromkeys(ids))
    return f'<p class="saarthi-verse"><span class="saarthi-tick">✓</span>{body}{tags}</p>'


def verse_translation(verse: dict, lang: str) -> str:
    keys = (["hi_ramsukhdas", "hi_tejomayananda"] if lang == "hi" else []) + ["en_gambirananda", "en_sivananda"]
    return next((verse["translations"][k] for k in keys if verse["translations"].get(k)), "")


def render_answer_panel(answer: dict, verses_by_id: dict, default_lang: str = "en", key: str = "answer") -> None:
    """Draw the answer panel.

    Args:
        answer: pipeline.ask(...)["answer"] (the output of pipeline.check_answer)
        verses_by_id: {verse id: verse record}, e.g. pipeline.verses_by_id()
        default_lang: "en", "hi" or "hinglish" (use the language the question was asked in)
        key: keeps widgets apart if a page shows two panels
    """
    if not answer:
        st.warning("No verses were found for this question, so there is no answer to show.")
        return

    st.markdown(CSS, unsafe_allow_html=True)

    labels = list(LANGS)
    index = list(LANGS.values()).index(default_lang) if default_lang in LANGS.values() else 0
    lang = LANGS[st.radio("Answer language", labels, index=index, horizontal=True, key=f"{key}_lang")]

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
    for sentence in split_sentences(text):
        st.markdown(sentence_html(sentence), unsafe_allow_html=True)
    st.caption("✓ = checked against the cited verse by our fact-checker · "
               "italic lines are gentle suggestions, not claims about the text")

    if answer["citations"]:
        st.markdown("**Verses cited**")
        for vid in answer["citations"]:
            verse = verses_by_id.get(vid)
            if not verse:
                continue
            with st.expander(f"Chapter {verse['chapter']}, verse {verse['verse']}  ·  {vid}"):
                if verse.get("sanskrit"):
                    st.markdown(verse["sanskrit"].replace("\n", "  \n"))
                if verse.get("transliteration"):
                    st.caption(verse["transliteration"])
                st.write(verse_translation(verse, lang))
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