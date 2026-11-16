import html
import json
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generation.fact_check import CITATION, keep_supported, split_sentences, verify  # noqa: E402
from generation.rag import explain  # noqa: E402
from generation.translate import translate_answer  # noqa: E402

LANGS = {"English": "en", "हिन्दी": "hi", "Hinglish": "hinglish"}
VERSES_PATH = os.path.join("data", "verses.json")

CSS = """
<style>
.saarthi-verse {margin: 0 0 .55rem 0; line-height: 1.6;}
.saarthi-tick {color: #2e7d32; font-weight: 700; margin-right: .35rem;}
.saarthi-app {margin: 0 0 .55rem 0; line-height: 1.6; font-style: italic; opacity: .8;}
.saarthi-pill {display: inline-block; font-size: .78rem; font-weight: 600; padding: .05rem .45rem;
               margin-left: .3rem; border-radius: .6rem; background: rgba(46,125,50,.12); color: #2e7d32;}
</style>
"""


def prepare_answer(question_en: str, verses: list[dict]) -> dict:
    """Explain, fact-check, and keep only what the checker accepts.

    Returns:
        {"text_en": verified English answer,
         "citations": ["2.47", ...] verse ids cited in the verified answer,
         "removed": [{"sentence", "verse_ids", "score"}, ...] what the checker dropped,
         "safety_note": str | None}
    """
    answer = explain(question_en, verses)
    checks = verify(answer["text"], verses)
    text_en = keep_supported(checks)
    return {
        "text_en": text_en,
        "citations": list(dict.fromkeys(CITATION.findall(text_en))),
        "removed": [c for c in checks if c["supported"] is False],
        "safety_note": answer["safety_note"],
    }


def sentence_html(sentence: str) -> str:
    """One sentence as HTML: ✓ + text + verse tags if it cites a verse, else italic."""
    ids = CITATION.findall(sentence)
    body = html.escape(" ".join(CITATION.sub("", sentence).split()))
    if not ids:
        return f'<p class="saarthi-app">{body}</p>'
    tags = "".join(f'<span class="saarthi-pill">{html.escape(v)}</span>' for v in dict.fromkeys(ids))
    return f'<p class="saarthi-verse"><span class="saarthi-tick">✓</span>{body}{tags}</p>'


def verse_translation(verse: dict, lang: str) -> str:
    keys = (["hi_ramsukhdas", "hi_tejomayananda"] if lang == "hi" else []) + ["en_gambirananda", "en_sivananda"]
    return next((verse["translations"][k] for k in keys if verse["translations"].get(k)), "")


def render_answer_panel(prepared: dict, verses_by_id: dict, default_lang: str = "en", key: str = "answer") -> None:
    """Draw the answer panel. `key` keeps widgets apart if a page shows two panels."""
    st.markdown(CSS, unsafe_allow_html=True)

    labels = list(LANGS)
    index = list(LANGS.values()).index(default_lang) if default_lang in LANGS.values() else 0
    lang = LANGS[st.radio("Answer language", labels, index=index, horizontal=True, key=f"{key}_lang")]

    def show(text: str) -> str:
        if lang == "en" or not text:
            return text
        with st.spinner("Translating..."):
            return translate_answer(text, lang)

    if prepared.get("safety_note"):
        st.info(show(prepared["safety_note"]))

    text = show(prepared["text_en"])
    if not text.strip():
        st.warning("None of the generated sentences passed the fact-checker, so there is no answer to show.")
    for sentence in split_sentences(text):
        st.markdown(sentence_html(sentence), unsafe_allow_html=True)
    st.caption("✓ = checked against the cited verse by our fact-checker · "
               "italic lines are gentle suggestions, not claims about the text")

    if prepared["citations"]:
        st.markdown("**Verses cited**")
        for vid in prepared["citations"]:
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

    if prepared["removed"]:
        with st.expander(f"Removed by the fact-checker ({len(prepared['removed'])})"):
            st.caption("These sentences claimed more than their cited verse says, so they were not shown. "
                       "(Shown in English, as generated.)")
            for c in prepared["removed"]:
                st.markdown(f"- ~~{html.escape(c['sentence'])}~~  ·  support score {c['score']}")

@st.cache_resource
def _load_verses() -> dict:
    with open(VERSES_PATH, encoding="utf-8") as f:
        return {v["id"]: v for v in json.load(f)}


def _demo() -> None:
    from indexing.embeddings import search_dense
    from query.language import process_query
    from query.rewrite import rewrite

    st.set_page_config(page_title="Saarthi answer panel", layout="centered")
    st.title("Saarthi")
    st.caption("Answer panel test: meaning search only, until the full pipeline is wired up")
    question = st.text_input("Ask anything, in English, Hindi or Hinglish")
    if not question.strip():
        return

    verses_by_id = _load_verses()
    if st.session_state.get("demo_question") != question:
        with st.spinner("Finding verses and writing an answer..."):
            pq = process_query(question)
            concepts = rewrite(pq["english"])
            query = f"{pq['english']} {' '.join(concepts)}" if concepts else pq["english"]
            top = [verses_by_id[r["id"]] for r in search_dense(query, k=5)]
            st.session_state.demo_question = question
            st.session_state.demo_question_lang = pq["lang"]
            st.session_state.demo_prepared = prepare_answer(pq["english"], top)

    render_answer_panel(st.session_state.demo_prepared, verses_by_id,
                        default_lang=st.session_state.demo_question_lang, key="demo")


if __name__ == "__main__":
    _demo()