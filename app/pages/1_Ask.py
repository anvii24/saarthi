"""Ask page: a question in any language -> a checked answer and its verses."""

import os
import sys

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.components.answer_panel import render_answer_panel  # noqa: E402
from app.components.background import run_with_loader, start_warm_up  # noqa: E402
from app.components.decor import lotus_divider, page_header  # noqa: E402
from app.components.style import LANG_NAMES, apply_style, chips, footer, nice_theme, speaker_name  # noqa: E402
from app.components.verse_card import render_verse_card  # noqa: E402
from pipeline import ask, verses_by_id  # noqa: E402

EXAMPLES = [
    ("English", "I studied hard but still failed my exam. How do I cope?"),
    ("हिन्दी", "मुझे बहुत गुस्सा आता है, मैं इसे कैसे नियंत्रित करूँ?"),
    ("Hinglish", "Career ko lekar bahut confused hoon, kya karun?"),
]

st.set_page_config(page_title="Ask · Saarthi", page_icon="🪔", layout="centered")
apply_style()
start_warm_up()


def use_example(question: str) -> None:
    st.session_state["ask_q"] = question
    st.session_state["ask_run"] = True


def run(question: str) -> None:
    try:
        st.session_state["ask_result"] = run_with_loader(
            lambda on_step: ask(question, on_step=on_step), "Reading your question",
            note="the first question after starting the app takes the longest")
        st.session_state["ask_question"] = question
    except Exception as err:
        st.error(f"Saarthi could not answer just now ({err}). Check the internet connection and the "
                 "API key in .env, then try again.")


# A question chosen on the home page arrives here.
if "ask_prefill" in st.session_state:
    st.session_state["ask_q"] = st.session_state.pop("ask_prefill")
    st.session_state["ask_run"] = True

page_header("Ask Saarthi", "krishna_flute")
st.write("Ask about anything on your mind, in English, Hindi or Hinglish.")

with st.form("ask_form", border=False):
    question = st.text_input("Your question", key="ask_q", label_visibility="collapsed",
                             placeholder="e.g. Exam mein fail ho gaya, ab kya karun?")
    submitted = st.form_submit_button("Find guidance", type="primary")

st.caption("Or try one:")
for column, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    column.button(label, key=f"ex_{label}", use_container_width=True, on_click=use_example, args=(text,),
                  help=text)

if submitted and question.strip():
    run(question.strip())
elif st.session_state.pop("ask_run", False) and st.session_state.get("ask_q", "").strip():
    run(st.session_state["ask_q"].strip())

result = st.session_state.get("ask_result")
if result:
    st.markdown(f"<div class='saarthi-quote'>“{result['question']}”</div>", unsafe_allow_html=True)
    st.markdown(
        chips("Language", [LANG_NAMES.get(result["lang"], result["lang"])])
        + chips("Theme", [nice_theme(result["theme"])])
        + (chips("Gita ideas", result["concepts"]) if result["concepts"] else ""),
        unsafe_allow_html=True,
    )

    st.subheader("What the Gita says")
    with st.container(border=True):
        render_answer_panel(result["answer"], verses_by_id(), default_lang=result["lang"], key="ask")

    lotus_divider()
    st.subheader("All verses Saarthi found")
    cited = set(result["answer"]["citations"]) if result["answer"] else set()
    for verse in result["verses"]:
        mark = "  ·  ✓ cited above" if verse["id"] in cited else ""
        with st.expander(f"Chapter {verse['chapter']}, verse {verse['verse']}  ·  {speaker_name(verse)}{mark}"):
            render_verse_card(verse, result["lang"], border=False, title=False)

footer()
