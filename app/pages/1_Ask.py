import os
import sys

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline import ask, verses_by_id
from app.components.answer_panel import render_answer_panel
from app.components.verse_card import render_verse_card

st.title("Ask Saarthi")
st.write(
    "Ask a question about your life, challenges or decisions. "
    "Saarthi will find relevant Bhagavad Gita verses and provide "
    "an answer grounded in their teachings."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. I am worried about my exams. What should I do?"
)

if st.button("Find guidance", type="primary", disabled=not question.strip()):
    with st.spinner("Understanding your question and finding relevant verses..."):
        try:
            st.session_state.ask_result = ask(question)
            st.session_state.ask_question = question
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

if st.session_state.get("ask_result"):
    result = st.session_state.ask_result

    if st.session_state.get("ask_question") == question:
        st.divider()

        st.subheader("Understanding your question")
        st.write(f"**Detected language:** {result['lang']}")
        st.write(f"**Predicted theme:** {result['theme']}")

        if result["concepts"]:
            st.write("**Related Gita concepts:** " + ", ".join(result["concepts"]))

        st.divider()

        st.subheader("Guidance from the Bhagavad Gita")
        render_answer_panel(
            result["answer"],
            verses_by_id(),
            default_lang=result["lang"],
            key="ask"
        )

        st.divider()

        st.subheader("Top 5 relevant verses")
        for verse in result["verses"]:
            render_verse_card(verse, result["lang"])
