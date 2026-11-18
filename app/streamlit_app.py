"""Saarthi home page. Run: streamlit run app/streamlit_app.py"""

import streamlit as st

st.set_page_config(
    page_title="Saarthi",
    page_icon="🕉️",
    layout="wide"
)

st.title("Saarthi")
st.subheader("Find guidance in the Bhagavad Gita")

st.write(
    "Saarthi is a Bhagavad Gita-based question-answering system that helps "
    "you explore the teachings of the Gita in response to everyday questions. "
    "It uses language processing, semantic search and retrieval techniques "
    "to find relevant verses and generate answers grounded in the original text."
)

st.info("You can ask questions in English, Hindi or Hinglish.")

st.markdown("### Get started")
st.write(
    "Use the pages in the sidebar to ask a question, explore verses by chapter, "
    "and discover connections between the teachings of the Bhagavad Gita."
)
