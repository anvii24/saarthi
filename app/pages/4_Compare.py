"""
Compare page (Person 2: Kashvi).

One question, four rankings side by side, so the audience can see what each step
of the search changes (these are the intermediate lists that pipeline.retrieve() returns):

  1. keyword search   BM25 over the English translations
  2. meaning search   multilingual embeddings
  3. fused            the two lists merged with Reciprocal Rank Fusion
  4. final            fused list re-ranked with PageRank and topic PageRank

The question is searched in English, together with the Gita concepts from query
rewriting, exactly as in the main pipeline.
"""

import os
import sys
from functools import lru_cache

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.background import run_with_loader, start_warm_up  # noqa: E402
from app.components.decor import page_header  # noqa: E402
from app.components.style import apply_style, footer  # noqa: E402
from pipeline import retrieve, verses_by_id  # noqa: E402
from query.classifier import classify_theme  # noqa: E402
from query.rewrite import rewrite  # noqa: E402

EXAMPLE = "I studied really hard but still failed my exam. How do I stop feeling like a failure?"

COLUMNS = [
    ("bm25", "1. Keyword search", "BM25"),
    ("dense", "2. Meaning search", "embeddings"),
    ("fused", "3. Fused", "RRF of 1 and 2"),
    ("results", "4. Final", "+ PageRank and theme"),
]

st.set_page_config(page_title="Compare · Saarthi", page_icon="⚖️", layout="wide")
apply_style()
start_warm_up()


@lru_cache(maxsize=32)  # runs in a background thread, so a plain cache instead of st.cache_data
def search_all(question: str) -> tuple[list[str], str, dict]:
    """Concepts, predicted theme and every intermediate ranking for one question."""
    concepts = rewrite(question)
    theme = classify_theme(question)["predicted_theme"]
    return concepts, theme, retrieve(question, concepts, theme)


def excerpt(verse_id: str, limit: int = 170) -> str:
    verse = verses_by_id().get(verse_id, {})
    translations = verse.get("translations", {})
    text = translations.get("en_gambirananda") or next(
        (t for key, t in translations.items() if key.startswith("en") and t), ""
    )
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def move_label(verse_id: str, position: int, fused_rank: dict) -> str:
    """How far the re-ranker moved a verse compared with the fused list."""
    before = fused_rank.get(verse_id)
    if before is None:
        return "new in the top list"
    if before > position:
        return f"▲ up {before - position} from fused"
    if before < position:
        return f"▼ down {position - before} from fused"
    return "same place as fused"


page_header("Compare the search steps", "sun")
st.write("See what each step of Saarthi's search changes for one question.")

with st.form("compare"):
    question = st.text_input("Question (English)", value=EXAMPLE)
    top_n = st.slider("Verses to show per column", 3, 10, 5)
    submitted = st.form_submit_button("Compare")

if submitted and question.strip():
    try:
        concepts, theme, out = run_with_loader(lambda on_step: search_all(question.strip()),
                                               "Rewriting the question and running every search step")
    except FileNotFoundError:
        st.error("No embeddings yet. Run  python indexing/embeddings.py  once, then reload this page.")
        st.stop()
    except Exception as err:
        st.error(f"Search failed: {err}")
        st.stop()

    if concepts:
        st.markdown("**Gita concepts added to the question:** " + ", ".join(concepts))
    else:
        st.warning("Query rewriting returned no concepts (check the LLM key in .env). "
                   "The rankings below use the question only.")
    if theme == "unclassified":
        st.caption("Predicted theme: none (no words matched the classifier), so topic PageRank was not used.")
    else:
        st.caption(f"Predicted theme: {theme.replace('_', ' ')}")

    fused_rank = {r["id"]: i for i, r in enumerate(out["fused"])}
    columns = st.columns(len(COLUMNS))

    for column, (key, title, subtitle) in zip(columns, COLUMNS):
        with column:
            st.subheader(title)
            st.caption(subtitle)
            for position, result in enumerate(out[key][:top_n]):
                with st.container(border=True):
                    st.markdown(f"**{position + 1}. Gita {result['id']}**  ·  score {result['score']:.3f}")
                    st.write(excerpt(result["id"]))
                    if key == "results":
                        st.caption(move_label(result["id"], position, fused_rank))

footer()
