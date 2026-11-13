import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.llm import ask_json  # noqa: E402

SYSTEM = (
    "You help a search engine over the Bhagavad Gita. Given a modern, everyday "
    "question, list the philosophical concepts from the Gita that the question is "
    "really about. Use plain English phrases of 1-5 words that are likely to appear "
    "in English translations of the Gita (e.g. 'fruits of action', 'duty', "
    "'steady mind', 'attachment', 'desire and anger', 'the self'). "
    "Do not quote verses and do not give advice."
)

PROMPT = (
    'Question: "{question}"\n\n'
    'Reply as JSON: {{"concepts": ["...", "..."]}} with 3 to 5 concepts.'
)


def rewrite(query_en: str) -> list[str]:
    """Return 3-5 Gita concepts for an English question (empty list on failure)."""
    if not query_en.strip():
        return []
    try:
        reply = ask_json(PROMPT.format(question=query_en.strip()), system=SYSTEM)
    except Exception as err:  # no key, no credits, no internet: search still works without concepts
        print(f"[rewrite] LLM call failed, searching without concepts: {err}")
        return []
    concepts = reply.get("concepts", [])
    return [str(c).strip() for c in concepts if str(c).strip()][:5]


def expanded_query(query_en: str) -> str:
    """The question plus its concepts, ready to pass to BM25 or dense search."""
    concepts = rewrite(query_en)
    return query_en if not concepts else f"{query_en} {' '.join(concepts)}"


if __name__ == "__main__":
    q = "I studied really hard but still failed my exam. How do I stop feeling like a failure?"
    print(rewrite(q))
    print(expanded_query(q))