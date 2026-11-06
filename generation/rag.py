"""
RAG explanation (Person 3: Anvi).

explain() asks the LLM to connect the top verses to the user's question,
citing chapter.verse after every claim, in the user's language.

STUB: returns a template explanation until the LLM call is added.
"""


def explain(question: str, verses: list[dict], lang: str = "en") -> dict:
    """Write an explanation that uses only the given verses.

    Args:
        question: the user's question
        verses: top 3-5 verse records (schemas.Verse)
        lang: "en", "hi" or "hinglish"

    Returns:
        {"text": str, "citations": ["2.47", ...]}

    TODO: build the prompt (use only these verses, cite chapter.verse,
    no preaching, answer in `lang`), call the LLM, add the distress safety line.
    """
    ids = [v["id"] for v in verses]
    text = " ".join(f"This idea appears in verse {vid} [{vid}]." for vid in ids)
    return {"text": text, "citations": ids}  # stub


if __name__ == "__main__":
    print(explain("How do I handle failure?", [{"id": "2.47"}, {"id": "2.48"}]))
