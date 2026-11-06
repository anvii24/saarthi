"""
LLM query rewriting (Person 3: Anvi).

rewrite() turns an everyday question into the underlying Gita concepts,
e.g. "I failed my exam" -> ["attachment to results", "equanimity", "duty"].

STUB: returns fixed dummy concepts until the LLM call is added.
"""


def rewrite(query_en: str) -> list[str]:
    """Return 3-5 Gita concepts for an English question.

    TODO: call the LLM (key from .env), ask for a JSON list of concepts,
    and cache responses in data/processed/rewrite_cache.json.
    """
    return ["duty", "action", "equanimity"]  # stub


if __name__ == "__main__":
    print(rewrite("I studied hard but still failed my exam"))
