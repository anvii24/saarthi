"""
NLI fact-checker (Person 3: Anvi).

verify() checks every sentence of the explanation against the verse it cites
and marks whether the verse actually supports it.

STUB: marks every cited sentence as supported until the NLI model is added.
"""

import re

CITATION = re.compile(r"\[(\d{1,2}\.\d{1,2})\]")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? and the Hindi full stop ।"""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p for p in parts if p]


def verify(explanation: str, verses: list[dict]) -> list[dict]:
    """Return one entry per sentence: {"sentence", "verse_id", "supported"}.

    TODO: load a pre-trained NLI model, run (verse text -> sentence),
    supported = label is "entailment".
    """
    checks = []
    for sentence in split_sentences(explanation):
        match = CITATION.search(sentence)
        verse_id = match.group(1) if match else None
        checks.append({
            "sentence": sentence,
            "verse_id": verse_id,
            "supported": verse_id is not None,  # stub: no citation = unsupported
        })
    return checks


if __name__ == "__main__":
    text = "Focus on your actions, not results [2.47]. Success is guaranteed."
    for c in verify(text, []):
        print(c)
