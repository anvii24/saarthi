"""
RAG explanation (Person 3: Anvi).

explain() gives the LLM the user's question plus the top retrieved verses and asks
for a short explanation that uses ONLY those verses, citing [chapter.verse] after
every claim. The explanation is written in English so the fact-checker can verify
it; translate_answer() then converts it to Hindi or Hinglish for the user.

Safety: if the question suggests serious distress, a fixed support note is added
(Tele-MANAS, the Government of India's free mental-health helpline: 14416).
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.llm import ask  # noqa: E402

CITATION = re.compile(r"\[(\d{1,2}\.\d{1,3})\]")

SYSTEM = (
    "You explain verses of the Bhagavad Gita to someone asking an everyday life question. "
    "Rules:\n"
    "1. Use ONLY the verses provided. Never quote or mention any other verse.\n"
    "2. Put the verse id in square brackets, like [2.47], at the end of every sentence "
    "that says what a verse teaches.\n"
    "3. Only say what the verse text actually says; do not add promises, outcomes or "
    "interpretations the text does not state.\n"
    "4. Be warm, simple and non-preachy. No commands like 'you must'. Present it as what "
    "the Gita suggests, not as absolute truth.\n"
    "5. 4 to 7 sentences, plain English, no headings or lists."
)

PROMPT = "Question: {question}\n\nVerses:\n{verses}\n\nWrite the explanation."

DISTRESS_WORDS = [
    "suicide", "kill myself", "end my life", "want to die", "self harm", "self-harm",
    "no reason to live", "hurt myself", "marna chahta", "marna chahti", "jeena nahi",
    "आत्महत्या", "मरना चाहता", "मरना चाहती", "जीना नहीं",
]
SAFETY_NOTE = (
    "If you are going through something very heavy right now, please talk to someone you "
    "trust or call Tele-MANAS on 14416 (free, 24x7, in many Indian languages). "
    "You don't have to handle this alone."
)

TRANSLATIONS_PER_VERSE = ["en_gambirananda", "en_prabhupada", "en_sivananda", "en_adidevananda"]


def format_verses(verses: list[dict]) -> str:
    """Each verse as '[2.47] translation one / translation two'."""
    lines = []
    for verse in verses:
        texts = [verse["translations"][k] for k in TRANSLATIONS_PER_VERSE if verse["translations"].get(k)]
        if not texts:
            texts = [t for k, t in verse["translations"].items() if k.startswith("en_")]
        lines.append(f"[{verse['id']}] " + " / ".join(texts[:2]))
    return "\n".join(lines)


def needs_safety_note(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in DISTRESS_WORDS)


def explain(question: str, verses: list[dict]) -> dict:
    """Write an English explanation that uses only the given verses.

    Args:
        question: the user's question (English version from process_query)
        verses: top 3-5 verse records (schemas.Verse)

    Returns:
        {"text": str, "citations": ["2.47", ...], "safety_note": str | None}
    """
    if not verses:
        return {"text": "I couldn't find verses that clearly answer this question.",
                "citations": [], "safety_note": None}

    text = ask(PROMPT.format(question=question, verses=format_verses(verses)),
               system=SYSTEM, temperature=0.3).strip()

    allowed = {v["id"] for v in verses}
    citations = [vid for vid in dict.fromkeys(CITATION.findall(text)) if vid in allowed]
    return {
        "text": text,
        "citations": citations,
        "safety_note": SAFETY_NOTE if needs_safety_note(question) else None,
    }


if __name__ == "__main__":
    import json

    with open(os.path.join("data", "verses.json"), encoding="utf-8") as f:
        by_id = {v["id"]: v for v in json.load(f)}
    answer = explain("I studied hard but still failed my exam. How do I cope?",
                     [by_id["2.47"], by_id["2.48"], by_id["2.38"]])
    print(answer)
