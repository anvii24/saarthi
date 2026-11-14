"""
RAG explanation

explain() gives the LLM the user's question plus the top retrieved verses and asks
for a short explanation that uses ONLY those verses, citing [chapter.verse] after
every claim about a verse. The explanation is written in English so the
fact-checker can verify it; translate_answer() then converts it to Hindi or
Hinglish for the user.

Two prompt versions:
  v1  first version. The LLM read Gambhirananda + Prabhupada translations, but the
      fact-checker verifies against the first two English translations (Sivananda +
      Purohit), so faithful sentences could still look "unsupported".
  v2  fixes that: the LLM reads exactly the translations the checker uses, and the
      rules separate two kinds of sentence:
        - verse sentences: say what ONE verse says, close to its wording, cite it
        - application sentences: relate it to the user's situation, no citation,
          phrased tentatively, never claiming "the Gita says"
  evaluation/run_rag.py --prompt v1 / v2 compares them.

Safety: if the question suggests serious distress, a fixed support note is added
(Tele-MANAS, the Government of India's free mental-health helpline: 14416).
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.llm import ask  # noqa: E402

DEFAULT_PROMPT = "v2"  

CITATION = re.compile(r"\[(\d{1,2}\.\d{1,3})\]")

SYSTEM_V1 = (
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

SYSTEM_V2 = (
    "You explain verses of the Bhagavad Gita to someone asking an everyday life question. "
    "Use ONLY the verses provided; never mention any other verse.\n\n"
    "Write two kinds of sentences:\n"
    "A) VERSE sentences: say what ONE verse says, staying close to its own wording, and end "
    "with that verse's id in square brackets, like [2.47]. Cite exactly one verse per "
    "sentence. Do not add anything the verse text does not say: no promises, outcomes, "
    "reasons, or meanings of your own.\n"
    "B) APPLICATION sentences: gently connect those verses to the person's situation. They "
    "have NO citation, are phrased tentatively ('This might mean...', 'In your situation, "
    "perhaps...'), and never claim that the Gita or a verse says something.\n\n"
    "Write 2 to 4 verse sentences and 1 to 3 application sentences, 4 to 7 sentences in all. "
    "Be warm, simple and non-preachy; no commands like 'you must'. Plain English, no headings "
    "or lists."
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

TRANSLATIONS_V1 = ["en_gambirananda", "en_prabhupada", "en_sivananda", "en_adidevananda"]


def texts_v1(verse: dict) -> list[str]:
    texts = [verse["translations"][k] for k in TRANSLATIONS_V1 if verse["translations"].get(k)]
    return texts or [t for k, t in verse["translations"].items() if k.startswith("en_")]


def texts_v2(verse: dict) -> list[str]:
    return [t for k, t in verse["translations"].items() if k.startswith("en_") and t]


def format_verses(verses: list[dict], version: str = "v1") -> str:
    """Each verse as '[2.47] translation one / translation two'."""
    pick = texts_v2 if version == "v2" else texts_v1
    return "\n".join(f"[{v['id']}] " + " / ".join(pick(v)[:2]) for v in verses)


def needs_safety_note(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in DISTRESS_WORDS)


def explain(question: str, verses: list[dict], prompt_version: str = DEFAULT_PROMPT) -> dict:

    if not verses:
        return {"text": "I couldn't find verses that clearly answer this question.",
                "citations": [], "safety_note": None}

    system = SYSTEM_V2 if prompt_version == "v2" else SYSTEM_V1
    text = ask(PROMPT.format(question=question, verses=format_verses(verses, prompt_version)),
               system=system, temperature=0.3).strip()

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
    picked = [by_id["2.47"], by_id["2.48"], by_id["2.38"]]
    for version in ("v1", "v2"):
        print(f"--- {version} ---")
        print(explain("I studied hard but still failed my exam. How do I cope?", picked, version))