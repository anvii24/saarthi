"""
Answer translation (Person 3: Anvi).

translate_answer() converts the verified English explanation into Hindi
(Devanagari) or Hinglish (Hindi in Roman letters), keeping every [2.47]
citation exactly as it is so the answer panel can still link to the verses.
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.llm import ask  # noqa: E402

CITATION = re.compile(r"\[(\d{1,2}\.\d{1,3})\]")

TARGETS = {
    "hi": "natural, simple Hindi in Devanagari script",
    "hinglish": "Hinglish: casual Hindi written in Roman letters, the way people text in India",
}

SYSTEM = (
    "You translate short explanations about the Bhagavad Gita. Keep the meaning and tone "
    "exactly, add nothing, and keep every citation like [2.47] unchanged and in the same "
    "place in its sentence. Reply with the translation only."
)


def translate_answer(text: str, target_lang: str) -> str:
    """Translate an English explanation into "en", "hi" or "hinglish"."""
    if target_lang not in {"en", "hi", "hinglish"}:
        raise ValueError(f"unknown language: {target_lang}")
    if target_lang == "en" or not text.strip():
        return text

    translated = ask(f"Translate into {TARGETS[target_lang]}:\n\n{text}", system=SYSTEM).strip()

    # safety net: if citations went missing, fall back to the English text
    if set(CITATION.findall(translated)) != set(CITATION.findall(text)):
        print("[translate] citations changed during translation; showing English instead")
        return text
    return translated


if __name__ == "__main__":
    english = "The Gita suggests focusing on your effort rather than the result [2.47]."
    print(translate_answer(english, "hi"))
    print(translate_answer(english, "hinglish"))
