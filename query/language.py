"""
Language handling for user questions (Person 3: Anvi).

process_query() works out whether a question is English, Hindi or Hinglish,
and returns a Devanagari Hindi version and an English version, so both the
English and the Hindi indexes can be searched.

- Language detection: rule-based (Devanagari characters -> Hindi; Roman letters
  with common Hindi words -> Hinglish; otherwise English).
- Transliteration + translation: one LLM call (cached). We use the LLM rather
  than IndicTrans2 / IndicXlit because those models are large and hard to install
  on Windows, and the LLM handles Hinglish's loose spelling well.
  (Mention this choice in the report.)
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.llm import ask_json  # noqa: E402

# common Hindi words typed in Roman letters, used to spot Hinglish
HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kaise", "kyun", "kyon", "mera", "meri", "mere",
    "nahi", "nahin", "main", "mein", "ho", "gaya", "gayi", "karu", "karun",
    "karoon", "kar", "raha", "rahi", "tha", "thi", "aur", "bhi", "ko", "se",
    "ki", "ka", "ke", "par", "lekin", "kuch", "bahut", "accha", "acha", "hoon",
    "hun", "mujhe", "mujhko", "apne", "apna", "sab", "jab", "toh", "kyunki",
}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

SYSTEM = (
    "You convert user questions between English, Hindi (Devanagari script) and "
    "Hinglish (Hindi typed in Roman letters, often mixed with English words). "
    "Keep the meaning exactly; do not answer the question or add anything."
)

PROMPT = (
    'The question below is written in {lang_name}.\n'
    'Question: "{text}"\n\n'
    'Reply as JSON: {{"hindi": "<the question in natural Hindi, Devanagari script>", '
    '"english": "<the question in natural English>"}}'
)

LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}


def detect_language(text: str) -> str:
    """Return "hi" (Devanagari), "hinglish" (Roman-script Hindi) or "en"."""
    if DEVANAGARI.search(text):
        return "hi"

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return "en"

    hindi_like = sum(1 for w in words if w in HINGLISH_MARKERS)
    # two or more Hindi marker words, or at least a fifth of the words
    if hindi_like >= 2 or hindi_like / len(words) >= 0.2:
        return "hinglish"
    return "en"


def process_query(raw: str) -> dict:
    """Turn a raw question in any language into Hindi and English versions.

    Returns:
        {"lang": "en" | "hi" | "hinglish", "hindi": str, "english": str, "original": str}
    If the LLM call fails, the original text is used for both versions.
    """
    raw = raw.strip()
    lang = detect_language(raw)

    try:
        reply = ask_json(PROMPT.format(lang_name=LANG_NAMES[lang], text=raw), system=SYSTEM)
    except Exception as err:  # no key, no internet, API error
        print(f"[language] translation failed, using original text: {err}")
        reply = {}

    hindi = (reply.get("hindi") or "").strip() or raw
    english = (reply.get("english") or "").strip() or raw
    if lang == "en":
        english = raw  # never change an English question
    if lang == "hi":
        hindi = raw

    return {"lang": lang, "hindi": hindi, "english": english, "original": raw}


if __name__ == "__main__":
    for q in [
        "I failed my exam, how do I cope?",
        "exam mein fail ho gaya, kya karun",
        "मैं परीक्षा में फेल हो गया, क्या करूँ?",
    ]:
        print(process_query(q))
