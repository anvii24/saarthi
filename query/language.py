"""
Language handling for user questions (Person 3: Anvi).

process_query() works out whether a question is English, Hindi or Hinglish
and returns a Hindi and an English version, so both indexes can be searched.

STUB: detection works with simple rules; transliteration and translation
return placeholders until IndicXlit / IndicTrans2 are wired in.
"""

import re

# common Hindi words typed in Roman letters, used to spot Hinglish
HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kaise", "kyun", "kyon", "mera", "meri", "mere",
    "nahi", "nahin", "main", "mein", "ho", "gaya", "gayi", "karu", "karun",
    "karoon", "kar", "raha", "rahi", "tha", "thi", "aur", "bhi", "ko", "se",
    "ki", "ka", "ke", "par", "lekin", "kuch", "bahut", "accha", "acha",
}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


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


def transliterate_to_hindi(text: str) -> str:
    """Hinglish (Roman letters) -> Devanagari Hindi.

    TODO: replace with AI4Bharat IndicXlit.
    """
    return text  # stub: returns the input unchanged


def translate_to_english(text: str) -> str:
    """Hindi -> English.

    TODO: replace with AI4Bharat IndicTrans2.
    """
    return text  # stub: returns the input unchanged


def translate_to_hindi(text: str) -> str:
    """English -> Hindi.

    TODO: replace with AI4Bharat IndicTrans2.
    """
    return text  # stub: returns the input unchanged


def process_query(raw: str) -> dict:
    """Turn a raw question in any language into Hindi and English versions.

    Returns:
        {"lang": "en" | "hi" | "hinglish", "hindi": str, "english": str, "original": str}
    """
    raw = raw.strip()
    lang = detect_language(raw)

    if lang == "en":
        english = raw
        hindi = translate_to_hindi(raw)
    elif lang == "hi":
        hindi = raw
        english = translate_to_english(raw)
    else:  # hinglish
        hindi = transliterate_to_hindi(raw)
        english = translate_to_english(hindi)

    return {"lang": lang, "hindi": hindi, "english": english, "original": raw}


if __name__ == "__main__":
    for q in [
        "I failed my exam, how do I cope?",
        "exam mein fail ho gaya, kya karun",
        "मैं परीक्षा में फेल हो गया",
    ]:
        print(process_query(q))
