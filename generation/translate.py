"""
Answer translation (Person 3: Anvi).

translate_answer() shows the final explanation in English, Hindi or Hinglish.
Hinglish = translate to Hindi, then write it in Roman letters.

STUB: returns the text unchanged until IndicTrans2 / IndicXlit are wired in.
"""


def translate_answer(text: str, target_lang: str) -> str:
    """Translate an explanation into "en", "hi" or "hinglish".

    TODO: en <-> hi with IndicTrans2; hi -> hinglish with IndicXlit romanisation.
    """
    if target_lang not in {"en", "hi", "hinglish"}:
        raise ValueError(f"unknown language: {target_lang}")
    return text  # stub


if __name__ == "__main__":
    print(translate_answer("Focus on your actions, not the results.", "hi"))
