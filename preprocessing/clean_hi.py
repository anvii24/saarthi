
import re

STOPWORDS = {
    "है", "था", "थे", "थी", "हैं", "हो", "हूँ", "हे",
    "का", "की", "के", "को", "में", "से", "पर", "और",
    "एक", "यह", "वह", "इस", "उस", "जो", "तो", "भी",
    "नहीं", "अपने", "अपनी", "अपना", "मुझे", "तुम्हें",
    "हम", "वे", "मैं", "आप", "क्यों", "जब", "तक",
    "लिए", "द्वारा", "साथ", "बहुत", "कुछ"
}

NUKTA_MAP = str.maketrans({
    "क़": "क",
    "ख़": "ख",
    "ग़": "ग",
    "ज़": "ज",
    "ड़": "ड",
    "ढ़": "ढ",
    "फ़": "फ",
    "य़": "य"
})

SUFFIXES = ("ाओं", "ियों", "ुओं", "ों", "ें")


def clean_hi(text):
    text = text.translate(NUKTA_MAP)
    text = re.sub(r"[।॥।,!?;:\"'()\[\]{}]", " ", text)

    tokens = text.split()

    cleaned_tokens = []

    for token in tokens:
        token = token.strip()

        if not token or token in STOPWORDS:
            continue

        for suffix in SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break

        cleaned_tokens.append(token)

    return cleaned_tokens
