"""Fact-checker: an NLI model checks each cited sentence against its verse; a sentence is
supported if P(entailment) >= 0.5, and unsupported ones are dropped.

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7. The verse text starts
with its speaker ("Arjuna says: ...") because answers name the speaker.
"""

import re
import threading

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
THRESHOLD = 0.5  # tune on a few hand-checked answers; report the value used

CITATION = re.compile(r"\[(\d{1,2}\.\d{1,3})\]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

_tokenizer = None
_model = None
_lock = threading.Lock()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? and the Hindi full stop ।"""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _load_model():
    global _tokenizer, _model
    with _lock:
        if _model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _model.eval()
    return _tokenizer, _model


def entailment_prob(premise: str, hypothesis: str) -> float:
    """Probability that the premise supports (entails) the hypothesis."""
    import torch

    tokenizer, model = _load_model()
    inputs = tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits[0], dim=-1)
    labels = {name.lower(): i for i, name in model.config.id2label.items()}
    return float(probs[labels["entailment"]])


SPEAKERS = {"श्रीभगवान्": "Krishna", "अर्जुन": "Arjuna", "सञ्जय": "Sanjaya", "धृतराष्ट्र": "Dhritarashtra"}


def premise_for(verse: dict, sentence: str) -> str:
    """The verse text to check against, in the same script as the sentence (two translations),
    starting with who speaks it, e.g. "Arjuna says: My mind is confused about duty ..."."""
    hindi = bool(DEVANAGARI.search(sentence))
    prefix = "hi_" if hindi else "en_"
    texts = [t for k, t in verse.get("translations", {}).items() if k.startswith(prefix) and t]
    if not texts:  # fall back to any translation
        texts = [t for t in verse.get("translations", {}).values() if t]
    text = " ".join(texts[:2])
    speaker = verse.get("speaker", "")
    if speaker in SPEAKERS:
        text = (f"{speaker} ने कहा: " if hindi else f"{SPEAKERS[speaker]} says: ") + text
    return text


def verify(explanation: str, verses: list[dict], threshold: float = THRESHOLD) -> list[dict]:
    """Return one entry per sentence:
        {"sentence", "verse_ids", "score", "supported"}
    supported is True / False for sentences that cite a verse, None for the rest.
    """
    by_id = {v["id"]: v for v in verses}
    checks = []
    for sentence in split_sentences(explanation):
        cited = [vid for vid in CITATION.findall(sentence)]
        claim = CITATION.sub("", sentence).strip()

        if not cited:
            checks.append({"sentence": sentence, "verse_ids": [], "score": None, "supported": None})
            continue

        known = [by_id[vid] for vid in cited if vid in by_id]
        if not known:  # cites a verse that was not retrieved: cannot be supported
            checks.append({"sentence": sentence, "verse_ids": cited, "score": 0.0, "supported": False})
            continue

        score = max(entailment_prob(premise_for(v, claim), claim) for v in known)
        checks.append({
            "sentence": sentence,
            "verse_ids": cited,
            "score": round(score, 3),
            "supported": score >= threshold,
        })
    return checks


def keep_supported(checks: list[dict]) -> str:
    """Rebuild the explanation without the unsupported sentences."""
    return " ".join(c["sentence"] for c in checks if c["supported"] is not False)


if __name__ == "__main__":
    verse = {"id": "2.47", "translations": {
        "en_gambirananda": "Your right is for action alone, never for the results.",
    }}
    text = ("Here is what the Gita says. Focus on doing your work, not on the outcome [2.47]. "
            "The Gita promises that hard work always brings success [2.47].")
    for c in verify(text, [verse]):
        print(c)