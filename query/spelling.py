
import re

from indexing.dictionary import (
    build_kgram_index,
    edit_distance,
    soundex,
)
from retrieval.spell_correction import load_index


_VOCAB_CACHE = {}


def _get_vocabulary(lang, zone):
    """Load vocabulary, k-gram index and term frequencies."""
    key = (lang, zone)

    if key not in _VOCAB_CACHE:
        index = load_index(lang)
        postings = index.index[zone]

        vocabulary = set(postings.keys())
        kgram_index = build_kgram_index(vocabulary, k=2)

        frequencies = {
            term: sum(len(positions) for positions in docs.values())
            for term, docs in postings.items()
        }

        _VOCAB_CACHE[key] = (
            vocabulary,
            kgram_index,
            frequencies,
        )

    return _VOCAB_CACHE[key]


def _get_kgrams(word, k=2):
    """Get character k-grams with boundary markers."""
    padded = f"${word}$"

    return {
        padded[i:i + k]
        for i in range(len(padded) - k + 1)
    }


def _correct_word(word, lang, zone):
    """Correct one word using Jaccard, edit distance and Soundex."""
    if not word:
        return word

    index = load_index(lang)

    if zone not in ("translation", "commentary", "sanskrit"):
        raise ValueError("Invalid search zone")

    if lang == "en":
        from preprocessing.clean_en import clean_en
        tokens = clean_en(word)
    elif lang == "hi":
        from preprocessing.clean_hi import clean_hi
        tokens = clean_hi(word)
    else:
        raise ValueError("Language must be 'en' or 'hi'")

    if len(tokens) != 1:
        return word

    normalized = tokens[0]

    vocabulary, kgram_index, frequencies = _get_vocabulary(
        lang, zone
    )

    if normalized in vocabulary:
        return word

    query_grams = _get_kgrams(normalized)

    # Generate candidates using the k-gram index.
    candidates = set()

    for gram in query_grams:
        candidates.update(kgram_index.get(gram, set()))

        ranked = []
    query_soundex = soundex(normalized)

    # Reject candidates that are too different.
    max_distance = 1 if len(normalized) <= 4 else 2

    for candidate in candidates:
        candidate_grams = _get_kgrams(candidate)

        union = query_grams | candidate_grams
        intersection = query_grams & candidate_grams

        jaccard = len(intersection) / len(union)

        if jaccard < 0.3:
            continue

        distance = edit_distance(normalized, candidate)

        if distance > max_distance:
            continue

        # Allow two-edit corrections only when phonetic codes match.
        if distance == 2 and soundex(candidate) != query_soundex:
            continue

        same_soundex = soundex(candidate) == query_soundex

        ranked.append((
            distance,
            0 if same_soundex else 1,
            -frequencies.get(candidate, 0),
            candidate,
        ))

    if not ranked:
        return word

    ranked.sort()
    return ranked[0][3]


def correct(query, lang="en", zone="translation"):
    """Correct misspelled words in a complete search query."""
    if not query or not query.strip():
        return query

    def replace_word(match):
        token = match.group(0)

        # Preserve punctuation around each word.
        parts = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", token)

        if not parts:
            return token

        prefix, word, suffix = parts.groups()
        corrected = _correct_word(word, lang, zone)

        return prefix + corrected + suffix

    return re.sub(r"\S+", replace_word, query)


__all__ = ["correct"]
