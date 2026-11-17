"""
Spelling correction (Person 2: Kashvi).

correct("What is dharmma") -> "What is dharma"

How one word is corrected:
  1. Clean it the same way the index does (clean_en / clean_hi). If the cleaned
     word is already in the index vocabulary, leave it alone.
  2. Find candidates with the bigram (k=2) index: vocabulary words sharing
     character bigrams with it, keeping those with Jaccard overlap >= 0.3.
  3. Keep candidates within edit distance 1 (words of 4 letters or fewer) or 2.
     A distance-2 candidate is only allowed if its Soundex code matches.
  4. Rank by edit distance, then same Soundex, then how often the word occurs.

Edit distance is language-aware: in Hindi, adding or dropping the virama (्),
as in धरम / धर्म, costs only 0.25, because it is a very common spelling variation.

Note: the corrected word is the index's cleaned (stemmed) form, so use this for
keyword search, not for text shown to the user.
"""

import re

from indexing.dictionary import build_kgram_index, soundex
from indexing.inverted_index import InvertedIndex
from preprocessing.clean_en import clean_en
from preprocessing.clean_hi import clean_hi

VIRAMA = "्"
ZONES = ("translation", "commentary", "sanskrit")

_INDEX_CACHE = {}
_VOCAB_CACHE = {}


def load_index(lang="en"):
    """Load the inverted index built by build_indexes.py (cached)."""
    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")

    if lang not in _INDEX_CACHE:
        index = InvertedIndex()
        index.load(f"data/processed/index_{lang}.json")
        _INDEX_CACHE[lang] = index

    return _INDEX_CACHE[lang]


def edit_distance(word1, word2, lang="en"):
    """Levenshtein distance; in Hindi, inserting or deleting a virama costs 0.25."""
    previous = list(range(len(word2) + 1))

    for i, char1 in enumerate(word1, start=1):
        current = [i]

        for j, char2 in enumerate(word2, start=1):
            insertion_cost = 0.25 if lang == "hi" and char2 == VIRAMA else 1
            deletion_cost = 0.25 if lang == "hi" and char1 == VIRAMA else 1
            substitution_cost = 0 if char1 == char2 else 1
            
            current.append(min(
                current[j - 1] + insertion_cost,
                previous[j] + deletion_cost,
                previous[j - 1] + substitution_cost,
            ))

        previous = current

    return previous[-1]


def get_max_distance(word):
    """Short words may differ by 1 edit, longer words by 2."""
    return 1 if len(word) <= 4 else 2


def _get_kgrams(word, k=2):
    """Character k-grams with $ marking the start and end of the word."""
    padded = f"${word}$"
    return {padded[i:i + k] for i in range(len(padded) - k + 1)}


def _get_vocabulary(lang, zone):
    """Vocabulary, k-gram index and term frequencies for one zone (cached)."""
    key = (lang, zone)

    if key not in _VOCAB_CACHE:
        postings = load_index(lang).index[zone]
        vocabulary = set(postings.keys())
        frequencies = {
            term: sum(len(positions) for positions in docs.values())
            for term, docs in postings.items()
        }
        _VOCAB_CACHE[key] = (vocabulary, build_kgram_index(vocabulary, k=2), frequencies)

    return _VOCAB_CACHE[key]


def correct_word(word, lang="en", zone="translation"):
    """Correct one word using the indexed vocabulary (see the top of this file)."""
    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")
    if zone not in ZONES:
        raise ValueError("Invalid search zone")
    if not word:
        return word

    tokens = clean_en(word) if lang == "en" else clean_hi(word)
    if len(tokens) != 1:  # a stopword, or not a single word
        return word

    normalized = tokens[0]
    vocabulary, kgram_index, frequencies = _get_vocabulary(lang, zone)
    if normalized in vocabulary:
        return word

    # Candidates: vocabulary words that share at least one bigram.
    query_grams = _get_kgrams(normalized)
    candidates = set()
    for gram in query_grams:
        candidates.update(kgram_index.get(gram, set()))

    query_soundex = soundex(normalized)
    max_distance = get_max_distance(normalized)
    ranked = []

    for candidate in candidates:
        candidate_grams = _get_kgrams(candidate)
        jaccard = len(query_grams & candidate_grams) / len(query_grams | candidate_grams)
        if jaccard < 0.3:
            continue

        distance = edit_distance(normalized, candidate, lang)
        if distance > max_distance:
            continue

        same_soundex = soundex(candidate) == query_soundex
        if distance >= 2 and not same_soundex:
            continue

        ranked.append((distance, 0 if same_soundex else 1, -frequencies.get(candidate, 0), candidate))

    if not ranked:
        return word

    ranked.sort()
    return ranked[0][3]


def correct(query, lang="en", zone="translation"):
    """Correct every word in a query, keeping punctuation around each word."""
    if not query or not query.strip():
        return query

    def replace_word(match):
        parts = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", match.group(0))
        if not parts:
            return match.group(0)
        prefix, word, suffix = parts.groups()
        return prefix + correct_word(word, lang, zone) + suffix

    return re.sub(r"\S+", replace_word, query)


correct_query = correct  # older name, kept so existing code keeps working

__all__ = ["correct", "correct_query", "correct_word", "edit_distance", "get_max_distance", "load_index"]