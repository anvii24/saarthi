from functools import lru_cache

from indexing.inverted_index import InvertedIndex
from preprocessing.clean_en import clean_en
from preprocessing.clean_hi import clean_hi


_INDEX_CACHE = {}


def load_index(lang="en"):
    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")

    if lang not in _INDEX_CACHE:
        index = InvertedIndex()
        index.load(f"data/processed/index_{lang}.json")
        _INDEX_CACHE[lang] = index

    return _INDEX_CACHE[lang]


def edit_distance(word1, word2, lang="en"):
    """Calculate language-aware edit distance."""

    previous = list(range(len(word2) + 1))

    for i, char1 in enumerate(word1, start=1):
        current = [i]

        for j, char2 in enumerate(word2, start=1):
            insertion_cost = 1
            deletion_cost = 1
            substitution_cost = 0 if char1 == char2 else 1

            # Virama insertion/deletion is a common Hindi variation.
            if lang == "hi":
                if char1 == "्":
                    deletion_cost = 0.25

                if char2 == "्":
                    insertion_cost = 0.25

            insertion = current[j - 1] + insertion_cost
            deletion = previous[j] + deletion_cost
            substitution = previous[j - 1] + substitution_cost

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def get_max_distance(word):
    """Set correction threshold based on word length."""

    if len(word) <= 4:
        return 1

    return 2


@lru_cache(maxsize=10000)
def _find_candidates(word, lang, zone):
    index = load_index(lang)
    vocabulary = index.index[zone]
    frequencies = index.document_frequencies[zone]

    max_distance = get_max_distance(word)
    candidates = []

    for term in vocabulary:
        if abs(len(term) - len(word)) > max_distance:
            continue

        distance = edit_distance(word, term, lang)

        if distance <= max_distance:
            candidates.append(
                (distance, -frequencies.get(term, 0), term)
            )

    candidates.sort()

    return tuple(candidates)


def correct_word(word, lang="en", zone="translation"):
    """Correct a single query word using the indexed vocabulary."""

    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")

    if zone not in ("translation", "commentary", "sanskrit"):
        raise ValueError("Invalid search zone")

    if not word:
        return word

    if lang == "en":
        tokens = clean_en(word)
    else:
        tokens = clean_hi(word)

    if len(tokens) != 1:
        return word

    normalized = tokens[0]
    index = load_index(lang)

    if normalized in index.index[zone]:
        return word

    candidates = _find_candidates(normalized, lang, zone)

    if not candidates:
        return word

    best_distance, _, best_term = candidates[0]

    if best_distance > get_max_distance(normalized):
        return word

    return best_term


def correct_query(query, lang="en", zone="translation"):
    """Correct individual words in a search query."""

    if not query or not query.strip():
        return query

    words = query.split()

    corrected_words = [
        correct_word(word, lang, zone)
        for word in words
    ]

    return " ".join(corrected_words)