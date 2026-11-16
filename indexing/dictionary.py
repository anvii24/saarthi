import re
from collections import defaultdict


def build_kgram_index(vocab, k=2):
    """Build a k-gram index with start and end boundaries."""
    if k < 1:
        raise ValueError("k must be at least 1")

    index = defaultdict(set)

    for word in vocab:
        padded_word = f"${word}$"

        for i in range(len(padded_word) - k + 1):
            gram = padded_word[i:i + k]
            index[gram].add(word)

    return dict(index)


def edit_distance(a, b):
    """Calculate Levenshtein edit distance."""
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(min(insertion, deletion, substitution))

        previous = current

    return previous[-1]


def soundex(term):
    """Generate a Soundex code for phonetic matching."""
    if not term:
        return ""

    term = term.lower()
    codes = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2",
        "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6"
    }

    first = term[0].upper()
    digits = []
    previous = codes.get(term[0], "")

    for char in term[1:]:
        code = codes.get(char, "")

        if code and code != previous:
            digits.append(code)

        previous = code

    return (first + "".join(digits) + "000")[:4]


def wildcard(pattern, kgram_index):
    """Find vocabulary terms matching a wildcard pattern."""
    vocab = set()

    for terms in kgram_index.values():
        vocab.update(terms)

    literal_parts = re.findall(r"[a-zA-Z0-9]+", pattern)

    candidates = vocab

    if literal_parts:
        grams = set()

        for part in literal_parts:
            padded = f"${part}$"

            for i in range(len(padded) - 1):
                grams.add(padded[i:i + 2])

        matching_sets = [
            kgram_index[gram]
            for gram in grams
            if gram in kgram_index
        ]

        if matching_sets:
            candidates = set.intersection(
                *(set(items) for items in matching_sets)
            )

    regex = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"

    return sorted(
        word for word in candidates
        if re.fullmatch(regex, word)
    )