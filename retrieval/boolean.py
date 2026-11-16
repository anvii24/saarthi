
import re

from indexing.inverted_index import InvertedIndex
from preprocessing.clean_en import clean_en
from preprocessing.clean_hi import clean_hi
from preprocessing.normalise import normalise_sanskrit

_INDEX_CACHE = {}


def load_index(lang="en"):
    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")

    if lang not in _INDEX_CACHE:
        index = InvertedIndex()
        path = f"data/processed/index_{lang}.json"
        index.load(path)
        _INDEX_CACHE[lang] = index

    return _INDEX_CACHE[lang]


def search_boolean(query, lang="en", zone="translation"):
    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")

    if zone not in ("translation", "commentary", "sanskrit"):
        raise ValueError("Invalid search zone")

    if not query or not query.strip():
        return []

    index = load_index(lang)
    all_documents = set(index.document_lengths[zone])

    # Split the query into words, operators and parentheses.
    tokens = re.findall(r"\(|\)|[^\s()]+", query)

    tokens = [
        token.upper() if token.upper() in ("AND", "OR", "NOT")
        else token
        for token in tokens
    ]

    position = 0

    def get_term_documents(term):
        if zone == "sanskrit":
            normalized = normalise_sanskrit(term.lower())
        elif lang == "en":
            cleaned = clean_en(term)
            if len(cleaned) != 1:
                raise ValueError(
                    f"Search term '{term}' must produce exactly one token"
                )
            normalized = cleaned[0]
        else:
            cleaned = clean_hi(term)
            if len(cleaned) != 1:
                raise ValueError(
                    f"Search term '{term}' must produce exactly one token"
                )
            normalized = cleaned[0]

        return set(index.get_postings(normalized, zone))

    # Parse individual terms and parenthesized expressions.
    def parse_primary():
        nonlocal position

        if position >= len(tokens):
            raise ValueError("Unexpected end of Boolean query")

        token = tokens[position]

        if token == "(":
            position += 1
            result = parse_or()

            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("Missing closing parenthesis")

            position += 1
            return result

        if token in ("AND", "OR", "NOT", ")"):
            raise ValueError(f"Unexpected operator: {token}")

        position += 1
        return get_term_documents(token)

    # NOT has the highest precedence.
    def parse_not():
        nonlocal position

        if position < len(tokens) and tokens[position] == "NOT":
            position += 1
            return all_documents - parse_not()

        return parse_primary()

    # AND has the second-highest precedence.
    def parse_and():
        nonlocal position

        result = parse_not()

        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            result = result & parse_not()

        return result

    # OR has the lowest precedence.
    def parse_or():
        nonlocal position

        result = parse_and()

        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            result = result | parse_and()

        return result

    result = parse_or()

    if position != len(tokens):
        raise ValueError(f"Unexpected token: {tokens[position]}")

    return sorted(
        result,
        key=lambda verse_id: tuple(map(int, verse_id.split(".")))
    )
