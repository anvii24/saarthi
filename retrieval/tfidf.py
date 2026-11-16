
import math
from collections import Counter, defaultdict

from indexing.inverted_index import InvertedIndex
from schemas import make_result

_INDEX_CACHE = {}
_NORM_CACHE = {}


def load_index(lang="en"):
    if lang not in ("en", "hi"):
        raise ValueError("Language must be 'en' or 'hi'")

    if lang not in _INDEX_CACHE:
        index = InvertedIndex()
        path = f"data/processed/index_{lang}.json"
        index.load(path)
        _INDEX_CACHE[lang] = index

    return _INDEX_CACHE[lang]


def calculate_tf(term_frequency):
    if term_frequency <= 0:
        return 0
    return 1 + math.log(term_frequency)


def calculate_idf(total_documents, document_frequency):
    if document_frequency == 0:
        return 0
    return math.log(total_documents / document_frequency)


def calculate_document_norms(index, zone="translation"):
    norms_squared = defaultdict(float)
    total_documents = len(index.document_lengths[zone])

    if total_documents == 0:
        return {}

    for term, postings in index.index[zone].items():
        df = index.doc_freq(term, zone)
        idf = calculate_idf(total_documents, df)

        for verse_id, positions in postings.items():
            tf = calculate_tf(len(positions))
            weight = tf * idf
            norms_squared[verse_id] += weight ** 2

    return {
        verse_id: math.sqrt(value)
        for verse_id, value in norms_squared.items()
    }


def search_tfidf(query_tokens, k=20, lang="en", zone="translation"):
    if k <= 0:
        return []

    if zone not in ("translation", "commentary", "sanskrit"):
        raise ValueError("Invalid search zone")

    index = load_index(lang)
    total_documents = len(index.document_lengths[zone])

    if total_documents == 0:
        return []

    query_frequency = Counter(token for token in query_tokens if token)

    if not query_frequency:
        return []

    cache_key = (lang, zone)

    if cache_key not in _NORM_CACHE:
        _NORM_CACHE[cache_key] = calculate_document_norms(index, zone)

    document_norms = _NORM_CACHE[cache_key]

    query_weights = {}

    for term, frequency in query_frequency.items():
        df = index.doc_freq(term, zone)

        if df == 0:
            continue

        tf = calculate_tf(frequency)
        idf = calculate_idf(total_documents, df)
        query_weights[term] = tf * idf

    if not query_weights:
        return []

    query_norm = math.sqrt(
        sum(weight ** 2 for weight in query_weights.values())
    )

    if query_norm == 0:
        return []

    scores = defaultdict(float)

    for term, query_weight in query_weights.items():
        postings = index.get_postings(term, zone)
        idf = calculate_idf(
            total_documents,
            index.doc_freq(term, zone)
        )

        for verse_id, positions in postings.items():
            tf = calculate_tf(len(positions))
            document_weight = tf * idf
            scores[verse_id] += query_weight * document_weight

    results = []

    for verse_id, dot_product in scores.items():
        document_norm = document_norms.get(verse_id, 0)

        if document_norm == 0:
            continue

        similarity = dot_product / (query_norm * document_norm)

        results.append(
            make_result(verse_id, similarity, "tfidf")
        )

    results.sort(
        key=lambda result: (
            -result["score"],
            tuple(map(int, result["id"].split(".")))
        )
    )

    return results[:k]
