
import math
from collections import Counter, defaultdict

from indexing.inverted_index import InvertedIndex
from schemas import make_result

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


def calculate_idf(total_documents, document_frequency):
    if total_documents <= 0 or document_frequency <= 0:
        return 0

    return math.log(
        1 + (total_documents - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def calculate_bm25_score(
    term_frequency,
    document_length,
    average_document_length,
    document_frequency,
    total_documents,
    k1=1.5,
    b=0.75
):
    if term_frequency <= 0 or total_documents <= 0:
        return 0

    if average_document_length <= 0:
        return 0

    idf = calculate_idf(total_documents, document_frequency)

    length_normalization = (
        1 - b + b * document_length / average_document_length
    )

    denominator = term_frequency + k1 * length_normalization

    return (
        idf
        * term_frequency
        * (k1 + 1)
        / denominator
    )


def search_bm25(
    query_tokens,
    k=20,
    lang="en",
    zone="translation",
    k1=1.5,
    b=0.75
):
    if k <= 0:
        return []

    if zone not in ("translation", "commentary", "sanskrit"):
        raise ValueError("Invalid search zone")

    if k1 < 0 or not 0 <= b <= 1:
        raise ValueError("Invalid BM25 parameters")

    index = load_index(lang)

    document_lengths = index.document_lengths[zone]
    total_documents = len(document_lengths)

    if total_documents == 0:
        return []

    average_document_length = (
        sum(document_lengths.values()) / total_documents
    )

    if average_document_length == 0:
        return []

    query_frequency = Counter(
        token for token in query_tokens if token
    )

    if not query_frequency:
        return []

    scores = defaultdict(float)

    for term, query_count in query_frequency.items():
        document_frequency = index.doc_freq(term, zone)

        if document_frequency == 0:
            continue

        postings = index.get_postings(term, zone)

        for verse_id, positions in postings.items():
            term_frequency = len(positions)
            document_length = document_lengths.get(verse_id, 0)

            term_score = calculate_bm25_score(
                term_frequency=term_frequency,
                document_length=document_length,
                average_document_length=average_document_length,
                document_frequency=document_frequency,
                total_documents=total_documents,
                k1=k1,
                b=b
            )

            scores[verse_id] += query_count * term_score

    results = [
        make_result(verse_id, score, "bm25")
        for verse_id, score in scores.items()
        if score > 0
    ]

    results.sort(
        key=lambda result: (
            -result["score"],
            tuple(map(int, result["id"].split(".")))
        )
    )

    return results[:k]
