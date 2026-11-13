
import json
from indexing.inverted_index import InvertedIndex


def sample_verses():
    return [
        {
            "id": "1.1",
            "translations": {
                "en_prabhupada": "The fruits of action are yours"
            },
            "commentary": "Fruits of action",
            "transliteration": "karma yoga"
        },
        {
            "id": "1.2",
            "translations": {
                "en_prabhupada": "Action without attachment"
            },
            "commentary": "The path of karma",
            "transliteration": "karman yoga"
        }
    ]


def test_build_index():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    assert "fruit" in index.index["translation"]
    assert "action" in index.index["translation"]


def test_document_length():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    assert index.doc_length("1.1") > 0


def test_document_frequency():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    assert index.doc_freq("action") == 2


def test_get_postings():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    postings = index.get_postings("action", zone="translation")

    assert "1.1" in postings
    assert "1.2" in postings


def test_commentary_zone():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    assert "fruit" in index.index["commentary"]
    assert "1.1" in index.index["commentary"]["fruit"]


def test_sanskrit_zone():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    assert "karma" in index.index["sanskrit"]
    assert "1.1" in index.index["sanskrit"]["karma"]
    assert "1.2" in index.index["sanskrit"]["karma"]


def test_phrase_search():
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    results = index.phrase_search("fruits of action")

    assert "1.1" in results


def test_save_and_load(tmp_path):
    index = InvertedIndex()
    index.build_index(sample_verses(), "en")

    path = tmp_path / "index.json"
    index.save(path)

    loaded_index = InvertedIndex()
    loaded_index.load(path)

    assert loaded_index.get_postings("action", zone="translation") == \
           index.get_postings("action", zone="translation")
