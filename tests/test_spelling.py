import pytest

from query.spelling import (
    edit_distance,
    get_max_distance,
    correct_word,
    correct_query,
)

def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3


def test_hindi_virama_distance():
    assert edit_distance("धरम", "धर्म", "hi") == 0.25


def test_max_distance():
    assert get_max_distance("cat") == 1
    assert get_max_distance("dharma") == 2


def test_english_correction():
    assert correct_word("dharmma", "en", "commentary") == "dharma"


def test_english_valid_word():
    assert correct_word("dharma", "en", "commentary") == "dharma"


def test_hindi_correction():
    assert correct_word("धरम", "hi", "translation") == "धर्म"


def test_hindi_valid_word():
    assert correct_word("धर्म", "hi", "translation") == "धर्म"


def test_correct_query():
    assert correct_query(
        "What is dharmma", "en", "commentary"
    ) == "What is dharma"


def test_hindi_query():
    assert correct_query(
        "धरम क्या है", "hi", "translation"
    ) == "धर्म क्या है"


def test_empty_query():
    assert correct_query("", "en") == ""


def test_invalid_language():
    with pytest.raises(ValueError):
        correct_word("dharma", "fr")


def test_invalid_zone():
    with pytest.raises(ValueError):
        correct_word("dharma", "en", "invalid")