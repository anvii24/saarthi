
from preprocessing.clean_en import clean_en


def test_clean_en_lowercase_and_stem():
    assert clean_en("RUNNING RUNS") == ["run", "run"]


def test_clean_en_removes_stopwords():
    assert clean_en("The fruits of action are not yours") == ["fruit", "action"]


def test_clean_en_removes_punctuation():
    assert clean_en("Hello, world!") == ["hello", "world"]


def test_clean_en_empty_string():
    assert clean_en("") == []
