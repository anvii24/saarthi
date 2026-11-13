
from preprocessing.clean_hi import clean_hi


def test_clean_hi_removes_stopwords():
    assert clean_hi("कर्म का फल") == ["कर्म", "फल"]


def test_clean_hi_removes_punctuation():
    assert clean_hi("कर्मों का फल नहीं मिलता।") == ["कर्म", "फल", "मिलता"]


def test_clean_hi_normalizes_nukta():
    assert clean_hi("क़िताब ज़रूरी") == ["किताब", "जरूरी"]


def test_clean_hi_normalizes_suffixes():
    assert clean_hi("कर्मों") == ["कर्म"]


def test_clean_hi_empty_string():
    assert clean_hi("") == []
