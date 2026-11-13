
from preprocessing.normalise import normalise_sanskrit


def test_karman_normalisation():
    assert normalise_sanskrit("karman") == "karma"


def test_krsna_normalisation():
    assert normalise_sanskrit("krsna") == "krishna"


def test_krushna_normalisation():
    assert normalise_sanskrit("krushna") == "krishna"


def test_dharm_normalisation():
    assert normalise_sanskrit("dharm") == "dharma"


def test_case_insensitive():
    assert normalise_sanskrit("KRSNA") == "krishna"


def test_unknown_token():
    assert normalise_sanskrit("unknownword") == "unknownword"


def test_whitespace():
    assert normalise_sanskrit("  karman  ") == "karma"
