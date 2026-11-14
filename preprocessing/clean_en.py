
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

EXTRA_STOPWORDS = {
    "thee", "thou", "thy", "thine", "ye", "hath", "unto"
}


def clean_en(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english")) | EXTRA_STOPWORDS

    tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    tokens = [
        stemmer.stem(token)
        for token in tokens
    ]

    return tokens
