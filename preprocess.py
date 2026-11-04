"""Text preprocessing — identical to the notebook's transform_text()."""
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def _ensure_nltk():
    for path, pkg in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
_ps = PorterStemmer()
_stop = set(stopwords.words("english"))  # set => much faster than the notebook's list lookup


def transform_text(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in _stop and t not in string.punctuation]
    return " ".join(_ps.stem(t) for t in tokens)
