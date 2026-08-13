import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = {
            "a", "an", "the", "and", "or", "but", "if", "then", "so", "this", "that",
            "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
            "for", "with", "without", "it", "as", "at", "by", "from", "my", "our", "your",
            "i", "we", "they", "he", "she", "me", "us", "them", "their", "his", "her"
        }


def tokenize(text: str):
    if not isinstance(text, str):
        return []

    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    try:
        tokens = word_tokenize(clean_text)
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            tokens = word_tokenize(clean_text)
        except Exception:
            tokens = clean_text.split()

    return [token for token in tokens if token.isalnum() and token not in STOPWORDS]


def preprocess_text(text: str) -> str:
    tokens = tokenize(text)
    return " ".join(tokens)
