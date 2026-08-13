from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1)
