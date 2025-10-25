import random
from typing import Literal

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def setup_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def pos_tag_text(text: str) -> str:
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([tag for _, tag in pos_tags])


def preprocessing(
    train_pos_tags: list[str] | pd.Series,
    test_pos_tags: list[str] | pd.Series,
    ngram_range: tuple[int, int] = (1, 3),
    min_doc_freq: int = 1,
    vectorizer_name: Literal["tfidf", "count"] = "tfidf",
):
    """Vectorizes the POS tags and normalizes data"""
    match vectorizer_name:
        case "tfidf":
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                lowercase=False,
                min_df=min_doc_freq,
                token_pattern=r"\S+",
                stop_words=None,
            )
        case "count":
            vectorizer = CountVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                lowercase=False,
                min_df=min_doc_freq,
                token_pattern=r"\S+",
                stop_words=None,
            )
    X_train = vectorizer.fit_transform(train_pos_tags)
    X_test = vectorizer.transform(test_pos_tags)

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, vectorizer
