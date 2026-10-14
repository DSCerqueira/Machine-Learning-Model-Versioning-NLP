from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline

from app.modeling import classifier_probabilities, get_pipeline_classifier


def _as_texts(texts: Any) -> list[str]:
    if isinstance(texts, pd.Series):
        values = texts.tolist()
    elif isinstance(texts, np.ndarray):
        values = texts.reshape(-1).tolist()
    elif isinstance(texts, (list, tuple)):
        values = list(texts)
    else:
        raise TypeError("texts must be a pandas Series, list, tuple, or NumPy array")
    return ["" if pd.isna(value) else str(value) for value in values]


def extract_nlp_monitoring_features(
    texts: pd.Series | list[Any] | np.ndarray,
    model: Pipeline,
    positive_label: Any = 1,
    selected_terms: Iterable[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Extract interpretable NLP features from a fitted TF-IDF classifier pipeline."""
    if not isinstance(model, Pipeline) or "tfidf" not in model.named_steps:
        raise ValueError("model must be a fitted Pipeline with a 'tfidf' step")
    vectorizer = model.named_steps["tfidf"]
    classifier = get_pipeline_classifier(model)
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(classifier, "classes_"):
        raise ValueError("model Pipeline must be fitted before monitoring")

    text_values = _as_texts(texts)
    word_counts = np.fromiter((len(value.split()) for value in text_values), dtype=np.int64)
    character_counts = np.fromiter((len(value) for value in text_values), dtype=np.int64)
    average_word_lengths = np.fromiter(
        (
            sum(map(len, value.split())) / len(value.split()) if value.split() else 0.0
            for value in text_values
        ),
        dtype=np.float64,
    )

    tfidf_matrix = vectorizer.transform(text_values)
    tfidf_nonzero_counts = np.asarray(tfidf_matrix.getnnz(axis=1)).reshape(-1)
    predicted = classifier.predict(tfidf_matrix)
    classes, probabilities = classifier_probabilities(classifier, tfidf_matrix)
    positive_matches = np.flatnonzero(classes == positive_label)
    if len(positive_matches) != 1:
        raise ValueError(f"positive_label {positive_label!r} is not a model class")
    positive_probabilities = probabilities[:, int(positive_matches[0])]

    features = pd.DataFrame({
        "word_count": word_counts,
        "char_count": character_counts,
        "avg_word_length": average_word_lengths,
        "tfidf_nonzero_count": tfidf_nonzero_counts,
        "positive_probability": positive_probabilities,
        "predicted_sentiment": [str(value) for value in predicted],
    })
    terms = list(selected_terms or [])
    if terms:
        indices = np.asarray([int(term["index"]) for term in terms], dtype=int)
        if np.any(indices < 0) or np.any(indices >= tfidf_matrix.shape[1]):
            raise ValueError("selected TF-IDF term index is outside the fitted vocabulary")
        selected_matrix = sparse.csr_matrix(tfidf_matrix[:, indices])
        selected_matrix.data = np.ones_like(selected_matrix.data)
        for index, term in enumerate(terms):
            features[f"tfidf_term:{term['term']}"] = selected_matrix[:, index].toarray().ravel()
    return features


def important_tfidf_indices(model: Pipeline, top_n: int) -> np.ndarray:
    """Return the most important TF-IDF terms for a fitted classifier."""
    if top_n < 0:
        raise ValueError("top_n must be nonnegative")
    if not isinstance(model, Pipeline) or "tfidf" not in model.named_steps:
        raise ValueError("model must be a Pipeline containing a 'tfidf' step")
    classifier = get_pipeline_classifier(model)
    if hasattr(classifier, "coef_"):
        importance = np.max(np.abs(np.asarray(classifier.coef_)), axis=0)
    elif hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_)
    else:
        raise ValueError("model classifier must be fitted before selecting terms")
    count = min(top_n, importance.size)
    return np.argsort(importance)[-count:][::-1] if count else np.array([], dtype=int)


def extract_selected_tfidf_presence(
    texts: pd.Series | list[Any] | np.ndarray,
    model: Pipeline,
    selected_terms: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """Extract only selected vocabulary columns, preserving sparse transforms."""
    terms = list(selected_terms)
    if not terms:
        return pd.DataFrame(index=range(len(_as_texts(texts))))
    if not isinstance(model, Pipeline) or "tfidf" not in model.named_steps:
        raise ValueError("model must be a fitted Pipeline with a 'tfidf' step")
    get_pipeline_classifier(model)
    vectorizer = model.named_steps["tfidf"]
    text_values = _as_texts(texts)
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("model TF-IDF vectorizer must be fitted before monitoring")

    indices = np.asarray([int(term["index"]) for term in terms], dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(vectorizer.get_feature_names_out())):
        raise ValueError("selected TF-IDF term index is outside the fitted vocabulary")
    matrix = vectorizer.transform(text_values)[:, indices]
    presence = sparse.csr_matrix(matrix)
    presence.data = np.ones_like(presence.data)
    values = presence.toarray()
    names = [f"tfidf_term:{term['term']}" for term in terms]
    return pd.DataFrame(values, columns=names)
