from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from app.scr.preprocessing_utils import preprocess_text

SEED = 42

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "sgd_classifier": "SGD Classifier",
    "lightgbm": "LightGBM",
}


def create_model(model_type: str) -> tuple[BaseEstimator, dict[str, list[Any]]]:
    """Create a classifier and its GridSearchCV parameter grid."""
    if model_type == "logistic_regression":
        classifier = LogisticRegression(random_state=SEED)
        parameters: dict[str, list[Any]] = {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__max_iter": [1000, 2000],
        }
    elif model_type == "sgd_classifier":
        classifier = SGDClassifier(random_state=SEED, tol=1e-3)
        parameters = {
            "classifier__loss": ["log_loss", "modified_huber"],
            "classifier__alpha": [1e-5, 1e-4, 1e-3],
            "classifier__penalty": ["l2", "elasticnet"],
            "classifier__max_iter": [1000, 2000],
        }
    elif model_type == "lightgbm":
        classifier = LGBMClassifier(
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
        )
        parameters = {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__num_leaves": [15, 31],
            "classifier__max_depth": [-1, 10],
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return classifier, parameters


def create_text_pipeline(model_type: str) -> tuple[Pipeline, dict[str, list[Any]]]:
    classifier, parameters = create_model(model_type)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_text,
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
            max_features=20000,
            ngram_range=(1, 2),
        )),
        ("classifier", classifier),
    ])
    return pipeline, parameters


def get_pipeline_classifier(pipeline: Pipeline) -> BaseEstimator:
    """Get the classifier from new pipelines or historical ``lr`` pipelines."""
    if "classifier" in pipeline.named_steps:
        return pipeline.named_steps["classifier"]
    if "lr" in pipeline.named_steps:
        return pipeline.named_steps["lr"]
    raise ValueError("Pipeline does not contain a supported classifier step")


def classifier_probabilities(
    classifier: BaseEstimator,
    features: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class labels and probabilities, deriving them from decision scores if needed."""
    classes = np.asarray(classifier.classes_)
    predict_proba = getattr(classifier, "predict_proba", None)
    if callable(predict_proba):
        try:
            return classes, np.asarray(predict_proba(features), dtype=float)
        except (AttributeError, NotImplementedError):
            pass

    decision_function = getattr(classifier, "decision_function", None)
    if not callable(decision_function):
        raise ValueError("Classifier supports neither predict_proba nor decision_function")

    scores = np.asarray(decision_function(features), dtype=float)
    if scores.ndim == 1:
        if len(classes) != 2:
            raise ValueError("One-dimensional decision scores require a binary classifier")
        positive_probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -709, 709)))
        probabilities = np.column_stack((1.0 - positive_probability, positive_probability))
    else:
        shifted_scores = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted_scores)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)

    return classes, probabilities
