from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from app import main
from app.modeling import (
    MODEL_DISPLAY_NAMES,
    classifier_probabilities,
    create_text_pipeline,
    get_pipeline_classifier,
)
from app.monitoring.nlp_features import (
    extract_nlp_monitoring_features,
    important_tfidf_indices,
)

TEXTS = [
    "a wonderful enjoyable movie",
    "excellent and delightful film",
    "the story was warm and inspiring",
    "great acting and a satisfying ending",
    "a charming and memorable comedy",
    "beautifully made and deeply moving",
    "funny, clever, and full of heart",
    "an entertaining and uplifting story",
    "a good film with strong performances",
    "I loved this excellent movie",
    "a terrible and boring movie",
    "awful acting and a painful script",
    "the story was dull and uninspired",
    "a disappointing film with a weak ending",
    "a tedious and forgettable comedy",
    "poorly made and emotionally empty",
    "unfunny, confusing, and lacking heart",
    "a miserable and unpleasant story",
    "a bad film with weak performances",
    "I hated this awful movie",
]
LABELS = np.array([1] * 10 + [0] * 10)


@pytest.mark.parametrize("model_type", list(MODEL_DISPLAY_NAMES))
def test_classifier_tunes_and_trains_with_tfidf(model_type):
    pipeline, parameter_grid = create_text_pipeline(model_type)
    reduced_grid = {name: [values[0]] for name, values in parameter_grid.items()}
    X_train, X_test, y_train, y_test = train_test_split(
        TEXTS,
        LABELS,
        test_size=0.25,
        random_state=42,
        stratify=LABELS,
    )
    search = GridSearchCV(
        pipeline,
        reduced_grid,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
        error_score="raise",
    ).fit(X_train, y_train)
    model = search.best_estimator_

    predictions = model.predict(X_test)
    classifier = get_pipeline_classifier(model)
    classes, probabilities = classifier_probabilities(
        classifier, model.named_steps["tfidf"].transform(X_test)
    )

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    assert np.isfinite(search.best_score_)
    assert predictions.shape == y_test.shape
    assert 0 <= accuracy <= 1
    assert probabilities.shape == (len(X_test), len(classes))
    assert 0 <= auc <= 1
    assert important_tfidf_indices(model, top_n=3).size > 0
    assert len(extract_nlp_monitoring_features(X_test, model)) == len(X_test)


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_text_pipeline("not-a-model")


def test_dashboard_offers_all_model_identifiers():
    dashboard = main.home()
    for model_type in MODEL_DISPLAY_NAMES:
        assert f'value="{model_type}"' in dashboard
    assert "JSON.stringify({model_type})" in dashboard


def test_training_endpoint_rejects_unknown_model_identifier(monkeypatch):
    monkeypatch.setattr(
        main,
        "read_state",
        lambda: {"active_dataset": {"name": "sst2", "version": 1}},
    )
    with pytest.raises(HTTPException) as error:
        main.train_model({"model_type": "not-a-model"})
    assert error.value.status_code == 400
