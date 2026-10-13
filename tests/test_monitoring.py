from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.monitoring.drift_evaluator import evaluate_drift
from app.monitoring.drift_metrics import (
    calculate_feature_js,
    calculate_feature_ks,
    calculate_feature_psi,
)
from app.monitoring.nlp_features import (
    extract_nlp_monitoring_features,
    extract_selected_tfidf_presence,
)
from app.monitoring.psi_baseline import (
    build_psi_baseline,
    load_psi_baseline,
    save_psi_baseline,
)


@pytest.fixture
def fitted_model() -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(random_state=3)),
    ])
    model.fit(
        ["good movie", "great film", "bad movie", "awful film"],
        [2, 2, 1, 1],
    )
    return model


def _baseline(values: list[float], feature: str = "score", **kwargs):
    return build_psi_baseline(
        pd.DataFrame({feature: values}),
        bins=kwargs.pop("bins", 10),
        **kwargs,
    )["features"][feature]


def test_identical_distribution_has_zero_psi():
    reference = np.arange(1, 11, dtype=float)
    config = _baseline(reference.tolist())
    assert calculate_feature_psi(reference, config) == pytest.approx(0.0, abs=1e-12)


def test_changed_numerical_distribution_has_positive_psi():
    config = _baseline(np.arange(1, 11, dtype=float).tolist())
    assert calculate_feature_psi(np.array([8.0, 9.0, 10.0]), config) > 0


def test_js_divergence_uses_fixed_reference_bins():
    config = _baseline([1.0, 2.0, 3.0, 4.0])
    assert calculate_feature_js(np.array([1.0, 2.0, 3.0, 4.0]), config) == pytest.approx(0.0)
    assert calculate_feature_js(np.array([10.0, 11.0, 12.0]), config) > 0


def test_constant_reference_equal_is_zero_and_changed_is_positive():
    config = _baseline([4.0] * 8)
    assert calculate_feature_psi(np.array([4.0] * 3), config) == pytest.approx(0.0)
    assert calculate_feature_psi(np.array([5.0] * 3), config) > 0


def test_constant_zero_is_not_misclassified_as_sparse():
    config = _baseline([0.0] * 8)
    assert config["distribution_type"] == "constant"
    assert calculate_feature_psi(np.array([1.0]), config) > 0


def test_sparse_feature_uses_zero_nonzero_distribution():
    config = _baseline([0.0] * 8 + [0.5, 0.7], feature="tfidf_term:great")
    assert config["distribution_type"] == "sparse_binary"
    assert calculate_feature_psi(np.array([0.0, 1.0]), config) > 0


def test_empty_current_values_are_rejected():
    config = _baseline([1.0, 2.0])
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_feature_psi(np.array([]), config)


def test_nonfinite_reference_and_current_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        _baseline([1.0, np.inf])
    config = _baseline([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        calculate_feature_psi(np.array([np.nan]), config)


def test_collapsed_quantile_boundaries_and_out_of_range_values_are_supported():
    config = _baseline([1.0] * 8 + [2.0, 3.0], bins=10)
    assert config["boundaries"][0] == -np.inf
    assert config["boundaries"][-1] == np.inf
    assert calculate_feature_psi(np.array([-100.0, 100.0]), config) > 0


def test_evaluation_does_not_mutate_saved_boundaries():
    reference = pd.DataFrame({"score": np.arange(10, dtype=float)})
    baseline = build_psi_baseline(reference)
    before = copy.deepcopy(baseline["features"]["score"]["boundaries"])
    for _ in range(2):
        evaluate_drift(pd.DataFrame({"score": np.arange(10, dtype=float)}), baseline)
    assert baseline["features"]["score"]["boundaries"] == before


def test_serialization_round_trip_preserves_evaluation(tmp_path):
    reference = pd.DataFrame({"score": np.arange(10, dtype=float)})
    baseline = build_psi_baseline(reference, model_version=4, dataset_version="sst2_v002")
    path = tmp_path / "nlp_psi_baseline.joblib"
    save_psi_baseline(baseline, path)
    loaded = load_psi_baseline(path)
    current = pd.DataFrame({"score": np.arange(10, dtype=float)})
    assert evaluate_drift(current, baseline) == evaluate_drift(current, loaded)
    with pytest.raises(FileExistsError):
        save_psi_baseline(baseline, path)


def test_predicted_sentiment_is_categorical_without_ks():
    baseline = build_psi_baseline(
        pd.DataFrame({"predicted_sentiment": ["0", "1", "1", "0"]}),
        categorical_classes={"predicted_sentiment": ["0", "1"]},
    )
    config = baseline["features"]["predicted_sentiment"]
    assert config["distribution_type"] == "categorical"
    assert calculate_feature_ks(np.array(["0", "1"]), config) == (None, None)
    assert calculate_feature_psi(np.array(["1", "1"]), config) > 0


def test_ks_uses_raw_reference_samples():
    config = _baseline([0.0, 1.0, 2.0, 3.0])
    statistic, pvalue = calculate_feature_ks(np.array([8.0, 9.0, 10.0]), config)
    assert statistic is not None and statistic > 0
    assert pvalue is not None and pvalue < 1


def test_feature_extraction_uses_fitted_pipeline_and_positive_class(fitted_model):
    texts = pd.Series(["good movie", None, ""])
    features = extract_nlp_monitoring_features(texts, fitted_model, positive_label=2)
    assert list(features.columns) == [
        "word_count", "char_count", "avg_word_length", "tfidf_nonzero_count",
        "positive_probability", "predicted_sentiment",
    ]
    assert features.loc[1, "word_count"] == 0
    assert features.loc[2, "avg_word_length"] == 0
    matrix = fitted_model.named_steps["tfidf"].transform(["good movie", "", ""])
    expected = fitted_model.named_steps["lr"].predict_proba(matrix)[:, 1]
    np.testing.assert_allclose(features["positive_probability"], expected)


def test_optional_tfidf_terms_are_sparse_presence_features(fitted_model):
    indices = np.argsort(
        np.abs(fitted_model.named_steps["lr"].coef_[0])
    )[::-1][:2]
    vocabulary = fitted_model.named_steps["tfidf"].get_feature_names_out()
    selected_terms = [
        {"term": str(vocabulary[index]), "index": int(index)}
        for index in indices
    ]
    presence = extract_selected_tfidf_presence(
        ["good movie", "bad film"],
        fitted_model,
        selected_terms,
    )
    assert list(presence.columns) == [
        f"tfidf_term:{term['term']}" for term in selected_terms
    ]
    assert set(np.unique(presence.to_numpy())).issubset({0, 1})


def test_evaluator_rejects_model_baseline_version_mismatch():
    baseline = build_psi_baseline(
        pd.DataFrame({"score": [0.0, 1.0]}),
        model_version=2,
    )
    with pytest.raises(ValueError, match="version mismatch"):
        evaluate_drift(pd.DataFrame({"score": [0.0, 1.0]}), baseline, model_version=3)


def test_evaluator_reports_categorical_ks_as_null():
    reference = pd.DataFrame({
        "score": [1.0, 2.0, 3.0, 4.0],
        "predicted_sentiment": ["0", "0", "1", "1"],
    })
    baseline = build_psi_baseline(
        reference,
        categorical_classes={"predicted_sentiment": ["0", "1"]},
    )
    result = evaluate_drift(reference, baseline)
    category = next(row for row in result["feature_results"] if row["feature"] == "predicted_sentiment")
    assert category["ks"] is None
    assert category["ks_pvalue"] is None


def test_baseline_loading_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.joblib"
    baseline = build_psi_baseline(pd.DataFrame({"score": [1.0, 2.0]}))
    baseline["schema_version"] = 999
    joblib.dump(baseline, path)
    with pytest.raises(ValueError, match="Unsupported PSI baseline schema"):
        load_psi_baseline(path)


def test_application_imports_with_expected_routes():
    from app.main import app

    routes = {(route.path, tuple(sorted(route.methods or []))) for route in app.routes}
    assert ("/api/drift/evaluate", ("POST",)) in routes
    assert ("/api/models/train", ("POST",)) in routes


def test_drift_injection_and_reset_leave_baseline_unchanged(tmp_path, monkeypatch):
    from app import main

    dataset_dir = tmp_path / "datasets" / "sst2_v001"
    model_dir = tmp_path / "models" / "sst2" / "model_v001"
    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(main, "WORKSPACE", tmp_path)
    monkeypatch.setattr(main, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(main, "MODELS", tmp_path / "models")
    monkeypatch.setattr(main, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")

    reference = pd.DataFrame({
        "sentence": ["great film", "awful movie", "nice story", "boring plot"],
        "label": [1, 0, 1, 0],
    })
    reference.to_csv(dataset_dir / "reference.csv", index=False)
    reference.to_csv(dataset_dir / "current.csv", index=False)
    baseline_file = model_dir / "nlp_psi_baseline.joblib"
    baseline_file.write_bytes(b"baseline is immutable")
    original_baseline = baseline_file.read_bytes()
    main.STATE_PATH.write_text(json.dumps({
        "active_dataset": {"name": "sst2", "version": 1},
        "active_model": {"dataset_name": "sst2", "version": 1},
        "stage": "dataset_versioned",
        "drift": None,
        "last_evaluation": None,
    }), encoding="utf-8")

    main.inject_drift({"drift_text": "unexpected vocabulary", "drift_ratio": 0.5})
    injected = pd.read_csv(dataset_dir / "current.csv")
    assert not injected["sentence"].equals(reference["sentence"])
    assert injected["label"].equals(reference["label"])
    assert baseline_file.read_bytes() == original_baseline

    main.reset_current_data()
    restored = pd.read_csv(dataset_dir / "current.csv")
    pd.testing.assert_frame_equal(restored, reference)
    assert baseline_file.read_bytes() == original_baseline


def test_evaluation_updates_metrics_logs_reports_and_reuses_baseline(tmp_path, monkeypatch):
    from app import main
    from app.monitoring.nlp_features import extract_nlp_monitoring_features

    dataset_dir = tmp_path / "datasets" / "sst2_v001"
    model_dir = tmp_path / "models" / "sst2" / "model_v001"
    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(main, "WORKSPACE", tmp_path)
    monkeypatch.setattr(main, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(main, "MODELS", tmp_path / "models")
    monkeypatch.setattr(main, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")

    frame = pd.DataFrame({
        "sentence": [
            "great lovely movie", "excellent good film", "wonderful story",
            "nice acting", "awful boring movie", "terrible bad film",
            "poor story", "dreadful acting",
        ],
        "label": [1, 1, 1, 1, 0, 0, 0, 0],
    })
    frame.to_csv(dataset_dir / "reference.csv", index=False)
    frame.to_csv(dataset_dir / "current.csv", index=False)
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(random_state=2)),
    ]).fit(frame["sentence"], frame["label"])
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    baseline = build_psi_baseline(
        extract_nlp_monitoring_features(frame["sentence"], model),
        model_version=1,
        dataset_version="sst2_v001",
        categorical_classes={"predicted_sentiment": ["0", "1"]},
    )
    baseline_path = model_dir / "nlp_psi_baseline.joblib"
    save_psi_baseline(baseline, baseline_path)
    original_baseline = baseline_path.read_bytes()

    main.STATE_PATH.write_text(json.dumps({
        "active_dataset": {"name": "sst2", "version": 1},
        "active_model": {
            "dataset_name": "sst2",
            "version": 1,
            "dataset": "sst2_v001",
            "psi_baseline_path": baseline_path.name,
            "psi_baseline_version": 1,
        },
        "stage": "model_trained",
        "drift": None,
        "last_evaluation": None,
    }), encoding="utf-8")

    artifact_paths = []

    class RunContext:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="test-run", run_name="drift-check"))

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    mlflow_stub = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: None,
        start_run=lambda **kwargs: RunContext(),
        data=SimpleNamespace(from_pandas=lambda *args, **kwargs: object()),
        log_input=lambda *args, **kwargs: None,
        log_param=lambda *args, **kwargs: None,
        log_metric=lambda *args, **kwargs: None,
        log_artifact=lambda path, **kwargs: artifact_paths.append(path),
    )
    monkeypatch.setattr(main, "mlflow", mlflow_stub)
    main.PSI_METRIC.labels(feature="stale_previous_feature").set(7.0)

    result = main.evaluate_drift()
    assert result["baseline_version"] == 1
    assert result["monitored_features"] == 6
    assert result["breach"] is False
    assert result["feature_results"]
    assert any(item["feature"] == "predicted_sentiment" and item["ks"] is None for item in result["feature_results"])
    assert baseline_path.read_bytes() == original_baseline
    assert (tmp_path / "reports" / "latest.json").is_file()
    assert "Feature-level results" in (tmp_path / "reports" / "latest.html").read_text(encoding="utf-8")
    assert len(artifact_paths) == 3
    metrics = main.metrics().body.decode("utf-8")
    assert "stale_previous_feature" not in metrics
    assert 'feature="predicted_sentiment"' not in "\n".join(
        line for line in metrics.splitlines() if "ml_drift_ks_statistic" in line
    )
