from __future__ import annotations

import json
import os
import re
import shutil
import threading
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from nltk.stem import PorterStemmer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from app.scr.preprocessing_utils import preprocess_text
from mlflow.models.signature import infer_signature, ModelSignature

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = BASE_DIR / "workspace"
DATASETS = WORKSPACE / "datasets"
MODELS = WORKSPACE / "models"
REPORTS = WORKSPACE / "reports"
STATE_PATH = WORKSPACE / "state.json"

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.20"))

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000")

GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")

GRAFANA_BROWSER_URL = os.getenv("GRAFANA_BROWSER_URL", "http://localhost:3000")
MLFLOW_BROWSER_URL = os.getenv("MLFLOW_BROWSER_URL","http://localhost:5000")
PROMETHEUS_BROWSER_URL = os.getenv("PROMETHEUS_BROWSER_URL","http://localhost:9090")

TOP_DRIFT_FEATURES = int(os.getenv("TOP_DRIFT_FEATURES", "100"))
SEED = 42
LOCK = threading.Lock()

# Use a filesystem-safe internal name. Do not use "stanfordnlp/sst2" as a folder name.
DATASET_DESCRIPTIONS = {
    "sst2": "Stanford Sentiment Treebank SST-2 movie-review sentiment dataset."
}

# -----------------------------------------------------------------------------
# Prometheus metrics
# -----------------------------------------------------------------------------
DATASET_VERSION = Gauge("ml_dataset_version", "Active dataset version number")
MODEL_VERSION = Gauge("ml_model_version", "Active model version number")
PSI_METRIC = Gauge("ml_drift_psi", "Population Stability Index", ["feature"])
KS_METRIC = Gauge("ml_drift_ks_statistic", "KS statistic", ["feature"])
JS_METRIC = Gauge("ml_drift_js_divergence", "JS divergence", ["feature"])
DRIFT_BREACH = Gauge("ml_drift_breach", "1 when PSI exceeds its threshold")
QUALITY_PASS = Gauge("ml_quality_pass", "1 when the latest data quality gate passed")
MODEL_ACCURACY = Gauge("ml_model_accuracy", "Accuracy of the active model")
MODEL_F1 = Gauge("ml_model_f1", "F1 score of the active model")
PREDICTIONS = Counter("ml_predictions_total", "Prediction requests", ["status"])

app = FastAPI(title="ml_work_flow", version="2.0.0")
stemmer = PorterStemmer()


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_dirs() -> None:
    for directory in (WORKSPACE, DATASETS, MODELS, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)


def read_state() -> dict[str, Any]:
    ensure_dirs()
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "active_dataset": None,
        "active_model": None,
        "stage": "ready",
        "drift": None,
        "last_evaluation": None,
    }


def write_state(state: dict[str, Any]) -> None:
    ensure_dirs()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

def load_sst2() -> pd.DataFrame:
    """Load the SST-2 training split used by this project."""
    url = "hf://datasets/stanfordnlp/sst2/data/train-00000-of-00001.parquet"
    try:
        frame = pd.read_parquet(url)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load SST-2 from Hugging Face. Ensure pandas parquet "
                "support and Hugging Face filesystem dependencies are installed. "
                f"Original error: {exc}"
            ),
        ) from exc

    required = ["sentence", "label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"SST-2 is missing columns: {missing}")

    return frame[required].dropna().reset_index(drop=True)


def dataset_frame(name: str) -> pd.DataFrame:
    if name != "sst2":
        raise HTTPException(status_code=400, detail="Unknown dataset. Choose: sst2")
    return load_sst2()


def version_path(dataset_name: str, version: int) -> Path:
    return DATASETS / f"{dataset_name}_v{version:03d}"


def next_dataset_version(dataset_name: str) -> int:
    ensure_dirs()
    versions: list[int] = []
    for path in DATASETS.glob(f"{dataset_name}_v*"):
        try:
            versions.append(int(path.name.rsplit("_v", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(versions, default=0) + 1


def next_model_version(dataset_name: str) -> int:
    model_root = MODELS / dataset_name
    model_root.mkdir(parents=True, exist_ok=True)
    versions: list[int] = []
    for path in model_root.glob("model_v*"):
        try:
            versions.append(int(path.name.rsplit("_v", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(versions, default=0) + 1


def active_dataset_paths() -> tuple[Path, Path, dict[str, Any]]:
    state = read_state()
    active = state.get("active_dataset")
    if not active:
        raise HTTPException(status_code=400, detail="Create a dataset version first")
    directory = version_path(active["name"], active["version"])
    return directory / "reference.csv", directory / "current.csv", active


def active_model_paths() -> tuple[Path, Path, dict[str, Any]]:
    state = read_state()
    active_model = state.get("active_model")
    if not active_model:
        raise HTTPException(status_code=400, detail="Train a model first")

    directory = (
        MODELS
        / active_model["dataset_name"]
        / f"model_v{active_model['version']:03d}"
    )
    return directory / "model.joblib", directory / "tfidf.pkl", active_model


def dataset_catalog() -> list[dict[str, Any]]:
    versions = []
    for manifest in sorted(DATASETS.glob("*_v*/manifest.json")):
        versions.append(json.loads(manifest.read_text(encoding="utf-8")))
    return versions


def model_catalog() -> list[dict[str, Any]]:
    versions = []
    for manifest in sorted(MODELS.rglob("manifest.json")):
        versions.append(json.loads(manifest.read_text(encoding="utf-8")))
    return versions


# -----------------------------------------------------------------------------
# Drift helpers
# -----------------------------------------------------------------------------
def histogram_probabilities(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create comparable normalized histograms robust to sparse/constant TF-IDF data."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size == 0 or current.size == 0:
        raise ValueError("Cannot calculate drift on an empty feature")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))

    # Sparse TF-IDF columns are often constant/mostly zero. Fall back to a
    # shared min/max range if reference quantiles collapse to one value.
    if edges.size < 2:
        combined_min = float(min(reference.min(), current.min()))
        combined_max = float(max(reference.max(), current.max()))
        if np.isclose(combined_min, combined_max):
            # Identical constant distributions -> no drift.
            return np.array([1.0]), np.array([1.0])
        edges = np.linspace(combined_min, combined_max, bins + 1)

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected = np.histogram(reference, bins=edges)[0].astype(float)
    actual = np.histogram(current, bins=edges)[0].astype(float)

    expected /= max(expected.sum(), 1.0)
    actual /= max(actual.sum(), 1.0)

    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected /= expected.sum()
    actual /= actual.sum()
    return expected, actual


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    expected, actual = histogram_probabilities(reference, current, bins=bins)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def quality_gate(reference: pd.DataFrame, current: pd.DataFrame) -> int:
    if list(reference.columns) != list(current.columns):
        raise HTTPException(status_code=400, detail="Schema check failed: columns differ")
    if reference.empty or current.empty:
        raise HTTPException(status_code=400, detail="Completeness check failed: empty dataset")

    required = {"sentence", "label"}
    if not required.issubset(reference.columns):
        raise HTTPException(
            status_code=400,
            detail="Schema check failed: expected 'sentence' and 'label' columns",
        )

    if reference["sentence"].isna().any() or current["sentence"].isna().any():
        raise HTTPException(status_code=400, detail="Missing text values were found")
    if reference["label"].isna().any() or current["label"].isna().any():
        raise HTTPException(status_code=400, detail="Missing label values were found")

    duplicate_rows = int(reference.duplicated().sum() + current.duplicated().sum())
    QUALITY_PASS.set(1)
    return duplicate_rows


def important_tfidf_indices(model: Pipeline, top_n: int) -> np.ndarray:
    """Return indices of TF-IDF terms with largest absolute LR coefficients."""
    classifier: LogisticRegression = model.named_steps["lr"]
    coefficients = np.asarray(classifier.coef_)
    importance = np.max(np.abs(coefficients), axis=0)
    top_n = min(top_n, importance.size)
    return np.argsort(importance)[-top_n:][::-1]


# -----------------------------------------------------------------------------
# Grafana dashboard
# -----------------------------------------------------------------------------
def grafana_dashboard() -> dict[str, Any]:
    datasource = {"type": "prometheus", "uid": "prometheus"}
    return {
        "uid": "ml-work-flow",
        "title": "ml_work_flow control center",
        "tags": ["ml_work_flow", "nlp", "drift", "models", "datasets"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": [
            {"id": 1, "type": "stat", "title": "Dataset version", "gridPos": {"h": 5, "w": 6, "x": 0, "y": 0}, "datasource": datasource, "targets": [{"refId": "A", "expr": "ml_dataset_version"}]},
            {"id": 2, "type": "stat", "title": "Model version", "gridPos": {"h": 5, "w": 6, "x": 6, "y": 0}, "datasource": datasource, "targets": [{"refId": "A", "expr": "ml_model_version"}]},
            {"id": 3, "type": "stat", "title": "Model accuracy", "gridPos": {"h": 5, "w": 6, "x": 12, "y": 0}, "datasource": datasource, "targets": [{"refId": "A", "expr": "ml_model_accuracy * 100"}], "fieldConfig": {"defaults": {"unit": "percent", "decimals": 1}}},
            {"id": 4, "type": "stat", "title": "Drift breach", "gridPos": {"h": 5, "w": 6, "x": 18, "y": 0}, "datasource": datasource, "targets": [{"refId": "A", "expr": "ml_drift_breach"}]},
            {"id": 5, "type": "timeseries", "title": "Drift metrics", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6}, "datasource": datasource, "targets": [{"refId": "A", "expr": "ml_drift_psi", "legendFormat": "PSI {{feature}}"}, {"refId": "B", "expr": "ml_drift_ks_statistic", "legendFormat": "KS {{feature}}"}, {"refId": "C", "expr": "ml_drift_js_divergence", "legendFormat": "JS {{feature}}"}]},
            {"id": 6, "type": "timeseries", "title": "Quality and model", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6}, "datasource": datasource, "targets": [{"refId": "A", "expr": "ml_quality_pass", "legendFormat": "quality"}, {"refId": "B", "expr": "ml_model_f1", "legendFormat": "F1"}]},
        ],
    }


# -----------------------------------------------------------------------------
# API routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (
        DASHBOARD_HTML
        .replace("__GRAFANA_BROWSER_URL__", GRAFANA_BROWSER_URL)
        .replace("__MLFLOW_BROWSER_URL__", MLFLOW_BROWSER_URL)
        .replace("__PROMETHEUS_BROWSER_URL__", PROMETHEUS_BROWSER_URL)
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/state")
def state() -> dict[str, Any]:
    current_state = read_state()
    return {
        "state": current_state,
        "datasets": dataset_catalog(),
        "models": model_catalog(),
        "datasets_available": DATASET_DESCRIPTIONS,
        "active_features": (current_state.get("active_dataset") or {}).get("feature_names", []),
    }


@app.post("/api/datasets/version")
def create_dataset_version(payload: dict[str, str]) -> dict[str, Any]:
    name = payload.get("name", "sst2")
    frame = dataset_frame(name)

    with LOCK:
        version = next_dataset_version(name)
        directory = version_path(name, version)
        directory.mkdir(parents=True, exist_ok=False)

    frame.to_csv(directory / "reference.csv", index=False)
    frame.to_csv(directory / "current.csv", index=False)

    manifest = {
        "name": name,
        "version": version,
        "rows": len(frame),
        "features": 1,
        "feature_names": ["sentence"],
        "target": "label",
        "created_at": now(),
        "description": DATASET_DESCRIPTIONS[name],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    current_state = read_state()
    current_state.update({
        "active_dataset": manifest,
        "active_model": None,
        "stage": "dataset_versioned",
        "drift": None,
        "last_evaluation": None,
    })
    write_state(current_state)

    DATASET_VERSION.set(version)
    MODEL_VERSION.set(0)
    return manifest


@app.post("/api/models/train")
def train_model(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    state = read_state()
    active = state.get("active_dataset")
    if not active:
        raise HTTPException(status_code=400, detail="Create a dataset version before training")

    directory = version_path(active["name"], active["version"])
    frame = pd.read_csv(directory / "reference.csv")
    X = frame["sentence"].fillna("").astype(str)
    y = frame["label"].astype(int)

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_temp,
    )

    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_text,
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
            max_features=20000,
            ngram_range=(1, 2),
        )),
        ("lr", LogisticRegression(random_state=SEED)),
    ])

    param_grid = {
        "lr__C": [0.01, 0.1, 1, 10],
        "lr__penalty": ["l2"],
        "lr__solver": ["lbfgs", "saga"],
        "lr__max_iter": [1000, 2000],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    validation_predictions = grid_search.best_estimator_.predict(X_validation)
    validation_accuracy = float(accuracy_score(y_validation, validation_predictions))

    print("Validation Accutacy: ",validation_accuracy)

    # Refit the selected configuration on train + validation, leaving test untouched.
    X_train_final = pd.concat([X_train, X_validation])
    y_train_final = pd.concat([y_train, y_validation])
    model: Pipeline = grid_search.best_estimator_
    model.fit(X_train_final, y_train_final)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, average="weighted"))

    with LOCK:
        version = next_model_version(active["name"])
        model_directory = MODELS / active["name"] / f"model_v{version:03d}"
        model_directory.mkdir(parents=True, exist_ok=False)

    model_path = model_directory / "model.joblib"
    tfidf_path = model_directory / "tfidf.pkl"
    manifest_path = model_directory / "manifest.json"

    joblib.dump(model, model_path)

    # Save the FITTED TF-IDF transformer from the final fitted pipeline.
    fitted_tfidf: TfidfVectorizer = model.named_steps["tfidf"]
    joblib.dump(fitted_tfidf, tfidf_path)

    manifest = {
        "version": version,
        "dataset_name": active["name"],
        "dataset": f"{active['name']}_v{active['version']:03d}",
        "algorithm": "TF-IDF + LogisticRegression",
        "validation_accuracy": validation_accuracy,
        "accuracy": accuracy,
        "f1": f1,
        "best_params": grid_search.best_params_,
        "created_at": now(),
        "path": str(model_path),
        "tfidf_path": str(tfidf_path),
    }

    manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8"
        )
    # (model_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ml_work_flow_training")
    registered_model = f"ml_work_flow_model_{active['name']}"

    with mlflow.start_run(run_name=f"model-v{version:03d}") as run:
        mlflow.log_param("dataset_version", manifest["dataset"])
        mlflow.log_param("algorithm", manifest["algorithm"])
        mlflow.log_params({f"best_{k}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("validation_accuracy", validation_accuracy)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)

        training_dataset = mlflow.data.from_pandas(
            frame,
            source=str(directory / "reference.csv"),
            name=manifest["dataset"],
            targets="label",
        )

        mlflow.log_input(training_dataset, context="training")

        example_inputs = X_test.head(5).tolist()
        example_predictions = model.predict(example_inputs)
        
        signature = infer_signature(
            example_inputs,
            example_predictions
        )
        
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=example_inputs,
            registered_model_name=registered_model,
            serialization_format="skops",
            skops_trusted_types=[
                "app.scr.preprocessing_utils.preprocess_text"
            ],
            
        )

        mlflow.set_tag("logged_model_id", model_info.model_id)
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model"
        )
        
        artifact_example = X_test.head(10).to_frame(name="sentence")
        example_path = model_directory / "input_example.csv"

        artifact_example.to_csv(example_path, index=False)
        mlflow.log_artifact(str(example_path),artifact_path="examples")
                
        mlflow.log_artifact(str(tfidf_path), artifact_path="feature_engineering")
        mlflow.log_dict(manifest, "model_manifest.json")

    registry_versions = mlflow.MlflowClient(tracking_uri=MLFLOW_URI).search_model_versions(
        f"name='{registered_model}'"
    )
    registry_version = next(
        (item.version for item in registry_versions if item.run_id == run.info.run_id),
        None,
    )

    manifest.update({
        "mlflow_experiment": "ml_work_flow_training",
        "mlflow_run_id": run.info.run_id,
        "registered_model": registered_model,
        "registered_model_version": registry_version,
        "model_uri": model_info.model_uri,
    })
    (model_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    state.update({"active_model": manifest, "stage": "model_trained"})
    write_state(state)
    MODEL_VERSION.set(version)
    MODEL_ACCURACY.set(accuracy)
    MODEL_F1.set(f1)
    return manifest


@app.post("/api/drift/inject")
def inject_drift(payload: dict[str, Any]) -> dict[str, Any]:
    """Inject controlled text drift into current.csv only."""
    reference_path, current_path, active = active_dataset_paths()
    reference = pd.read_csv(reference_path)
    current = reference.copy()

    text_column = payload.get("feature", "sentence")
    drift_text = str(payload.get("drift_text", "excellent amazing wonderful"))
    drift_ratio = float(payload.get("drift_ratio", 0.30))

    if text_column != "sentence" or text_column not in reference.columns:
        raise HTTPException(status_code=400, detail="Choose the 'sentence' feature")
    if not 0 < drift_ratio <= 1:
        raise HTTPException(status_code=400, detail="drift_ratio must be > 0 and <= 1")
    if not drift_text.strip():
        raise HTTPException(status_code=400, detail="drift_text cannot be empty")

    n_drift = max(1, int(len(current) * drift_ratio))
    drift_indices = current.sample(n=n_drift, random_state=SEED).index
    current.loc[drift_indices, text_column] = (
        current.loc[drift_indices, text_column].astype(str) + " " + drift_text
    )
    current.to_csv(current_path, index=False)

    state = read_state()
    state.update({
        "stage": "drift_injected",
        "drift": {
            "feature": text_column,
            "drift_text": drift_text,
            "drift_ratio": drift_ratio,
            "samples_modified": n_drift,
            "injected_at": now(),
        },
        "last_evaluation": None,
    })
    write_state(state)

    return {
        "stage": "drift_injected",
        "dataset": f"{active['name']}_v{active['version']:03d}",
        "feature": text_column,
        "drift_text": drift_text,
        "drift_ratio": drift_ratio,
        "samples_modified": n_drift,
    }


@app.post("/api/drift/reset")
def reset_current_data() -> dict[str, str]:
    reference_path, current_path, _ = active_dataset_paths()
    shutil.copyfile(reference_path, current_path)

    state = read_state()
    state.update({"stage": "dataset_ready", "drift": None, "last_evaluation": None})
    write_state(state)
    DRIFT_BREACH.set(0)
    QUALITY_PASS.set(0)
    return {"stage": "dataset_ready", "message": "Current data restored to the reference snapshot"}


@app.post("/api/drift/evaluate")
def evaluate_drift() -> dict[str, Any]:
    reference_path, current_path, active = active_dataset_paths()
    model_path, tfidf_path, active_model = active_model_paths()

    # Ensure the active model belongs to the active dataset version.
    expected_dataset = f"{active['name']}_v{active['version']:03d}"
    if active_model.get("dataset") != expected_dataset:
        raise HTTPException(
            status_code=400,
            detail="The active model was not trained on the active dataset version. Train a model first.",
        )

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    QUALITY_PASS.set(0)
    duplicate_rows = quality_gate(reference, current)

    state = read_state()
    text_column = (state.get("drift") or {}).get("feature") or "sentence"
    if text_column not in reference.columns:
        raise HTTPException(status_code=400, detail=f"Text column '{text_column}' not found")

    if not model_path.exists() or not tfidf_path.exists():
        raise HTTPException(status_code=500, detail="Active model or TF-IDF artifact was not found")

    model: Pipeline = joblib.load(model_path)
    tfidf: TfidfVectorizer = joblib.load(tfidf_path)

    X_reference = tfidf.transform(reference[text_column].fillna("").astype(str))
    X_current = tfidf.transform(current[text_column].fillna("").astype(str))
    feature_names = tfidf.get_feature_names_out()

    monitored_indices = important_tfidf_indices(model, TOP_DRIFT_FEATURES)
    feature_results: list[dict[str, Any]] = []

    for index in monitored_indices:
        feature_name = str(feature_names[index])
        reference_values = X_reference[:, index].toarray().ravel()
        current_values = X_current[:, index].toarray().ravel()

        expected, actual = histogram_probabilities(reference_values, current_values)
        psi_value = psi(reference_values, current_values)
        ks_value, ks_pvalue = ks_2samp(reference_values, current_values)
        js_value = float(jensenshannon(expected, actual, base=2) ** 2)

        feature_results.append({
            "feature": feature_name,
            "psi": float(psi_value),
            "ks": float(ks_value),
            "ks_pvalue": float(ks_pvalue),
            "js": js_value,
            "breach": bool(psi_value > PSI_THRESHOLD),
        })

    if not feature_results:
        raise HTTPException(status_code=500, detail="No TF-IDF features available for drift evaluation")

    feature_results.sort(key=lambda item: item["psi"], reverse=True)
    most_drifted = feature_results[0]
    breached_features = sum(item["breach"] for item in feature_results)
    breach_ratio = breached_features / len(feature_results)

    # Overall breach retains the assignment's PSI threshold rule: if the most
    # drifted monitored feature exceeds the threshold, flag a breach.
    breach = bool(most_drifted["psi"] > PSI_THRESHOLD)

    PSI_METRIC.labels(feature=most_drifted["feature"]).set(most_drifted["psi"])
    KS_METRIC.labels(feature=most_drifted["feature"]).set(most_drifted["ks"])
    JS_METRIC.labels(feature=most_drifted["feature"]).set(most_drifted["js"])
    DRIFT_BREACH.set(int(breach))

    result = {
        "dataset": expected_dataset,
        "model_version": active_model["version"],
        "input_feature": text_column,
        "monitored_tfidf_features": len(feature_results),
        "most_drifted_tfidf_feature": most_drifted["feature"],
        "psi": most_drifted["psi"],
        "ks": most_drifted["ks"],
        "ks_pvalue": most_drifted["ks_pvalue"],
        "js": most_drifted["js"],
        "threshold": PSI_THRESHOLD,
        "breached_features": breached_features,
        "breach_ratio": breach_ratio,
        "breach": breach,
        "duplicate_rows_warning": duplicate_rows,
        "evaluated_at": now(),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    latest_json = REPORTS / "latest.json"
    latest_html = REPORTS / "latest.html"
    feature_csv = REPORTS / "drift_features.csv"

    latest_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    latest_html.write_text(report_html(result), encoding="utf-8")
    pd.DataFrame(feature_results).to_csv(feature_csv, index=False)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ml_work_flow_drift_evaluation")

    with mlflow.start_run(run_name="drift-breach" if breach else "drift-check") as run:
        reference_dataset = mlflow.data.from_pandas(
            reference,
            source=str(reference_path),
            name=f"{expected_dataset}_reference",
        )
        current_dataset = mlflow.data.from_pandas(
            current,
            source=str(current_path),
            name=f"{expected_dataset}_current",
        )
        mlflow.log_input(reference_dataset, context="drift_reference")
        mlflow.log_input(current_dataset, context="drift_current")
        mlflow.log_param("dataset_version", expected_dataset)
        mlflow.log_param("model_version", active_model["version"])
        mlflow.log_param("text_column", text_column)
        mlflow.log_param("most_drifted_tfidf_feature", most_drifted["feature"])
        mlflow.log_param("monitored_tfidf_features", len(feature_results))
        mlflow.log_param("threshold", PSI_THRESHOLD)
        mlflow.log_metric("psi", most_drifted["psi"])
        mlflow.log_metric("ks", most_drifted["ks"])
        mlflow.log_metric("ks_pvalue", most_drifted["ks_pvalue"])
        mlflow.log_metric("js", most_drifted["js"])
        mlflow.log_metric("breached_features", breached_features)
        mlflow.log_metric("breach_ratio", breach_ratio)
        mlflow.log_artifact(str(latest_html), artifact_path="drift-report")
        mlflow.log_artifact(str(latest_json), artifact_path="drift-report")
        mlflow.log_artifact(str(feature_csv), artifact_path="drift-report")

    result.update({
        "mlflow_experiment": "ml_work_flow_drift_evaluation",
        "mlflow_run_id": run.info.run_id,
        "mlflow_run_name": run.info.run_name,
    })

    state.update({"stage": "evaluated", "last_evaluation": result})
    write_state(state)
    return result


@app.post("/api/reset")
def reset_workspace() -> dict[str, str]:
    for directory in (DATASETS, MODELS, REPORTS):
        if directory.exists():
            shutil.rmtree(directory)
    STATE_PATH.unlink(missing_ok=True)
    ensure_dirs()
    for metric in (DATASET_VERSION, MODEL_VERSION, DRIFT_BREACH, QUALITY_PASS, MODEL_ACCURACY, MODEL_F1):
        metric.set(0)
    return {"stage": "ready", "message": "Workspace reset. No dataset or model versions remain."}


@app.post("/api/grafana/dashboard")
def create_grafana_dashboard() -> dict[str, str]:
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            json={"dashboard": grafana_dashboard(), "folderId": 0, "overwrite": True},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Grafana dashboard creation failed: {exc}") from exc

    result = response.json()
    uid = result.get("uid", "ml-work-flow")
    return {
        "message": "Grafana dashboard created or updated",
        "uid": uid,
        "url": f"{GRAFANA_BROWSER_URL}/d/{uid}",
    }


@app.post("/api/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    model_path, _, active_model = active_model_paths()
    if not model_path.exists():
        raise HTTPException(status_code=500, detail="Active model artifact was not found")

    sentence = payload.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Provide a non-empty 'sentence' string")

    try:
        model: Pipeline = joblib.load(model_path)
        prediction = int(model.predict([sentence])[0])
        probabilities = model.predict_proba([sentence])[0]
        probability = float(np.max(probabilities))
        PREDICTIONS.labels(status="success").inc()
        return {
            "prediction": prediction,
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": round(probability, 4),
            "model_version": active_model["version"],
        }
    except Exception as exc:
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/report", response_class=HTMLResponse)
def report() -> HTMLResponse:
    path = REPORTS / "latest.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluate drift first")
    return HTMLResponse(path.read_text(encoding="utf-8"))


def report_html(result: dict[str, Any]) -> str:
    status = "DRIFT BREACH" if result["breach"] else "NORMAL"
    rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in result.items()
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ml_work_flow report</title>
<style>
body{{font-family:system-ui;max-width:850px;margin:40px auto;background:#f5f7fb;color:#172033}}
main{{background:white;border:1px solid #dbe2ef;border-radius:12px;padding:28px}}
.status{{font-weight:800}}
table{{width:100%;border-collapse:collapse}}
td{{padding:10px;border-bottom:1px solid #e5e9f2}}
</style>
</head>
<body><main><h1>ml_work_flow drift report</h1><p class="status">{status}</p><table>{rows}</table></main></body>
</html>"""


DASHBOARD_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ml_work_flow</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f3f6fb}
*{box-sizing:border-box}body{margin:0}.top{background:#132238;color:white;padding:28px 6vw 34px}
.top h1{margin:0 0 8px;font-size:32px}.top p{margin:0;color:#b8c8db}.wrap{max-width:1200px;margin:26px auto;padding:0 24px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}.card{background:white;border:1px solid #dbe2ef;border-radius:12px;padding:20px;box-shadow:0 8px 24px #192b4610}
.span-4{grid-column:span 4}.span-6{grid-column:span 6}.span-12{grid-column:span 12}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#56708e}
.value{font-size:25px;font-weight:800;margin:8px 0}.muted{color:#64748b}button{border:0;border-radius:8px;padding:12px 16px;background:#0f766e;color:white;font-weight:750;cursor:pointer;margin:4px}
button.secondary{background:#e7eef7;color:#1d3654}input,select{padding:11px;border:1px solid #b8c7d9;border-radius:7px;background:white;margin:4px;min-width:180px}
pre{background:#101b2e;color:#d9e8f7;padding:16px;border-radius:8px;min-height:110px;white-space:pre-wrap}a{color:#0f766e;font-weight:700;margin-right:18px}
@media(max-width:800px){.span-4,.span-6{grid-column:span 12}}
</style>
</head>
<body>
<header class="top"><h1>ml_work_flow</h1><p>Version SST-2. Train sentiment model. Inject text drift. Evaluate and log evidence.</p></header>
<main class="wrap"><section class="grid">
<div class="card span-4"><div class="eyebrow">Current dataset</div><div id="dataset" class="value">None</div><div id="datasetInfo" class="muted">Create a version below</div></div>
<div class="card span-4"><div class="eyebrow">Current model</div><div id="model" class="value">None</div><div id="modelInfo" class="muted">Train after versioning data</div></div>
<div class="card span-4"><div class="eyebrow">Latest drift</div><div id="drift" class="value">Not evaluated</div><div id="driftInfo" class="muted">PSI threshold configured on server</div></div>

<div class="card span-6"><div class="eyebrow">1. Dataset versioning</div><h2>SST-2</h2><select id="datasetChoice"><option value="sst2">Stanford SST-2</option></select><button onclick="createDataset()">Create dataset version</button><p class="muted">Creates immutable reference.csv and mutable current.csv snapshots.</p></div>
<div class="card span-6"><div class="eyebrow">2. Model versioning</div><h2>TF-IDF + Logistic Regression</h2><button onclick="trainModel()">Train and register model</button><p class="muted">Tunes Logistic Regression, evaluates on a held-out test set, saves the fitted TF-IDF separately, and logs to MLflow.</p></div>

<div class="card span-12"><div class="eyebrow">3. Controlled NLP drift</div><h2>Inject text drift</h2>
<select id="featureChoice"><option value="sentence">sentence</option></select>
<input id="driftText" type="text" value="excellent amazing wonderful" size="32">
<input id="driftRatio" type="number" value="0.30" min="0.01" max="1" step="0.05">
<button onclick="injectDrift()">Inject drift</button>
<button onclick="run(()=>api('/api/drift/evaluate',{method:'POST'}))">Evaluate quality and drift</button>
<button class="secondary" onclick="resetCurrent()">Restore current data</button>
<p class="muted">Drift ratio is the fraction of reviews modified. Evaluation uses the saved fitted TF-IDF transformer and the top model-important terms.</p></div>

<div class="card span-12"><div class="eyebrow">4. Prediction</div><h2>Test sentiment</h2>
<input id="predictionText" type="text" value="this movie was excellent" size="50"><button onclick="predictSentiment()">Predict</button></div>

<div class="card span-12"><div class="eyebrow">Evidence and services</div>
<button onclick="run(()=>api('/api/grafana/dashboard',{method:'POST'}))">Create Grafana dashboard</button>
<a href="__GRAFANA_BROWSER_URL__" target="_blank">Grafana</a><a href="__MLFLOW_BROWSER_URL__" target="_blank">MLflow</a><a href="__PROMETHEUS_BROWSER_URL__" target="_blank">Prometheus</a><a href="/report" target="_blank">Latest report</a><a href="/docs" target="_blank">Control API</a>
<button class="secondary" onclick="resetAll()">Reset entire workspace</button><pre id="log">Ready.</pre></div>
</section></main>
<script>
const log=document.getElementById('log');
async function api(path,options={}){const response=await fetch(path,{headers:{'Content-Type':'application/json'},...options});const text=await response.text();let data;try{data=JSON.parse(text)}catch{data={detail:text||'Empty server response'}}if(!response.ok)throw new Error(data.detail||'Request failed');return data}
function show(data){log.textContent='Success\n'+JSON.stringify(data,null,2);refresh()}
async function run(fn){log.textContent='Working...';try{show(await fn())}catch(error){log.textContent='Error\n'+error.message}}
async function refresh(){const data=await api('/api/state');const state=data.state;document.getElementById('dataset').textContent=state.active_dataset?state.active_dataset.name+'_v'+String(state.active_dataset.version).padStart(3,'0'):'None';document.getElementById('datasetInfo').textContent=state.active_dataset?state.active_dataset.description:'Create a version';document.getElementById('model').textContent=state.active_model?'model_v'+String(state.active_model.version).padStart(3,'0'):'None';document.getElementById('modelInfo').textContent=state.active_model?'Accuracy '+(state.active_model.accuracy*100).toFixed(1)+'% | F1 '+(state.active_model.f1*100).toFixed(1)+'%':'Train a model';document.getElementById('drift').textContent=state.last_evaluation?(state.last_evaluation.breach?'BREACH':'NORMAL'):'Not evaluated';document.getElementById('driftInfo').textContent=state.last_evaluation?'PSI '+state.last_evaluation.psi.toFixed(3)+' | '+state.last_evaluation.most_drifted_tfidf_feature:'PSI threshold configured on server'}
async function createDataset(){run(()=>api('/api/datasets/version',{method:'POST',body:JSON.stringify({name:'sst2'})}))}
async function trainModel(){run(()=>api('/api/models/train',{method:'POST',body:'{}'}))}
async function injectDrift(){run(()=>api('/api/drift/inject',{method:'POST',body:JSON.stringify({feature:'sentence',drift_text:document.getElementById('driftText').value,drift_ratio:Number(document.getElementById('driftRatio').value)})}))}
async function predictSentiment(){run(()=>api('/api/predict',{method:'POST',body:JSON.stringify({sentence:document.getElementById('predictionText').value})}))}
async function resetCurrent(){run(()=>api('/api/drift/reset',{method:'POST'}))}
async function resetAll(){if(confirm('Delete all dataset and model versions?'))run(()=>api('/api/reset',{method:'POST'}))}
refresh();setInterval(refresh,5000);
</script>
</body></html>"""


ensure_dirs()
